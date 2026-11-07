"""First load the Titanic datasets and print basic data checks."""
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

# Allow the data directory to be specified when running the script.
parser = argparse.ArgumentParser(
    description="Load Titanic data and predict passenger survival."
)
parser.add_argument(
    "--data-dir",
    type=Path,
    default=Path(__file__).resolve().parent.parent / "data",
    help="Directory containing train.csv and test.csv.",
)
args = parser.parse_args()
data_dir = args.data_dir.resolve()
# Check that both input files exist before reading them.
train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
for path in (train_path, test_path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing file: {path}. "
            "Download train.csv and test.csv and place them "
            "in the data directory. "
            "Use --data-dir to specify its location."
        )

# Then read the original CSV files without modifying them.
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print("Step 1: Load Titanic data")
print(f"Data directory: {data_dir}")
print(f"Training data: {train.shape[0]} rows, {train.shape[1]} columns")
print(f"Test data: {test.shape[0]} rows, {test.shape[1]} columns")
print("\nTraining columns:")
print(train.columns.tolist())
print("\nTest columns:")
print(test.columns.tolist())
print("\nMissing values in training data:")
print(train.isna().sum().to_string())
print("\nMissing values in test data:")
print(test.isna().sum().to_string())
if "Survived" not in train.columns:
    raise ValueError("train.csv must contain the Survived target column.")
print("\nTraining survival counts:")
print(train["Survived"].value_counts().sort_index().to_string())
if "Survived" not in test.columns:
    print(
        "\nTest accuracy cannot be calculated directly: test.csv does not contain true Survived labels."
    )
print("\nData loading and basic checks completed.")

# Select the features used to predict survival.
print("\nStep 2: Select features")
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]
features = numeric_features + categorical_features
# Then check that the selected columns exist in both datasets.
for name, dataset in (("train.csv", train), ("test.csv", test)):
    missing_columns = [
        column for column in features if column not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{name} is missing required columns: {missing_columns}"
        )
X = train[features].copy()
y = train["Survived"].copy()
X_test = test[features].copy()
print(f"Selected features: {features}")
print("Target: Survived (0 means did not survive, 1 means survived)")
print("ALso excluded from model: PassengerId, Name, Ticket, Cabin")
print("No passenger rows have been removed.")
# Split before fitting any preprocessing steps so that we can avoid data leakage
print("\n3: Create training and validation subsets")
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
print(f"Training subset: {len(X_train)} passengers")
print(f"Validation subset: {len(X_valid)} passengers")
print("Split: 80% training, 20% validation and random_state=42")
print("I will handle missing values in the next preprocessing step.")

# Define preprocessing for numerical and categorical features.
print("\nStep 4: Configure preprocessing")
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)
#for missing values, I will fill them with traning subset medians and then standardize it
print("Numerical missing values: fill with training-subset medians.")
print("Numerical features: standardize using training-subset statistics.")
#same for categorical missing values, I will fill them with training subset modes 
print("Categorical missing values: fill with training-subset modes.")
#also will apply one hot encoding for categorical featuers to avoid any nonsense relationship
print("Categorical features: apply one-hot encoding.")
print("Unseen categories: encode as all zeros for that feature.")
#since we cna only train the training set, so to avoid data leackage, I will not fit any validation data
print("Validation data will not be used to fit preprocessing rules.")
# Combine preprocessing and logistic regression into one pipeline.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            #this time I will use logistic, but for R I will use another model
            LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)
print("\nStep 5: Train logistic regression")
model.fit(X_train, y_train)
print("Preprocessing and model fitted using the training subset only.")
# Show the actual values learned for missing-value replacement.
fitted_preprocessor = model.named_steps["preprocessor"]
numeric_fill_values = (
    fitted_preprocessor.named_transformers_["numeric"]
    .named_steps["imputer"]
    .statistics_
)
categorical_fill_values = (
    fitted_preprocessor.named_transformers_["categorical"]
    .named_steps["imputer"]
    .statistics_
)
print("\nLearned numerical fill values:")
for feature, value in zip(numeric_features, numeric_fill_values):
    print(f"{feature}: {value}")
print("\nLearned categorical fill values:")
for feature, value in zip(categorical_features, categorical_fill_values):
    print(f"{feature}: {value}")
# in order to confirm that preprocessing removes missing values.
X_train_processed = fitted_preprocessor.transform(X_train)
X_valid_processed = fitted_preprocessor.transform(X_valid)
print(f"\nProcessed training shape: {X_train_processed.shape}")
print(f"Processed validation shape: {X_valid_processed.shape}")
print(
    "Missing values after preprocessing (training): "
    f"{pd.isna(X_train_processed).sum()}"
)
print(
    "Missing values after preprocessing (validation): "
    f"{pd.isna(X_valid_processed).sum()}"
)
# Evaluate on training and held-out validation data.
print("\nStep 6: Evaluate the model")

train_predictions = model.predict(X_train)
valid_predictions = model.predict(X_valid)
train_accuracy = accuracy_score(y_train, train_predictions)
valid_accuracy = accuracy_score(y_valid, valid_predictions)

print(f"Training-subset accuracy: {train_accuracy:.4f}")
print(f"Validation accuracy: {valid_accuracy:.4f}")
print("This prediction rule is that choose the class with the highest probability.")
print("Validation accuracy measures performance on held-out labeled data.")
print("It is not the accuracy on Kaggle test.csv.")


# Fit a fresh pipeline on all labeled training data.
print("\nStep 7: Refit on all training data")
final_model = clone(model)
final_model.fit(X, y)
print(f"Final model fitted on all {len(X)} training passengers.")
print("Preprocessing rules are now learned from the full training set.")
full_train_predictions = final_model.predict(X)
full_train_accuracy = accuracy_score(y, full_train_predictions)
print(f"Full-training-set accuracy: {full_train_accuracy:.4f}")
print("This is actually a in-sample score, not the independent evaluation.")
# Apply the fitted preprocessing rules and model to test.csv.
print("\nStep 8: Predict test passengers")
test_processed = final_model.named_steps["preprocessor"].transform(X_test)
print(f"Processed test shape: {test_processed.shape}")
print(
    "Missing values after preprocessing (test): "
    f"{pd.isna(test_processed).sum()}"
)
test_predictions = final_model.predict(X_test)
predictions = pd.DataFrame(
    {
        "PassengerId": test["PassengerId"].to_numpy(),
        "Survived": test_predictions.astype(int),
    }
)
if len(predictions) != len(test):
    raise ValueError("Prediction count does not match test passenger count.")
if predictions["Survived"].isna().any():
    raise ValueError("Missing survival predictions were generated.")
print(f"Generated predictions for {len(predictions)} test passengers.")
# I will print out first 10 predictions for preview
print("\nFirst 10 predictions:")
print(predictions.head(10).to_string(index=False))
print("\nPredicted survival counts:")
print(predictions["Survived"].value_counts().sort_index().to_string())

# Save predictions separately from the original input files.
output_dir = Path("/app/output")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "python_predictions.csv"
predictions.to_csv(output_path, index=False)
print(f"\nPredictions saved to: {output_path}")

if "Survived" not in test.columns:
    print(
        "Kaggle test accuracy: unavailable because test.csv has no true Survived labels."
    )
else:
    test_accuracy = accuracy_score(test["Survived"], test_predictions)
    print(f"Test accuracy: {test_accuracy:.4f}")

print("\nPython version completed successfully.")