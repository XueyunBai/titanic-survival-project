# Load the original Titanic datasets and print basic checks.
# This time I will use R to redo all previous steps that did in python
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else "/data"
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")
# Stop with a clear message if either input file is missing.
for (path in c(train_path, test_path)) {
  if (!file.exists(path)) {
    stop(
      "Missing input file: ", path,
      ". Place train.csv and test.csv in src/data."
    )
  }
}
train <- read.csv(
  train_path,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA")
)
test <- read.csv(
  test_path,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA")
)

cat("Step 1: Load Titanic data\n")
cat("Data directory:", data_dir, "\n")
cat("Training data:", nrow(train), "rows,", ncol(train), "columns\n")
cat("Test data:", nrow(test), "rows,", ncol(test), "columns\n")
cat("\nTraining columns:\n")
print(names(train))
cat("\nTest columns:\n")
print(names(test))
cat("\nMissing values in training data:\n")
print(colSums(is.na(train)))
cat("\nMissing values in test data:\n")
print(colSums(is.na(test)))

if (!"Survived" %in% names(train)) {
  stop("train.csv must contain the Survived column.")
}

cat("\nTraining survival counts:\n")
print(table(train$Survived, useNA = "ifany"))

if (!"Survived" %in% names(test)) {
  cat(
    "\nKaggle test accuracy cannot be calculated directly: test.csv has no true Survived labels.\n"
  )
}
cat("\nData loading and basic checks completed.\n")

# Then select model features.
numeric_features <- c("Pclass", "Age", "SibSp", "Parch", "Fare")
categorical_features <- c("Sex", "Embarked")
features <- c(numeric_features, categorical_features)

for (dataset_name in c("train", "test")) {
  dataset <- get(dataset_name)
  missing_columns <- setdiff(features, names(dataset))

  if (length(missing_columns) > 0) {
    stop(
      dataset_name, " is missing required columns: ",
      paste(missing_columns, collapse = ", ")
    )
  }
}

if (anyNA(train$Survived) ||
    !all(train$Survived %in% c(0, 1))) {
  stop("Training survival labels must be 0 or 1 with no missing values.")
}

cat("\nStep 2: Select features\n")
cat("Selected features:", paste(features, collapse = ", "), "\n")
cat("Target: Survived (0 means did not survive, 1 means survived)\n")
cat("Excluded from model: PassengerId, Name, Ticket, Cabin\n")
cat("No passenger rows have been removed.\n")

# after that: Split within each survival class to preserve class proportions.
set.seed(42)
train_indices <- unlist(
  lapply(c(0, 1), function(label) {
    indices <- which(train$Survived == label)
    sample(indices, size = floor(0.8 * length(indices)))
  }),
  use.names = FALSE
)

training <- train[train_indices, c(features, "Survived")]
validation <- train[-train_indices, c(features, "Survived")]

cat("\nStep 3: Create training and validation subsets\n")
cat("Training subset:", nrow(training), "passengers\n")
cat("Validation subset:", nrow(validation), "passengers\n")
cat("Split: approximately 80% training and 20% validation.\n")
cat("Random seed: 42; sampling is stratified by survival class.\n")
cat("\nTraining-subset survival proportions:\n")
print(prop.table(table(training$Survived)))
cat("\nValidation-subset survival proportions:\n")
print(prop.table(table(validation$Survived)))
cat("\nMissing values will be handled using training-subset statistics only.\n")


cat("\nStep 4: Fill missing feature values\n")

fill_values <- list()

#for numeric features, I will fill na with median number of trainning dataset
for (column in numeric_features) {
  fill_values[[column]] <- median(training[[column]], na.rm = TRUE)
}
#but for categorical feature, I will fill na with mode of trainning dataset
for (column in categorical_features) {
  counts <- table(training[[column]])
  fill_values[[column]] <- names(counts)[which.max(counts)]
}

apply_fill_values <- function(data, values) {
  for (column in names(values)) {
    missing <- is.na(data[[column]])
    data[[column]][missing] <- values[[column]]
  }
  data
}

training_clean <- apply_fill_values(training, fill_values)
validation_clean <- apply_fill_values(validation, fill_values)

cat("Fill values calculated from the training subset:\n")
print(fill_values)
#then I will test whether there is still na values
cat("Remaining missing feature values in training:",
    sum(is.na(training_clean[features])), "\n")
cat("Remaining missing feature values in validation:",
    sum(is.na(validation_clean[features])), "\n")


cat("\nStep 5: Train a decision tree\n")
# Treat 0 and 1 as two outcome classes.
training_clean$Survived <- factor(training_clean$Survived, levels = c(0, 1))

tree_formula <- reformulate(features, response = "Survived")
tree_model <- rpart::rpart(
  tree_formula,
  data = training_clean,
  method = "class"
)
cat("Decision tree trained on", nrow(training_clean), "passengers.\n")

cat("\nStep 6: Evaluate the decision tree\n")

training_predictions <- predict(tree_model, newdata = training_clean, type = "class")
validation_predictions <- predict(tree_model, newdata = validation_clean, type = "class")

training_accuracy <- mean(training_predictions == training_clean$Survived)
validation_accuracy <- mean(
  as.character(validation_predictions) == as.character(validation_clean$Survived)
)
#then just print out these accuracies
cat(sprintf("Training-subset accuracy: %.4f\n", training_accuracy))
cat(sprintf("Validation accuracy: %.4f\n", validation_accuracy))

#then I will use all labeled dataset to retrain and make the predictions
cat("\nStep 7: Refit on all labeled passengers\n")

full_fill_values <- list()
for (column in numeric_features) {
  full_fill_values[[column]] <- median(train[[column]], na.rm = TRUE)
}
for (column in categorical_features) {
  counts <- table(train[[column]])
  full_fill_values[[column]] <- names(counts)[which.max(counts)]
}

full_training_clean <- apply_fill_values(train, full_fill_values)
full_training_clean$Survived <- factor(full_training_clean$Survived, levels = c(0, 1))

final_tree <- rpart::rpart(
  tree_formula,
  data = full_training_clean,
  method = "class"
)
cat("Final decision tree trained on", nrow(full_training_clean), "passengers.\n")

cat("\nStep 8: Predict survival for test passengers\n")

test_clean <- apply_fill_values(test, full_fill_values)
test_predictions <- predict(final_tree, newdata = test_clean, type = "class")
stopifnot(length(test_predictions) == nrow(test), !anyNA(test_predictions))

predictions <- data.frame(
  PassengerId = test$PassengerId,
  Survived = as.integer(as.character(test_predictions))
)

dir.create("output", showWarnings = FALSE)
output_path <- file.path("output", "r_predictions.csv")
write.csv(predictions, output_path, row.names = FALSE)

cat("Predictions generated:", nrow(predictions), "\n")
print(table(predictions$Survived))
cat("Predictions saved to:", output_path, "\n")
#in the slack professor mentioned that we don't need to worry about testing the model accuracy on test set, I will just print out the predictions
cat("Kaggle test accuracy is unavailable because test.csv has no true survival labels.\n")
