# titanic-survival-project
Titanic survival prediction using Python and R with Docker

This project predicts passenger survival using the Kaggle Titanic dataset. The Python workflow uses logistic regression, and the R workflow uses a decision tree. Each workflow runs in its own Docker container.

## 1. Set up the project

Install and start Docker Desktop. Clone this repository, open a terminal in the project root (the folder containing this README), and create the input and output directories:

```bash
mkdir -p src/data output
```

On Kaggle, open the **Titanic - Machine Learning from Disaster** competition and download `train.csv` and `test.csv` from the **Data** tab. If you download a ZIP archive, extract it first. Place the two files here:

- `src/data/train.csv` — 891 passengers with known survival outcomes
- `src/data/test.csv` — 418 passengers whose survival outcomes we predict

The input CSVs and generated predictions are excluded from Git. Docker installs the dependencies for each workflow, so you do not need to install Python or R packages on your computer.

## 2. Run the Python workflow

From the project root, run:

```bash
docker build -t titanic-python ./src/python &&
docker run --rm \
  -v "$(pwd)/src/data:/data:ro" \
  -v "$(pwd)/output:/app/output" \
  titanic-python python main.py --data-dir /data
```

This writes `output/python_predictions.csv`.

## 3. Run the R workflow

From the same directory, run:

```bash
docker build -t titanic-r ./src/r &&
docker run --rm \
  -v "$(pwd)/src/data:/data:ro" \
  -v "$(pwd)/output:/app/output" \
  titanic-r
```

This writes `output/r_predictions.csv`. Both commands mount the input folder as read-only and save predictions to the local `output/` folder.

## 4. Read the results

Each script reports training and validation accuracy in the terminal. Both workflows set aside approximately 20% of the labeled data for validation, then refit the model on all 891 labeled passengers before predicting survival for the 418 test passengers.

| Workflow | Training-subset accuracy | Validation accuracy |
| Python logistic regression | 80.76% | 80.45% |
| R decision tree | 83.15% | 79.89% |

The two workflows use their own validation splits, so the scores describe each run rather than a comparison on identical passengers. Kaggle's
`test.csv` has no `Survived` labels, so test accuracy cannot be calculated locally.

Each prediction file has 418 data rows and two columns: `PassengerId` and `Survived`. A prediction of `0` means the passenger did not survive, and `1` means the passenger survived.