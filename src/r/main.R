# Load the original Titanic datasets and print basic checks.
# Data files are supplied through a read-only Docker mount.
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
    "\nKaggle test accuracy cannot be calculated directly:",
    "test.csv has no true Survived labels.\n"
  )
}
cat("\nData loading and basic checks completed.\n")