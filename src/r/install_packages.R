# Install a fixed package version for reproducibility.
package_url <- paste0(
  "https://cran.r-project.org/src/contrib/Archive/rpart/",
  "rpart_4.1.24.tar.gz"
)
install.packages(
  package_url,
  repos = NULL,
  type = "source"
)
# Fail the Docker build if installation did not succeed.
stopifnot(requireNamespace("rpart", quietly = TRUE))
stopifnot(as.character(packageVersion("rpart")) == "4.1.24")
cat("Installed rpart version:", as.character(packageVersion("rpart")), "\n")