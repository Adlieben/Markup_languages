## SETUP ----
# Packages and library
Sys.setenv(LANG = "en")
library(MASS)
library(tidyverse)
library(ISLR)
library(rpart)
library(rpart.plot)
library(e1071) # for SVM
library(randomForest)
library(nnet)
library(ipred)
library(vcd) # Cramér's V
library(forcats) # For missing data
library(knitr)

# Functions
source('Testing codes functions.R')

## DATA TRANSFORMATION ----
# Load data
test <- read.csv('Data/test.csv')
train <- read.csv('Data/train.csv') # As can be expected, we have all variable + the outcome variable

# Convert characters to factors
train <- train %>%
  mutate(across(where(is.character), as.factor))

test <- test %>%
  mutate(across(where(is.character), as.factor))

# Check correlations to avoid multicollinearity
# Define threshold for high associations
threshold <- 0.7

# Separate numeric and categorical data
numeric_data <- train %>% select(where(is.numeric))
categorical_data <- train %>% select(where(is.factor))

# Initialize an empty list to store high associations
high_association <- list()

# 1. Process Numeric Variables (Correlation)
if (ncol(numeric_data) > 1) {
  # Loop through each unique pair of numeric variables for pairwise correlation
  for (i in 1:(ncol(numeric_data) - 1)) {
    for (j in (i + 1):ncol(numeric_data)) {
      # Extract the two variables for the pair and remove rows with missing values
      pair_data <- na.omit(numeric_data[, c(i, j)])
      if (nrow(pair_data) > 1) {  # Ensure there are at least 2 observations after omitting NAs
        # Calculate the correlation for the pair
        cor_value <- cor(pair_data[, 1], pair_data[, 2])
        
        # If correlation is above threshold, save the result
        if (abs(cor_value) > threshold) {
          high_association[[length(high_association) + 1]] <- list(
            Var1 = colnames(numeric_data)[i],
            Var2 = colnames(numeric_data)[j],
            Position1 = i,
            Position2 = j,
            Measure = "Correlation",
            Value = cor_value
          )
        }
      }
    }
  }
}

# 2. Process Categorical Variables (Cramér's V)
if (ncol(categorical_data) > 1) {
  # Loop through each unique pair of categorical variables for pairwise association
  for (i in 1:(ncol(categorical_data) - 1)) {
    for (j in (i + 1):ncol(categorical_data)) {
      # Extract the two variables for the pair and remove rows with missing values
      pair_data <- na.omit(categorical_data[, c(i, j)])
      
      # Ensure that both variables in the pair have at least two levels after dropping missing values
      if (nrow(pair_data) > 1 && length(unique(pair_data[[1]])) > 1 && length(unique(pair_data[[2]])) > 1) {
        # Calculate Cramér's V for the pair
        cramers_v <- assocstats(table(pair_data[[1]], pair_data[[2]]))$cramer
        
        # Check if Cramér's V is not NA and above the threshold
        if (is.na(cramers_v) | cramers_v > threshold) {
          high_association[[length(high_association) + 1]] <- list(
            Var1 = names(categorical_data)[i],
            Var2 = names(categorical_data)[j],
            Position1 = i,
            Position2 = j,
            Measure = "Cramér's V",
            Value = cramers_v
          )
        }
      }
    }
  }
}

# Convert high associations list to a data frame
high_association_df <- do.call(rbind, lapply(high_association, as.data.frame))

# Display the results
if (nrow(high_association_df) > 0) {
  print(high_association_df)
} else {
  print("No associations above the threshold found.")
}
# None of the correlations are particularly worrying

# Sort factor columns by number of levels (ascending order)
factor_levels <- sapply(train %>% select(where(is.factor)), nlevels)
sorted_factors <- sort(factor_levels)
print("Factor variables sorted by number of levels (lowest to highest):")
print(sorted_factors)

# Missing data per variable
missing_data <- sapply(train, function(x) sum(is.na(x)))
print("Missing data per variable:")
print(missing_data) # There are only a few variables with missing data, but some are almost entirely missing data
# For example PoolQC has 7 observations.
# This will be problematic as folds will only have 1 or 2 observations

# Something I noticed during analysis is that Utilities only has 1 different answer than the rest, so we should
# Better remove it when possible because it will not be very applicable, especially in cross-validation.
# Worse though, is the TotalBsmtSF which is actually simply the sum of trainset$BsmtFinSF1 + trainset$BsmtFinSF2 + trainset$BsmtUnfSF so we must remove it
# This is also the case for GrLivArea, which is also the sum of several square feet, that we thus want to remove.
# I noticed these last two by running lm then noticing it doesn't work because of collinearity. alias(model) was legendary for this.
# Alternatively, I could also have removed the partitioning variables instead of the total but that would remove granularity.

# Remove id column for later + check with no missing data
train_clean <- train %>% select(-c(Id,
                                   BsmtUnfSF, GrLivArea)) # due to collinearity
train_nomissing <- train %>% select(-c(Id, LotFrontage, Alley, MasVnrType, MasVnrArea, BsmtQual, BsmtCond, BsmtExposure,
                                       BsmtFinType1, BsmtFinType2, Electrical, FireplaceQu, GarageYrBlt, GarageFinish,
                                       GarageQual, GarageCond, Fence, PoolQC, MiscFeature, GarageType,
                                       BsmtUnfSF, GrLivArea)) # due to collinearity

## MISSING DATA ----
# LotFrontage NA appears to indicate not relevant (so 0). The same is obvious of the masonry codes (MasVnrType:none and MasVnrArea: 0).
# Alley NA is clearly 'none', and can be coded as such, given the coding scheme.
# Basement elements can also be assumed to apply in situations where there is no basement (BsmtQual(0), BsmtCond (none), BsmtExposure(none),
# BsmtFinType1(none), BsmtFinType2(none)). There is also an overlap in the missings I could show which makes sense. To note, unfinished
# Basements have 0 squared footage so this is consistent (loook at other variables).
# In fact, this is also the case for other variables related to the garage, the pool, the fence and the miscellaneous features (MiscFeature) 
# so imputation isn't necessary and they can all be replaced by 0 or none.
# Create train_zero and test_zero by copying train and test
train_zero <- train
test_zero <- test

# Replace missing values in numeric columns with 0 and in factor columns with 'none' for train_zero
train_zero <- train_zero %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), 0, .))) %>%
  mutate(across(where(is.factor), ~ fct_na_value_to_level(., "none")))

# Replace missing values in numeric columns with 0 and in factor columns with 'none' for test_zero
test_zero <- test_zero %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), 0, .))) %>%
  mutate(across(where(is.factor), ~ fct_na_value_to_level(., "none")))

# Check for missing data to confirm replacement
missing_data_train <- sapply(train_zero, function(x) sum(is.na(x)))
missing_data_test <- sapply(test_zero, function(x) sum(is.na(x)))

train_zero <- train_zero %>% select(-Id)
test_zero <- test_zero %>% select(-Id)

## ANALYSIS ----
# CROSS-VALIDATION will ensure that the model is reproducible on the test data
# Once we have a model, we can fit it to the whole data
# Set seed for reproducibility
set.seed(100)

# Run cross-validation
cv_results <- cross_validate_rmse(data = train_nomissing, target = "SalePrice",
                                  methods = c('linear', 'tree', 'rf', 'svm', 'nn'))

# Print the results
print(cv_results)

# These results could perhaps be better interpreted if we augment the amount of folds: 
# Define the number of folds to test (from 1 to 10)
folds_range <- 2:10
set.seed(200)

# List of methods to evaluate
methods <- c('linear', 'quadratic', 'cubic', 'tree', 'rf', 'svm', 'nn')

# Initialize an empty data frame to store the results
results_all <- data.frame(Folds = integer(), Method = character(), Mean_RMSE = numeric(), stringsAsFactors = FALSE)

# Loop through each fold count and each method
for (folds in folds_range) {
  print(folds)
  for (method in methods) {
    # Run cross-validation with the specified number of folds and method
    cv_result <- cross_validate_rmse(data = train_nomissing, target = "SalePrice", folds = folds, methods = c(method))
    
    # Add the number of folds to the result and store in results_all
    cv_result$Folds <- folds
    results_all <- rbind(results_all, cv_result)
  }
}

# Plot RMSE against the number of folds for each model
ggplot(results_all, aes(x = Folds, y = Mean_RMSE, color = Method)) +
  geom_line() +
  geom_point() +
  labs(title = "Effect of Number of Folds on RMSE for Each Model",
       x = "Number of Folds",
       y = "Mean RMSE") +
  theme_minimal() 

# It's clear from the plot that the support vector machine and random forest are vastly outperforming
# The other methods. The addition of the quadratic and cubic casts a bit of doubt as they perform equally quite well.

## ON THE MISSING DATASET ----
set.seed(200)

# List of methods to evaluate
methods <- c('linear', 'quadratic', 'rf', 'svm')  # Only specified methods

# Initialize an empty data frame to store the results
results_all <- data.frame(Folds = integer(), Method = character(), Mean_RMSE = numeric(), stringsAsFactors = FALSE)

# Loop through each fold count and each method
for (folds in folds_range) {
  print(folds)
  for (method in methods) {
    # Run cross-validation with the specified number of folds and method
    cv_result <- cross_validate_rmse(data = train_zero, target = "SalePrice", folds = folds, methods = c(method))
    
    # Add the number of folds to the result and store in results_all
    cv_result$Folds <- folds
    results_all <- rbind(results_all, cv_result)
  }
}

# Plot RMSE against the number of folds for each model
ggplot(results_all, aes(x = Folds, y = Mean_RMSE, color = Method)) +
  geom_line() +
  geom_point() +
  labs(title = "Effect of Number of Folds on RMSE for Each Model",
       x = "Number of Folds",
       y = "Mean RMSE") +
  theme_minimal()

## COMPARING PARAMETERS FOR THE SVM ----
set.seed(200)

# SVM parameter options
svm_kernels <- c("linear", "polynomial", "radial", "sigmoid")
svm_types <- c("eps-regression", "nu-regression")

# Initialize an empty data frame to store the results
results_svm <- data.frame(Folds = integer(), Kernel = character(), SVM_Type = character(), Mean_RMSE = numeric(), stringsAsFactors = FALSE)

# Loop through each fold count, kernel type, and SVM regression type
for (folds in folds_range) {
  print(folds)
  for (kernel in svm_kernels) {
    for (svm_type in svm_types) {
      # Run cross-validation for each SVM configuration
      cv_result <- cross_validate_rmse(data = train_zero, target = "SalePrice", folds = folds, 
                                       methods = c("svm"), svm_kernel = kernel, svm_type = svm_type)
      
      # Add configuration details to results
      cv_result$Folds <- folds
      cv_result$Kernel <- kernel
      cv_result$SVM_Type <- svm_type
      results_svm <- rbind(results_svm, cv_result)
    }
  }
}

# Plot RMSE against the number of folds for each SVM configuration
ggplot(results_svm, aes(x = Folds, y = Mean_RMSE, color = Kernel, linetype = SVM_Type)) +
  geom_line() +
  geom_point() +
  labs(title = "Effect of Number of Folds on RMSE for SVM Configurations",
       x = "Number of Folds",
       y = "Mean RMSE") +
  theme_minimal()

# Display the results as a table
kable(results_svm, caption = "RMSE Results for Each SVM Configuration across Fold Counts")

## PERFORMANCE ON THE TEST SET
# Define the formula for the model based on the target variable and predictors
target <- "SalePrice"
predictors <- as.formula(paste(target, "~ ."))

# Train the SVM model on the entire train_zero dataset using radial kernel and nu-regression
svm_model <- svm(predictors, data = train_zero, kernel = "radial", type = "nu-regression")

# Generate predictions on test_zero
test_zero$Predicted_SalePrice <- predict(svm_model, newdata = test_zero)

# Display the predictions
head(test_zero$Predicted_SalePrice)

# Combine the Id from test with the predicted SalePrice in test_zero
submission <- data.frame(Id = test$Id, SalePrice = test_zero$Predicted_SalePrice)

# Write the submission to a CSV file with the specified header
write.csv(submission, file = "submission_Adam_Maghout.csv", row.names = FALSE)
