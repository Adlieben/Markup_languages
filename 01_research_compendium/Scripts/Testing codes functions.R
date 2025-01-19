# Define RMSE function for continuous variables
rmse <- function(observed, predicted) {
  sqrt(mean((log(observed) - log(predicted))^2))
}

# Create function for cross-validation
cross_validate_rmse <- function(data, target, folds = 2, methods = c('linear', 'quadratic', 'rf', 'svm'), 
                                svm_kernel = "radial", svm_type = "eps") {
  
  # Validate the method argument
  allowed_methods <- c('linear', 'quadratic', 'rf', 'svm')
  if (!all(methods %in% allowed_methods)) {
    stop("Error: methods must be one of 'linear', 'quadratic', 'rf', or 'svm'")
  }
  
  # Validate the SVM kernel argument
  allowed_kernels <- c("linear", "polynomial", "radial", "sigmoid")
  if (!(svm_kernel %in% allowed_kernels)) {
    stop("Error: svm_kernel must be one of 'linear', 'polynomial', 'radial', or 'sigmoid'")
  }
  
  # Validate the SVM type argument
  allowed_types <- c("eps", "nu")
  if (!(svm_type %in% allowed_types)) {
    stop("Error: svm_type must be either 'eps' or 'nu'")
  }
  
  # Assign each data point to one of the folds
  n_fold <- ceiling(nrow(data) / folds)
  data$fold <- sample(rep(1:folds, length.out = nrow(data)))
  
  # Prepare storage for mean RMSEs
  results <- data.frame(Method = character(), Mean_RMSE = numeric(), stringsAsFactors = FALSE)
  
  # Loop through each method
  for (method in methods) {
    fold_rmse <- numeric(folds)
    
    # Cross-validation loop
    for (i in 1:folds) {
      # Split into training and testing sets
      trainset <- data %>% filter(fold != i) %>% select(-fold)
      testset <- data %>% filter(fold == i) %>% select(-fold)
      
      # Adjust factor levels in the training set
      factor_vars <- trainset %>% select(where(is.factor))
      for (var in names(factor_vars)) {
        trainset[[var]] <- droplevels(trainset[[var]])
        level_counts <- table(trainset[[var]])
        
        low_freq_levels <- names(level_counts[level_counts < 2])
        
        if (length(low_freq_levels) > 0) {
          combined_name <- paste(low_freq_levels, collapse = "_")
          trainset[[var]] <- factor(ifelse(trainset[[var]] %in% low_freq_levels, combined_name, as.character(trainset[[var]])))
          
          level_counts <- table(trainset[[var]])
          
          if (level_counts[combined_name] < 2) {
            sorted_levels <- names(sort(level_counts))
            next_level <- sorted_levels[2]
            new_combined_name <- paste(combined_name, next_level, sep = "_")
            trainset[[var]] <- factor(ifelse(trainset[[var]] == combined_name | trainset[[var]] == next_level, 
                                             new_combined_name, 
                                             as.character(trainset[[var]])))
          
          
          testset[[var]] <- factor(ifelse(testset[[var]] %in% low_freq_levels | testset[[var]] == next_level, 
                                          new_combined_name, 
                                          as.character(testset[[var]])), 
                                   levels = c(levels(testset[[var]]), new_combined_name))
          testset[[var]] <- droplevels(testset[[var]])
          }
        }
        
        if (length(unique(trainset[[var]])) < 2 | !all(levels(testset[[var]]) %in% levels(trainset[[var]]))) {
          trainset <- trainset %>% select(-all_of(var))
          testset <- testset %>% select(-all_of(var))
        }
      }
      
      # Dynamically create formula based on target and remaining predictors
      numeric_vars <- setdiff(names(trainset)[sapply(trainset, is.numeric)], target)
      factor_vars <- names(trainset)[sapply(trainset, is.factor)]
      
      if (method == 'linear') {
        predictors <- as.formula(paste(target, "~", paste(c(numeric_vars, factor_vars), collapse = " + ")))
        
      } else if (method == 'quadratic') {
        predictors <- as.formula(paste(target, "~", 
                                       paste(c(numeric_vars, factor_vars), collapse = " + "),
                                       "+", paste(paste0("I(", numeric_vars, "^2)"), collapse = " + ")))
        
      } else {
        predictors <- as.formula(paste(target, "~ ."))
      }
      
      # Fit the model and make predictions, replacing negative values with 10
      if (method == 'linear') {
        model <- lm(predictors, data = trainset)
        testset$Predicted <- pmax(predict(model, newdata = testset), 10)
        
      } else if (method == 'quadratic') {
        model <- lm(predictors, data = trainset)
        testset$Predicted <- pmax(predict(model, newdata = testset), 10)
        
      } else if (method == 'rf') {
        factor_vars_rf <- names(trainset)[sapply(trainset, is.factor)]
        trainset_rf <- trainset
        testset_rf <- testset
        
        vars_to_drop <- c()
        
        for (var in factor_vars_rf) {
          if (!identical(levels(trainset[[var]]), levels(testset[[var]]))) {
            vars_to_drop <- c(vars_to_drop, var)
          }
        }
        
        trainset_rf <- trainset_rf %>% select(-all_of(vars_to_drop))
        testset_rf <- testset_rf %>% select(-all_of(vars_to_drop))
        
        for (var in names(trainset_rf)[sapply(trainset_rf, is.factor)]) {
          testset_rf[[var]] <- factor(testset_rf[[var]], levels = levels(trainset_rf[[var]]))
        }
        
        model <- randomForest(predictors, data = trainset_rf, ntree = 100)
        testset$Predicted <- pmax(predict(model, newdata = testset_rf), 10)
        
      } else if (method == 'svm') {
        svm_args <- list(formula = predictors, data = trainset, kernel = svm_kernel)
        
        # Set SVM regression type
        if (svm_type == "eps") {
          model <- do.call(svm, c(svm_args, list(type = "eps-regression")))
        } else if (svm_type == "nu") {
          model <- do.call(svm, c(svm_args, list(type = "nu-regression")))
        }
        
        testset$Predicted <- pmax(predict(model, newdata = testset), 10)
      }
      
      # Calculate RMSE for the current fold
      fold_rmse[i] <- rmse(testset[[target]], testset$Predicted)
    }
    
    results <- rbind(results, data.frame(Method = method, Mean_RMSE = mean(fold_rmse), stringsAsFactors = FALSE))
  }
  
  return(results)
}