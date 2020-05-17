# intro
- Types of Analytics: Our Focus
    - Predictive Analytics
        - Unsupervised Learning
        - Supervised Learning
            - Machine predicts "target" feature from any available features
                - vs. classical hypothesis-driven research
            - Regression (quantitative target)
            - Classification (categorical target)

# Preprocessing (wrangling, munging, cleaning)
- Fix errors
- First-step feature engineering:
    - Binary indicator of “high” or “low” income
    - Aggregate deposits in the last month
    - Ratio of credit used to credit available
- Row exclusions
- Anti-aliasing, disambiguation
- Generate 1, row-keyed dataframe

# Encoding Data
- Encoding Text: tf-idf
- PCA: Dimensionality Reduction

# Random Number Generators
- Random numbers should be ~ Unif(0, 1)
- Prior RNG results should not predict subsequent RNG results
- Random numbers should be replicable
    - any number is replicable knowing the seed and the number’s position in the sequence
- Benford's law
    - is an observation about the frequency distribution of leading digits in many real-life sets of numerical data.
    - the leading significant digit is likely to be small
- True RNGs
    - Often based on physical phenomena
- Pseudo RNGs
    - Mathematical/logical operations generate pseudo-random numbers
    - Advantages:
        - Knowledgeable programmer can usually debug even “random” problems
        - Can behave as we want if carefully designed
    - Disadvantages:
        - Efficacy depends on generator and user’s knowledge of it
        - More mathematically complex PRNGs take longer to run

# Unsupervised Learning
- Clustering
    - K-means
    - How to interpret? Feed results into decision tree, and review segments

# Regression
- RSS: Residual sum of squares
- MSE: Mean standard error
- R-square: 1-RSS/TSS
    - TSS = total variability in the responses
- Beyond polynomials, could add: Splines, Cutpoints, Interactions
- Over-fittingthe data
    - Identifying/Avoiding Overfit
        - Adjusted R-square
        - Shrinkage(regularization)methods: shrink the coefficients toward 0, 
            - compared to LS -> reduce variance/increase smoothness
            - Ridge regression (see data science course)
            - Lasso regression

# Out-of-Sample Testing
- Train/Test Split
- Cross-Validation

# Model Assessment
- Specify a profit (loss) associated with your prediction, action, and “truth”

# Classification
- Predicting Labeled Targets
- Binary classification
- Multi-class classification 
- Multi-label classification
- Information gain (IG)
    - based on a purity measure called entropy
- Tree-Structured Models
- How to Avoid Overfitin Tree Induction?
    - Pre-pruning: Stops growing a branch when information becomes unreliable
    - Post-pruning: takes a fully-grown decision tree and discards unreliable parts
- 西瓜书p178(Bagging and Random Forests)
- Bagging:
    - For n trees:
        - Randomly select a subset of observations
        - Build tree
    - Average Predictions
- Random Forests
    - For n trees:
        - Randomly select a subset of observations (bagging)
        - At each split:
            - Select random subset of features
        - Build Tree
    - Average predictions

# Linear Discriminants
- Log Loss
    - logit regression
- Hinge Loss
    - Central to support vector machines
    - Maximize margin
    - Minimize hinge loss
- Regularization
    - $min_W[Loss+\lambda*penalty(W)]$
- ![](note1.png)

# Nearest Neighbors
- KNN
- MegaTelCo: Predicting Customer Churn

# Boosting
- Boosting for regression trees
    - Build a simple regression tree (F1)
    - Compute the residuals
    - Build a tree predicting the residuals (F2)
    - Predict the original value as F1 + F2
    - compute the residuals
    - Build a tree predicting the residuals (F3)
- Boosting for Classification
    - As regression, but instead weight misclassified observations more highly in F2, F3

# Confusion Matrix, P-R curve, ROC, AUC
||predict positive|predict negative|
|---|---|---|
|actual positive|TP |FN|
|actual negative|FP|TN|
||positive|negative|
---
- Accuracy: (TN+TP) / All
- F1 Score: 2\*PPV\*TPR / (PPV+TPR)
    - PPV(precision): TP / (TP+FP)
        - (true positive in we predict positive) (查准率)
    - TPR(sensitivity): TP/ (TP+FN)
        - (true positive in actual positive) (查全率)
- ROC
    - y-axis: TPR
        - TP / (TP+FN)
            - (correct percent in actual positive)
    - x-axis: FPR
        - FP / (TN+FP)
            - (incorrect percent in actual negative) (1-TNR)
- AUC
    - area under ROC curve
    - greater, better
    - Beyond AUC, the ROC curve can also help debug a model. By looking at the shape of the ROC curve, we can evaluate what the model is misclassifying. For example, if the bottom left corner of the curve is closer to the random line, it implies that the model is misclassifying at X=0. Whereas, if it is random on the top right, it implies the errors are occurring at X=1. Also, if there are spikes on the curve (as opposed to being smooth), it implies the model is not stable.

# Unbalanced classes
- ALWAYS CHECK THE CONFUSION MATRIX
- Several high-level solutions
    - resample
        - Downsample the majority class
        - Upsample the minority class
        - evaluate on actual class balance, not resampled one
    - Synthetic upsampling(e.g., adasyn, SMOTE)
    - Class weights

# multiclass classification
- Multiple, mutually exclusive classes
- advantage of binary
    - Deployment advantages
        - We often only care about (or can only act on) one class
        - Easier to visualize
    - Modeling advantages
        - More processing and models accommodate binary classification
        - Binary routines typically perform better
        - Binary routines run faster: 1 model vs. M or ~M^2/2
- 西瓜书p63
- One vs. All (OVA)
    - Construct M  models
    - For every model, set one class = 1, all others = 0; train model
    - Let each model "vote" 
        - choose the result with greatest confidence level
- One vs. One (OVO), aka All vs. All (AVA)
    - Construct M*(M-1)/2 models
    - Every model trained on only two classes
    - Let every model "vote"
- Many vs. Many
    - Error Correcting Output Codes

# Automatied ML
- python libraries
    - TPOT
    - auto-sklearn
        - only for linux
    - autokeras
        - Automates deep learning
- Genetic Algorithm
- questions:
    - only increasing the number of tpot generations likely won’t do much. Why do I make this claim?
    - why did the results get worse with upsampling the minority class?

# DR Classification
- how to use

# Valuing classification models
- Specify an action that you imagine taking based on a prediction
- Represent current baseline practice as a **confusion matrix**
- For any model, compute **cell probabilities**
- Specify a **profit or loss matrix** corresponding to each event in the confusion matrix
- For each model, find the **Expected value** of 1 decision
- Estimate your model’s value relative to the baseline
- How many times will you make this prediction?
- Can the model add value with different prediction thresholds?

# modeling and dataflow with value
- train/validate/test (the expression is vary for theorems)
    - Isolate the Holdout (test set)
    - train/validate get model* and action* (cross-validation and others)
    - then get holdhout score
- Deploy based on Holdout
- Have to use actual class distribution (maybe unbalanced classes)

# HW0
- ref-code\imdb clustering Brandon Rose.py
- (Clustering) can produce clusters based on any combination of features
- Clusters can only be good for a subjective purpose

# HW1
- machine learning VS OLS
- using DataRobot
- learn the stpes to use DataRobot

# HW2
- help prepare you for the exam
- feature Engineering (most important for performance)
- valuing the model! what can we get from using the models

# Internet
- data leakage
    - https://www.kaggle.com/alexisbcook/data-leakage
    - Data leakage (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction.
    - Target leakage occurs when your predictors include data that will not be available at the time you make predictions.
    - Train-Test Contamination, when not careful to distinguish training data from validation data.
    - Careful separation of training and validation data can prevent train-test contamination, and pipelines can help implement this separation.
- Model Validation
    - https://www.kaggle.com/dansbecker/model-validation
    - But if this pattern doesn't hold when the model sees new data, the model would be very inaccurate when used in practice.
    - Since models' practical value come from making predictions on new data, we measure performance on data that wasn't used to build the model. The most straightforward way to do this is to exclude some data from the model-building process, and then use those to test the model's accuracy on data it hasn't seen before. This data is called validation data.
- cross-validation
    - https://www.kaggle.com/alexisbcook/cross-validation
    - what predictive variables to use, what types of models to use, what arguments to supply to those models, etc.
    - drawbacks
    - In general, the larger the validation set, the less randomness (aka "noise") there is in our measure of model quality, and the more reliable it will be. 
- underfitting and overfitting
    - https://www.kaggle.com/dansbecker/underfitting-and-overfitting
    - overfitting, where a model matches the training data almost perfectly, but does poorly in validation and other new data.
    - When a model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data, that is called underfitting.
    - Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions.
    - Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
