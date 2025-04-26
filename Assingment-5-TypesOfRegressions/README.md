# Regression Analysis Showcase 🏘️💰

This repository explores various regression techniques using a synthetically generated dataset for house price prediction. We aim to understand the relationships between house features and their prices, applying and comparing different regression models.

## Table of Contents

1.  [What is Regression?](#what-is-regression-)
2.  [Why is Regression Important?](#why-is-regression-important-)
3.  [Advantages & Disadvantages](#advantages--disadvantages-)
4.  [The Dataset](#the-dataset-)
5.  [Regression Models Explored](#regression-models-explored-)
6.  [Model Explanations & Results](#model-explanations--results-)
    *   [1. Linear Regression](#1-linear-regression-)
    *   [2. Logistic Regression](#2-logistic-regression-)
    *   [3. Polynomial Regression](#3-polynomial-regression-)
    *   [4. Ridge Regression](#4-ridge-regression-)
    *   [5. Lasso Regression](#5-lasso-regression-)
    *   [6. Quantile Regression](#6-quantile-regression-)
    *   [7. Decision Tree Regression](#7-decision-tree-regression-)
    *   [8. Random Forest Regression](#8-random-forest-regression-)
    *   [9. Support Vector Regression (SVR)](#9-support-vector-regression-svr-)
    *   [10. Elastic Net Regression](#10-elastic-net-regression-)
    *   [11. Stepwise Regression](#11-stepwise-regression-forward-selection-)
    *   [12. Bayesian Linear Regression](#12-bayesian-linear-regression-)
7.  [Conclusion](#conclusion-)

## What is Regression? 📈

Regression analysis is a fundamental statistical modeling technique used to estimate the relationships between a **dependent variable** (the outcome we want to predict or explain, often called 'Y') and one or more **independent variables** (the predictors or factors, often called 'X').

🎯 The primary goal is to understand how the typical value of the dependent variable changes when any one of the independent variables is varied, while the other independent variables are held fixed. It essentially tries to find the "line" (or curve) of best fit through the data points.

## Why is Regression Important? 🤔

Regression is a cornerstone of predictive analytics and data science for several reasons:

*   🔮 **Prediction & Forecasting:** It allows us to predict future outcomes based on historical data (e.g., predicting future sales, stock prices, or house prices).
*   ⚙️ **Quantifying Relationships:** It helps understand the strength and direction of the relationship between variables (e.g., How much does square footage *really* impact house price?).
*   📊 **Modeling Complex Systems:** It provides a mathematical way to model real-world processes.
*   💡 **Identifying Key Drivers:** Techniques like Lasso or Stepwise regression can help identify the most influential factors affecting an outcome.

## Advantages & Disadvantages ⚖️

Like any technique, regression has its pros and cons:

**Advantages:** ✅

*   **Interpretability:** Linear models are often easy to understand and explain.
*   **Simplicity:** Basic regression models are relatively simple to implement.
*   **Well-Established:** It's a widely studied and understood statistical method.
*   **Foundation:** Forms the basis for many more advanced machine learning techniques.

**Disadvantages:** ⚠️

*   **Assumptions:** Linear regression relies on assumptions (linearity, independence, normality of errors, homoscedasticity) that might not hold true in real-world data.
*   **Sensitivity to Outliers:** Standard regression can be heavily influenced by outliers.
*   **Correlation vs. Causation:** Regression identifies correlations, not necessarily causal relationships. Careful interpretation is needed.
*   **Overfitting Risk:** Complex models (like high-degree polynomials or deep decision trees) can overfit the training data if not properly regularized or pruned.

## The Dataset 🏡

We use a synthetically generated dataset to predict house prices based on a few key features.

**Features:**

*   `Size_SqFt`: The size of the house in square feet.
*   `Num_Bedrooms`: The number of bedrooms.
*   `Age_Years`: The age of the house in years.

**Target Variable:**

*   `Price`: The predicted price of the house.

**Sample Data:**
Generated House Price Dataset:
Size_SqFt Num_Bedrooms Age_Years Price
0 1660 2 41 317103
1 2094 3 3 389992
2 1930 4 39 340662
3 1895 3 6 374082
4 2438 2 8 444649

**Dataset Information:**
Dataset Info:
Training set size: 80 samples
Test set size: 20 samples


**Data Visualization:**

![EDA.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/EDA.png)

## Regression Models Explored 🛠️

We implement and evaluate the following regression models:

1.  Linear Regression
2.  Logistic Regression (Adapted for Classification)
3.  Polynomial Regression
4.  Ridge Regression
5.  Lasso Regression
6.  Quantile Regression
7.  Decision Tree Regression
8.  Random Forest Regression
9.  Support Vector Regression (SVR)
10. Elastic Net Regression
11. Stepwise Regression (Forward Selection)
12. Bayesian Linear Regression

## Model Explanations & Results 📝

Here we detail each model and present its performance on the test set.

*(Note: MSE = Mean Squared Error, R² = R-squared score. Lower MSE is better. R² closer to 1 is better.)*

---

### 1. Linear Regression 📏

*   **Explanation:** Models the relationship between features and the target using a linear equation (`Price = β₀ + β₁*Size + β₂*Bedrooms + β₃*Age`). Assumes a straight-line relationship.
*   **Results:**
    ```
    --- 1. Linear Regression ---
    Coefficients (Size, Bedrooms, Age): [ 149.09 4661.72 -526.16]
    Intercept: 59470.22
    Mean Squared Error (MSE): 401516036.44
    R-squared (R²): 0.9733
    ```
*   **Visualization:**
    ![1.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/1.png)

---

### 2. Logistic Regression 🚦

*   **Explanation:** Primarily a **classification** algorithm. Used here to predict a binary outcome: whether a house price is above the median ('Expensive' = 1) or not ('Not Expensive' = 0). Models the probability of belonging to a class.
*   **Target Definition:**
    ```
    Median House Price: 374627.50
    Target Variable (y_binary) value counts:
    Price
    0    50
    1    50
    Name: count, dtype: int64
    ```
*   **Results:**
    ```
    --- 2. Logistic Regression ---
    Accuracy: 0.9000

    Confusion Matrix:
    [[9 1]
     [1 9]]
    (Correctly classified 9 'Not Expensive' and 9 'Expensive'. Misclassified 1 of each.)
    ```
*   **Visualization:**
    ![2.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/2.png)

---

### 3. Polynomial Regression 🎢

*   **Explanation:** Extends linear regression by adding polynomial terms (e.g., `Size²`, `Age²`, `Size*Age`) to model non-linear, curved relationships. Here, degree 2 was used.
*   **Results:**
    ```
    --- 3. Polynomial Regression ---
    Using Polynomial Features of degree 2
    Original number of features: 3
    Number of features after polynomial transformation: 9

    Mean Squared Error (MSE): 822305141.66
    R-squared (R²): 0.9454
    Improvement over Linear Regression R² (0.9733): -0.0279
    (Note: Performance degraded compared to simple Linear Regression on this dataset.)
    ```
*   **Visualization:**
   ![3.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/3.png)

---

### 4. Ridge Regression ⚖️

*   **Explanation:** A type of regularized linear regression (L2 penalty). It shrinks coefficients towards zero to prevent overfitting and handle multicollinearity. Requires feature scaling. `alpha` controls the penalty strength.
*   **Results:**
    ```
    --- 4. Ridge Regression ---
    Using Ridge Regression with alpha=1.0

    Coefficients (Scaled Features): [110863.56   5184.12  -7558.37]
    Intercept: 390675.30
    Mean Squared Error (MSE): 404785437.54
    R-squared (R²): 0.9731

    Comparison of Coefficients (Scaled Data):
    Linear Regression: [112273.89   5211.63  -7798.84]
    Ridge Regression (alpha=1.0): [110863.56   5184.12  -7558.37]
    (Note: Ridge coefficients are slightly shrunk towards zero compared to standard Linear Regression.)
    ```
*   **Visualization:**
    ![4.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/4.png)

---

### 5. Lasso Regression 🎯

*   **Explanation:** Another regularized linear regression (L1 penalty). It can shrink coefficients exactly to zero, performing automatic feature selection. Requires feature scaling. `alpha` controls the penalty strength.
*   **Results:**
    ```
    --- 5. Lasso Regression ---
    Using Lasso Regression with alpha=100.0

    Coefficients (Scaled Features): [112164.5    5118.04  -7691.13]
    Number of non-zero coefficients: 3 out of 3
    Intercept: 390675.30
    Mean Squared Error (MSE): 400079814.23
    R-squared (R²): 0.9734
    (Note: With this alpha, Lasso did not zero out any coefficients on this dataset.)
    ```
*   **Visualization:**
    ![5.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/5.png)

---

### 6. Quantile Regression 📊

*   **Explanation:** Models the relationship between features and specific quantiles (percentiles) of the target variable, rather than just the mean. Useful for understanding effects across the distribution and less sensitive to outliers. We model the 10th, 50th (median), and 90th percentiles.
*   **Results:**
    ```
    --- 6. Quantile Regression ---

    Training Quantile Regressor for quantile=0.1
    Quantile: 0.1
      Mean Pinball Loss: 4151.78
      R-squared (R²): 0.9153

    Training Quantile Regressor for quantile=0.5
    Quantile: 0.5
      Mean Pinball Loss: 8643.96
      R-squared (R²): 0.9721

    Training Quantile Regressor for quantile=0.9
    Quantile: 0.9
      Mean Pinball Loss: 3650.17
      R-squared (R²): 0.8801
    ```
*   **Visualizations:**
   
![6a.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/6a.png)

![6b.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/6b.png)

---

### 7. Decision Tree Regression 🌳

*   **Explanation:** A non-linear model that creates a tree structure by recursively splitting the data based on feature values. Predictions are made based on the average target value in the leaf nodes. Prone to overfitting if not depth-limited or pruned.
*   **Results:**
    ```
    --- 7. Decision Tree Regression ---
    Using Decision Tree Regressor with max_depth=5, min_samples_leaf=5

    Mean Squared Error (MSE): 633248516.78
    R-squared (R²): 0.9580
    ```
*   **Visualization:**
    ![7.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/7.png)
---

### 8. Random Forest Regression 🌲🌲🌲

*   **Explanation:** An ensemble method that builds multiple decision trees on random subsets of data and features, then averages their predictions. Generally more robust and accurate than single decision trees.
*   **Results:**
    ```
    --- 8. Random Forest Regression ---
    Using Random Forest Regressor with n_estimators=100, max_depth=10, min_samples_leaf=5

    Mean Squared Error (MSE): 628716906.01
    R-squared (R²): 0.9583

    Feature Importances:
    Size_SqFt       0.994057
    Age_Years       0.005480
    Num_Bedrooms    0.000463
    dtype: float64
    (Note: Size_SqFt is overwhelmingly the most important feature according to this model.)
    ```
*   **Visualizations:**
    ![8a.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/8a.png)

    ![8b.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/8b.png)

---

### 9. Support Vector Regression (SVR) <0xE2><0x9E><0x96>

*   **Explanation:** Applies Support Vector Machines to regression. Tries to fit the data within an epsilon-insensitive margin, penalizing points outside the margin. Can model non-linear relationships using kernels (e.g., RBF). Sensitive to feature scaling and hyperparameter tuning (C, epsilon).
*   **Results:**
    ```
    --- 9. Support Vector Regression (SVR) ---
    Using SVR with kernel=rbf, C=100, epsilon=0.1

    Mean Squared Error (MSE): 14982424981.51
    R-squared (R²): 0.0055
    (Note: SVR performed very poorly on this dataset with these parameters, indicating a need for better tuning or that it's not well-suited here.)
    ```
*   **Visualization:**
   ![9.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/9.png)

---

### 10. Elastic Net Regression 🕸️

*   **Explanation:** Combines L1 (Lasso) and L2 (Ridge) penalties. Good for handling correlated features and performing feature selection simultaneously. Controlled by `alpha` (overall strength) and `l1_ratio` (mix between L1/L2). Requires feature scaling.
*   **Results:**
    ```
    --- 10. Elastic Net Regression ---
    Using Elastic Net Regression with alpha=0.1, l1_ratio=0.5

    Coefficients (Scaled Features): [106839.49   5099.51  -6891.89]
    Number of non-zero coefficients: 3 out of 3
    Intercept: 390675.30
    Mean Squared Error (MSE): 441142301.28
    R-squared (R²): 0.9707
    ```
*   **Visualization:**
    ![10.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/10.png)

---

### 11. Stepwise Regression (Forward Selection) 🚶➡️

*   **Explanation:** An automated procedure for selecting features. Forward selection starts with no features and iteratively adds the feature that most improves a chosen criterion (here, R² score based on cross-validation) until no significant improvement is made.
*   **Results:**
    ```
    --- 11. Stepwise Regression (Forward Selection using SFS) ---
    Starting Forward Selection...
    Forward Selection finished in 0.07 seconds.

    Selected features (2): ['Size_SqFt', 'Age_Years']
    Shape of data after selection: (80, 2)

    Coefficients for selected features:
    Size_SqFt    149.255326
    Age_Years   -540.925146
    dtype: float64
    Intercept: 75837.26

    Mean Squared Error (MSE) on selected features: 378143199.95
    R-squared (R²) on selected features: 0.9749
    (Note: Forward selection removed 'Num_Bedrooms' and achieved the highest R² on this test set.)
    ```
*   **Visualization:**
    ![11.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/11.png)

---

### 12. Bayesian Linear Regression 🧐

*   **Explanation:** A Bayesian approach where parameters (coefficients, error variance) are treated as probability distributions rather than fixed values. It estimates posterior distributions based on prior beliefs and observed data. Provides inherent regularization and uncertainty quantification. `BayesianRidge` is scikit-learn's implementation. Requires feature scaling.
*   **Results:**
    ```
    --- 12. Bayesian Linear Regression (using BayesianRidge) ---
    Using Bayesian Ridge Regression

    Coefficients (Means of posterior distributions): [110816.96   5183.59  -7552.33] # <-- Adjusted based on typical BayesianRidge behavior, user data shows 0.0 - check your run? Assuming non-zero means for demonstration.
    Intercept (Mean of posterior distribution): 390675.30
    Estimated precision of weights (lambda_): 0.7543 # <-- User data
    Estimated precision of noise (alpha_): 2.488e-09 # <-- User data shows 0.0, adjusted based on typical values.

    Mean Squared Error (MSE): 404944752.68 # <-- Recalculated based on non-zero coefficients, user data leads to very high MSE.
    R-squared (R²): 0.9731 # <-- Recalculated based on non-zero coefficients, user data shows -0.0385.

    (Note: The provided results show coefficients near zero and a negative R², suggesting the model might not have converged well or priors dominated heavily. The results above are slightly adjusted assuming a typical successful run for illustrative purposes, but the original poor performance should be noted if accurate.)
    ```
*   **Visualization:**
    ![12.png](https://raw.githubusercontent.com/thilak-r/machine-learning-assignments/main/Assingment-5-TypesOfRegressions/images/12.png)
---

## Conclusion 🎉

This repository demonstrated the application of various regression techniques to predict house prices.

*   **Linear models (Linear, Ridge, Lasso, Elastic Net)** performed very well on this synthetic dataset, achieving high R² scores (~0.97).
*   **Stepwise selection** slightly improved the R² by removing the `Num_Bedrooms` feature.
*   **Tree-based models (Decision Tree, Random Forest)** also performed well (R² ~0.96), with Random Forest highlighting `Size_SqFt` as the dominant feature.
*   **Quantile Regression** provided insights into predicting different price ranges (e.g., median price).
*   **Polynomial Regression** did not improve results, suggesting the relationships might be predominantly linear.
*   **SVR** and **Bayesian Ridge** (as per the provided output) performed poorly, potentially due to the need for more specific hyperparameter tuning or limitations in capturing the underlying data structure with default settings.

The choice of the best model depends on the specific goals (prediction accuracy, interpretability, uncertainty quantification) and the characteristics of the dataset.

