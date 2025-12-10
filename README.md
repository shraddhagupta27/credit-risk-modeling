# credit-risk-modeling

# Credit Risk Modeling
Machine learning–based credit risk model to classify loan applicants into approval categories and support automated lending decisions.

## Overview
This project builds a complete data science pipeline to assess the creditworthiness of loan applicants using customer demographic, financial, and bureau information. It includes data cleaning, feature engineering, exploratory data analysis, multicollinearity checks, statistical testing, model training, and evaluation. The final model predicts whether a customer should be approved for a loan, enabling faster and more consistent decision-making.

## Problem Statement
Financial institutions must evaluate loan applicants accurately to minimize default risk while approving credit for reliable customers. Manual underwriting is slow, inconsistent, and prone to bias.
This project uses machine learning to predict loan approval categories based on customer data, improving both operational efficiency and risk accuracy.

## Dataset

The dataset contains two primary files:
Customer Application Data - Income details, Loan amount, Asset ownership
Customer Bureau Data - Credit history, Past loan performance, Overdues, defaults

## Tools & Technologies
- Python: pandas, numpy, matplotlib, seaborn, statsmodels, scipy
- Machine Learning: scikit-learn, XGBoost, Decision tree, Random forest
- EDA & Feature Selection: VIF, Chi-Square Test, ANOVA


## Methods

- Merged customer application & bureau datasets
- Performed EDA to identify trends, correlations, and risk patterns
- Handled missing values and outliers through imputation and filtering
- Checked multicollinearity using VIF and removed redundant features
- Applied Chi-square and ANOVA tests for feature importance
- Encoded categorical features & scaled numerical variables
- Trained multiple ML models – Decision Tree, XGBoost
- Evaluated model performance using Accuracy, Confusion Matrix, and F1-score

 ## Key Insights

- Statistical tests revealed income, spending behavior, and credit history as the strongest predictors of approval categories.
- The baseline Decision Tree achieved 71% accuracy, while XGBoost improved it to ~78% after tuning.
- F1-score for key approval categories reached 0.87, making the model reliable for high-confidence decisions.



