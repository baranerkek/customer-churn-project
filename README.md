# Customer Churn Prediction Using Logistic Regression

## Project Overview

This project develops a machine learning model to predict whether a customer will leave a company, also known as **customer churn**.

Customer churn prediction helps businesses identify customers who are likely to cancel their service. Companies can use these predictions to create retention strategies, offer personalized promotions, and reduce customer loss.

The project uses **Logistic Regression**, a supervised machine learning algorithm commonly used for binary classification problems.

The target variable has two possible outcomes:

* `0`: The customer does not churn
* `1`: The customer churns

## Project Structure

```text
churn-rate-logistic-regression/
│
├── data/
│   └── churn.csv
│
├── notebooks/
│   └── churn_logistic_regression.ipynb
│
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

The exact folder names may differ depending on the project structure.

## Dataset

The dataset contains customer information such as demographic characteristics, account information, subscription details, and service usage.

The target variable is:

```text
churn
```

Example predictor variables may include:

* Customer age
* Contract type
* Monthly charges
* Total charges
* Subscription duration
* Payment method
* Internet service type
* Customer support usage

The dataset used in this project is located in the `data` folder.

## Technologies Used

The project was developed using:

* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook
* Visual Studio Code
* Astral uv

## Project Workflow

The project follows these main steps:

1. Importing the required Python libraries
2. Loading and inspecting the dataset
3. Cleaning column names
4. Handling missing values
5. Exploring the target variable
6. Separating numerical and categorical variables
7. Encoding categorical variables with one-hot encoding
8. Scaling numerical variables
9. Splitting the data into training and test sets
10. Training a Logistic Regression model
11. Making predictions on the test dataset
12. Evaluating the model performance

## Data Preprocessing

Before training the model, the data is prepared using several preprocessing techniques.

### Missing Values

Missing values are identified and handled depending on the variable type.

Numerical missing values can be filled using the median, while categorical missing values can be filled using the most frequent category.

### One-Hot Encoding

Categorical variables cannot be used directly by most machine learning algorithms.

Therefore, categorical features are transformed into numerical dummy variables using **OneHotEncoder**.

### Feature Scaling

Numerical variables may have different measurement scales. For example, customer age and total charges may have very different ranges.

The numerical features are standardized using **StandardScaler**.

### Train-Test Split

The dataset is divided into training and test sets.

```text
Training set: 80%
Test set: 20%
```

The training set is used to train the model, while the test set is used to evaluate its performance on unseen data.

## Model

The model used in this project is Logistic Regression.

Logistic Regression estimates the probability that a customer belongs to the churn class.

```python
LogisticRegression(max_iter=1000)
```

The `max_iter` parameter is increased to help the optimization algorithm converge successfully.

## Model Evaluation

The model is evaluated using the following classification metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC score

### Accuracy

Accuracy represents the percentage of all customers that were classified correctly.

### Precision

Precision measures how many customers predicted as churners actually churned.

### Recall

Recall measures how many actual churners were correctly identified by the model.

Recall is especially important in churn prediction because failing to identify a customer who is likely to leave may result in losing that customer.

### F1-Score

The F1-score provides a balance between precision and recall.

### Confusion Matrix

The confusion matrix shows:

* True negatives
* False positives
* False negatives
* True positives

### Model Results

The final model produced the following results:

```text
Accuracy:  [Add result]
Precision: [Add result]
Recall:    [Add result]
F1-score:  [Add result]
ROC-AUC:   [Add result]
```

These results should be updated after running the final model.

## Possible Improvements

The project can be improved by:

* Performing hyperparameter optimization
* Comparing Logistic Regression with Decision Trees
* Comparing the model with Random Forest and XGBoost
* Adjusting the classification threshold
* Handling class imbalance
* Applying cross-validation
* Examining feature importance using Logistic Regression coefficients
* Deploying the model with Flask or FastAPI

## Conclusion

This project demonstrates how Logistic Regression can be used to predict customer churn.

The model creates a baseline classification system that can identify customers who may leave the company. The predictions can help businesses take preventive actions and improve customer retention.


