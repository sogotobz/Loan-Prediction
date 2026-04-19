# Loan-Prediction
Loan Approval Prediction using Python, Machine Learning & Data Analysis
# Loan Default Prediction

This project builds a logistic regression model to predict whether a loan will **default** based on applicant income, loan amount, and education level. It includes basic exploratory data analysis (EDA), data visualization, feature engineering, model training, and evaluation.

---

## 📂 Dataset

The dataset is loaded from:

```python
Loan_default.csv
Key columns used in this project:

Income — Applicant’s income

LoanAmount — Amount of the loan

Education — Applicant’s education level (categorical)

Default — Target variable (1 = default, 0 = no default)

Basic checks performed:

head() — preview of the data

columns — list of columns

describe() — summary statistics

info() — data types and non‑null counts

isnull().sum() — missing value counts

Income summary: sum, mean, median

Age distribution via value_counts() (if present in the dataset)

📊 Exploratory Data Analysis & Visualization
The following visualizations are created:

Distribution of Loan Amount

python
sns.histplot(df['LoanAmount'], bins=30, kde=True)
Shows how loan amounts are distributed across applicants.

Income vs Loan Amount by Education

python
sns.scatterplot(data=df, x='Income', y='LoanAmount', hue='Education')
Explores the relationship between income and loan amount, segmented by education level.

🛠️ Feature Engineering
Features and target:

python
X = df[['Income', 'LoanAmount', 'Education']]
y = df['Default']
Categorical encoding:

python
X = pd.get_dummies(X, columns=['Education'], drop_first=True)
This converts the Education column into dummy variables for use in the logistic regression model.

🤖 Model Training
Train–test split:

python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
Model:

python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
📈 Model Evaluation
Predictions:

python
y_pred = model.predict(X_test)
Coefficients:

python
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
print(coefficients)
Classification metrics:

python
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, classification_report
)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
These metrics provide insight into how well the model distinguishes between default and non‑default cases.

👤 Author
Toba  
Data Science & Analytics
Winnipeg, Mb
