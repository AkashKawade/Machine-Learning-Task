# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge

warnings.filterwarnings('ignore')


# Importing dataset
data = pd.read_csv('Boston_Housing.csv')


# View of data
print(data.head())
print(data.tail())


# Shape of dataset
print("Columns:", data.shape[1])
print("Rows:", data.shape[0])


# Info of dataset
data.info()


# Null Values in dataset
print("Null values:\n", data.isna().sum())


# Stats of dataset
print(data.describe())


# Duplicate values in dataset
print("Duplicate values:", data.duplicated().sum())


# Features of dataset
print(data.columns)


# Distribution of Independent variable
sns.displot(data['RM'])
plt.show()


# Distribution of Dependent variable
sns.displot(data['MEDV'])
plt.show()


# Features and target variable
X = data[['RM']]
y = data['MEDV']


# Splitting the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print(f"Model: {model.__class__.__name__}")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)
    print("R-squared (percentage): {:.2f}%".format(r2 * 100))
    print("Mean Absolute Error:", mae)
    print("Mean Absolute Percentage Error:", mape)
    print()

# Models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1)
}

# Evaluate each model
for name, model in models.items():
    print(f"Evaluating {name}...")
    evaluate_model(model, X_train, X_test, y_train, y_test)

# Selecting the best model based on evaluation
# For this example, we'll use Ridge Regression as the best model
best_model = Ridge(alpha=1.0)  # Replace with the best-performing model based on evaluation
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Plotting predictions vs true values for the best model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Best Model)')
plt.legend()
plt.show()
