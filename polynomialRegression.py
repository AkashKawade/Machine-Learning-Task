# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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


# Shape of train data
print(X_train.shape)
print(y_train.shape)


# Print training dataset
print("X train \n", X_train)
print("Y train \n", y_train)


# Print testing dataset
print("X test \n", X_test)
print("y test \n", y_test)


# Create a polynomial regression model with degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)

# Create a pipeline that first transforms the features and then applies linear regression
model = make_pipeline(poly, LinearRegression())

# Train the polynomial regression model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-squared
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

# R-squared as a percentage
r2_percentage = r2 * 100
print("R-squared (percentage): {:.2f}%".format(r2_percentage))

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_test, y_pred)
print("Mean Absolute Percentage Error:", mape)

# Print model coefficients (Note: coefficients are now for polynomial features)
print("Model coefficients:", model.named_steps['linearregression'].coef_)
print("Intercept:", model.named_steps['linearregression'].intercept_)

# Plotting predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', label='Ideal Fit')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True Values vs Predictions (Polynomial Regression)')
plt.legend()
plt.show()
