# Importing Libraries
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Importing dataset
df = pd.read_csv("../SVM_Regression/Dataset/Position_Salaries.csv")

# Dividing dependent and independent variables
X = df[['Level']]
Y = df[['Salary']]

# Checking for null values
missing_columns_X = [col for col in X.columns if X[col].isnull().any()]
missing_columns_Y = [col for col in Y.columns if Y[col].isnull().any()]

# Analyzing by visualizing
plt.scatter(X, Y, color='blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Creating model
svm_poly_reg = SVR(kernel="poly", degree=5, C=100, epsilon=0.1)
svm_poly_reg.fit(X, Y['Salary'])

# Predicting and plotting line
plt.scatter(X, Y['Salary'], color='blue')
plt.plot(X, svm_poly_reg.predict(X), color='red')
plt.title('Truth or Bluff (Support Vector Regression Model)')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
