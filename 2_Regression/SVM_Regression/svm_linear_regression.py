# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR

# Importing dataset
df = pd.read_csv("../SVM_Regression/Dataset/ecommerce.csv")

# Dividing dependent and independent variables
X = df[['Length of Membership']]
Y = df[['Yearly Amount Spent']]

# Checking for null values
missing_columns_X = [col for col in X.columns if X[col].isnull().any()]
missing_columns_Y = [col for col in Y.columns if Y[col].isnull().any()]

# Analyzing by visualizing
plt.scatter(X, Y, color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

# Creating model
svm_reg = LinearSVR(epsilon=1)
svm_reg.fit(X, Y['Yearly Amount Spent'])

# Predicting and plotting line
plt.scatter(X, Y, color='blue')
plt.plot(X, svm_reg.predict(X), color='green')
plt.title('Truth or Bluff (Support Vector Regression Model)')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()
