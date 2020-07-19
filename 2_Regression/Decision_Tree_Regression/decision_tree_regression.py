# Importing Libraries
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing dataset
df = pd.read_csv("../SVM_Regression/Dataset/Position_Salaries.csv")

# Dividing dependent and independent variables
X = df[['Level']]
y = df[['Salary']]

# Checking for null values
missing_columns_X = [col for col in X.columns if X[col].isnull().any()]
missing_columns_Y = [col for col in y.columns if y[col].isnull().any()]

# Creating model
tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X, y)

# Predicting and plotting line
y_hat = tree_reg.predict(X)
plt.scatter(X, y['Salary'], color='blue')
plt.plot(X, y_hat, color='red')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

# Decision Tree
plt.figure(figsize=[20, 10])
tree.plot_tree(tree_reg, rounded= True, filled= True)
