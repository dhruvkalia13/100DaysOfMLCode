# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# Importing dataset
complete_bottle_data = pd.read_csv("../Simple_Linear_Regression/water.csv")
partial_bottle_data = complete_bottle_data.loc[1:1000, ['T_degC', 'Salnty']]
print(partial_bottle_data.head())

# Creating train and test dataset
msk = np.random.rand(len(partial_bottle_data)) < 0.8
train_raw = partial_bottle_data[msk]
test_raw = partial_bottle_data[~msk]

# Fixing missing values issue and imputing
imputer = SimpleImputer()
train = pd.DataFrame(imputer.fit_transform(train_raw))
train.columns = partial_bottle_data.columns
train.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY'}, inplace=True)
train = train.reindex(columns={'SALINITY', 'TEMP'})
print(train.head())

# Analyzing dataset
plt.scatter(train.SALINITY, train.TEMP, color='blue')
plt.xlabel("SALINITY")
plt.ylabel("TEMP")
# plt.show()

# Modelling
train_x = np.asanyarray(train[['SALINITY']])
train_y = np.asanyarray(train[['TEMP']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

clf = linear_model.LinearRegression()
train_y_hat = clf.fit(train_x_poly, train_y)
# The coefficients
print('Coefficients: ', clf.coef_)
print('Intercept: ', clf.intercept_)

plt.scatter(train.SALINITY, train.TEMP, color='blue')
XX = np.arange(32.63, 34.65, 0.2)
# yy = clf.intercept_[0]+ clf.coef_[0][1]*XX
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * np.power(XX, 2)
plt.plot(XX, yy, '-r')
plt.xlabel("SALINITY")
plt.ylabel("TEMP")
plt.show()

# Predicting begins
# Applying same Fixing missing values issue and imputing
imputer = SimpleImputer()
test = pd.DataFrame(imputer.fit_transform(test_raw))
test.columns = partial_bottle_data.columns
test.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY'}, inplace=True)
test = test.reindex(columns={'SALINITY', 'TEMP'})
test.shape

test_x = np.asanyarray(test[['SALINITY']])
test_y = np.asanyarray(test[['TEMP']])

test_x_poly = poly.fit_transform(test_x)
test_y_hat = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
