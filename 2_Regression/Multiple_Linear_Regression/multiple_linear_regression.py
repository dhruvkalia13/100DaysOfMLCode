# Importing libraries
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn import linear_model

# Importing dataset
complete_bottle_data = pd.read_csv("../Simple_Linear_Regression/water.csv")
partial_bottle_data = complete_bottle_data.loc[1:1000, ['T_degC', 'Salnty', 'Depthm']]
print(partial_bottle_data.head())

# Creating train and test dataset
msk = np.random.rand(len(partial_bottle_data)) < 0.8
train_raw = partial_bottle_data[msk]
test_raw = partial_bottle_data[~msk]

# Fixing missing values issue and imputing
imputer = SimpleImputer()
train = pd.DataFrame(imputer.fit_transform(train_raw))
train.columns = partial_bottle_data.columns
train.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY', 'Depthm': 'DEPTH'}, inplace=True)
train = train.reindex(columns={'SALINITY', 'DEPTH', 'TEMP'})
print(train.head())

# Modelling
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['SALINITY', 'DEPTH']])
train_y = np.asanyarray(train[['TEMP']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Predicting begins
# Applying same Fixing missing values issue and imputing
imputer = SimpleImputer()
test = pd.DataFrame(imputer.fit_transform(test_raw))
test.columns = partial_bottle_data.columns
test.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY', 'Depthm': 'DEPTH'}, inplace=True)
test = test.reindex(columns={'SALINITY', 'DEPTH', 'TEMP'})
print(test.shape)

test_x = np.asanyarray(test[['SALINITY', 'DEPTH']])
test_y = np.asanyarray(test[['TEMP']])
test_y_hat = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))
