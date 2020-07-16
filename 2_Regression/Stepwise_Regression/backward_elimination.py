import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../Simple_Linear_Regression/water.csv")

num_col = df[['Cst_Cnt', 'Btl_Cnt', 'Depthm', 'Salnty']]
y = df[['T_degC']]

# Creating imputer with a strategy of filling in the entries with most frequent values in the column
simple_imputer = SimpleImputer(strategy='mean')
imputed_data_X = pd.DataFrame(simple_imputer.fit_transform(num_col))
imputed_data_Y = pd.DataFrame(simple_imputer.fit_transform(y))

imputed_data_X.columns = num_col.columns
imputed_data_Y.columns = y.columns

# Min max normalization
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df_X_normalized = pd.DataFrame(min_max_scaler.fit_transform(imputed_data_X))

X = np.append(arr=np.ones((1000, 1)).astype(int), values=df_X_normalized, axis=1)
X_opt = X[:, [0, 1, 2, 3]]
regressor_OLS = sm.OLS(endog=imputed_data_Y, exog=X_opt).fit()
print(regressor_OLS.pvalues)

X = np.append(arr=np.ones((1000, 1)).astype(int), values=df_X_normalized, axis=1)
X_opt = X[:, [1, 2, 3]]
regressor_OLS = sm.OLS(endog=imputed_data_Y, exog=X_opt).fit()
print(regressor_OLS.pvalues)
#
X = np.append(arr=np.ones((1000, 1)).astype(int), values=df_X_normalized, axis=1)
X_opt = X[:, [1, 3]]
regressor_OLS = sm.OLS(endog=imputed_data_Y, exog=X_opt).fit()
print(regressor_OLS.pvalues)
#
X = np.append(arr=np.ones((1000, 1)).astype(int), values=df_X_normalized, axis=1)
X_opt = X[:, [3]]
regressor_OLS = sm.OLS(endog=imputed_data_Y, exog=X_opt).fit()
print(regressor_OLS.pvalues)
