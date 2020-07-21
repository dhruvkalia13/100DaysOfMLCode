# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Importing dataset
df = pd.read_csv("../Simple_Linear_Regression/water.csv")
df = df.loc[1:1000, ['T_degC', 'Salnty']]

# Dividing dependent and independent variables
train = df[['Salnty']]
test = df[['T_degC']]
df.drop
# Fixing missing values issue and imputing
imputer = SimpleImputer(strategy="mean")
train_imputed = pd.DataFrame(imputer.fit_transform(train))
test_imputed = pd.DataFrame(imputer.fit_transform(test))
train_imputed.columns = train_imputed.columns
test_imputed.columns = test.columns

# Splitting into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(train_imputed, test_imputed, test_size=0.2, random_state=0)

# Define model
rfg = RandomForestRegressor(n_estimators=100, random_state=0)
rfg.fit(X_train, y_train['T_degC'])

# Predicting and plotting
test_y_hat = rfg.predict(X_test)
plt.scatter(X_test, y_test, color='blue')
plt.scatter(X_test, test_y_hat, color='red')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel("Salinity")
plt.ylabel("Temparature")
plt.show()

# Accuracy measures
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - np.asanyarray(y_test))))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - np.asanyarray(y_test)) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, np.asanyarray(y_test)))
