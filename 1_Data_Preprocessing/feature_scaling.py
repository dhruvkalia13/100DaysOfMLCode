# Importing libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Importing dataset
X = pd.read_csv("../1_Data_Preprocessing/Dataset/dataset_feature_scaling.csv")

# has the 4th column (Purchased)
df_Y = X.iloc[:, 3:4]

# has Age and Salary column
df_X = X.iloc[:, 1:3]

# Missing values
simple_imputer = SimpleImputer(strategy='mean')
imputed_X = pd.DataFrame(simple_imputer.fit_transform(df_X))
imputed_X.columns = df_X.columns

print(imputed_X)

# Min max normalization
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
df_X_normalized = min_max_scaler.fit_transform(imputed_X)
print(df_X_normalized)

# Standardization
standard_scaler = StandardScaler()
df_X_standardized = standard_scaler.fit_transform(imputed_X)
print(df_X_standardized)