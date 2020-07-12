# Importing the libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Importing dataset
df = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Columns with missing values
cols_missing_values = [col for col in df.columns if df[col].isnull().any()]

# Dropping the columns with missing values
df = df.drop(cols_missing_values, axis=1)

# Categorical columns
df_categorical = df.select_dtypes('object')

# One hot encoder which will return a numpy array instead of a sparse matrix and it will ignore if validating data
# doesn't have the encoded new columns
OH_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

OH_cols_df = pd.DataFrame(OH_encoder.fit_transform(df_categorical))

# Adding back the indexes
OH_cols_df.index = df.index

# Removing categorical columns as new encoded columns will be added
df_numerical = df.drop(df_categorical.columns, axis=1)

# Adding encoded columns
OH_df = pd.concat([df_numerical, OH_cols_df], axis=1)

print(OH_df.head())