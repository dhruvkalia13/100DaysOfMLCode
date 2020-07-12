# Importing the libraries
import pandas as pd

# Importing dataset
df = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Columns with missing values
cols_missing_values = [col for col in df.columns if df[col].isnull().any()]

# Dropping the columns with missing values
df = df.drop(cols_missing_values, axis=1)

# Categorical columns
df_categorical = df.select_dtypes('object')

print(df.shape)

# Dropping categorical columns (For this dataset, we are assuming that 'object' dtypes are categorical
df = df.drop(df_categorical, axis=1)

print(df.shape)