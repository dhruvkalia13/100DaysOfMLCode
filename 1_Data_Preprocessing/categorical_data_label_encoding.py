# Importing the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Importing dataset
df = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Columns with missing values
cols_missing_values = [col for col in df.columns if df[col].isnull().any()]

# Dropping the columns with missing values
df = df.drop(cols_missing_values, axis=1)

# Categorical columns
df_categorical = df.select_dtypes('object')

print(df.head())

# Creating a copy of original df
labeled_df = df.copy()

# Applying Label Encoding to each column having categorical data
label_encoder = LabelEncoder()
for col in df_categorical.columns:
    labeled_df[col] = label_encoder.fit_transform(df[col])

print(labeled_df.head())

# Unique values in SaleCondition categorical column = [4 0 5 1 2 3]
print(df['SaleCondition'].unique())
print(labeled_df['SaleCondition'].unique())