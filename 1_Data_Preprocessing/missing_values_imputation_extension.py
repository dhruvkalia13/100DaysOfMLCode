# Importing the libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing dataset
data = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Creating copy so that original dataframe is not affected
data_copy = data.copy()

# Columns with missing values
cols_missing_values = [col for col in data_copy.columns if data_copy[col].isnull().any()]

for col in cols_missing_values:
    data_copy[col + "_was_missing"] = data_copy[col].isnull()

# Number of missing columns
number_of_missing_cols = len(cols_missing_values)

print("Number of missing columns before imputing in data_copy is %s" % str(number_of_missing_cols))

# Creating imputer with a strategy of filling in the entries with most frequent values in the column
simple_imputer = SimpleImputer(strategy="most_frequent")
imputed_data = pd.DataFrame(simple_imputer.fit_transform(data_copy))
imputed_data.columns = data_copy.columns

# Number of missing columns
number_of_missing_cols = len([col for col in imputed_data.columns if imputed_data[col].isnull().any()])

print("Number of missing columns after imputing in data_copy is %s" % str(number_of_missing_cols))
