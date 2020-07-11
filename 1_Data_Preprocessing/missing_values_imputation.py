# Importing the libraries
import pandas as pd
from sklearn.impute import SimpleImputer

# Importing dataset
data = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Number of missing columns
number_of_missing_cols = len([col for col in data.columns if data[col].isnull().any()])

print("Number of missing columns before imputing is %s" % str(number_of_missing_cols))

# Creating imputer with a strategy of filling in the entries with most frequent values in the column
simple_imputer = SimpleImputer(strategy="most_frequent")
imputed_data = pd.DataFrame(simple_imputer.fit_transform(data))
imputed_data.columns = data.columns

# Number of missing columns
number_of_missing_cols = len([col for col in imputed_data.columns if imputed_data[col].isnull().any()])

print("Number of missing columns after imputing is %s" % str(number_of_missing_cols))
