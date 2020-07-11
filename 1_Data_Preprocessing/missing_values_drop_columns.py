# Importing the libraries
import pandas as pd

# Importing dataset
data = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Finding missing columns
missing_values_columns = [col for col in data.columns if data[col].isnull().any()]

print("Shape of data before dropping is %s" % str(data.shape))

# Dropping the columns with missing values
data = data.drop(missing_values_columns, axis=1)

print("Shape of data after dropping is %s" % str(data.shape))
