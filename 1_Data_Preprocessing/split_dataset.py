# Importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing dataset
df_X = pd.read_csv("../1_Data_Preprocessing/Dataset/sample_data.csv")

# Separating predicting value
df_Y = pd.DataFrame(df_X['SalePrice'])
df_X.drop('SalePrice', axis=1, inplace=True)

df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_X, df_Y, train_size=0.8, test_size=0.2, random_state=0)

print(df_X_train.shape)
print(df_X_test.shape)
print(df_Y_train.shape)
print(df_Y_test.shape)
