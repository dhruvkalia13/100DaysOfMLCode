import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras as keras

df = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")
df.head()

missing_cols = [col for col in df.columns if df[col].isnull().any()]
missing_cols

# Categorical columns
df_categorical = df.select_dtypes('object')

df_c = df.copy()
label_encoder = LabelEncoder()
for col in df_categorical.columns:
    df_c[col] = label_encoder.fit_transform(df[col])

y = df_c[['Exited']]
df_c.drop('Exited', axis=1, inplace=True)
X = df_c

X_train_c, X_test_c, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standardization
standard_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train_c))
X_test = pd.DataFrame(standard_scaler.fit_transform(X_test_c))
print(X_train.head())

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[X_train.shape[1]]))
model.add(keras.layers.Dense(100, kernel_initializer="normal", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(50, kernel_initializer="normal", activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(2, activation="softmax"))
s = 20 * len(X_train)
learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
optimizer = keras.optimizers.SGD(learning_rate)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

history=model.fit(X_train, y_train, epochs=100, batch_size=32)

model.evaluate(X_test, y_test)