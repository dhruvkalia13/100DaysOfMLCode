import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

df = pd.read_csv("../input/gooogle-stock-price/Google_Stock_Price_Train.csv")
df_test = pd.read_csv("/kaggle/input/gooogle-stock-price/Google_Stock_Price_Test.csv")

train = df.loc[:, ["Open"]].values
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)

plt.plot(train_scaled)
plt.show()

X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 1258):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

y_train = y_train.reshape(-1, 1)

regressor = Sequential()

regressor.add(SimpleRNN(45, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.15))

regressor.add(SimpleRNN(45, return_sequences=True))
regressor.add(Dropout(0.15))

regressor.add(SimpleRNN(45, return_sequences=True))
regressor.add(Dropout(0.15))

regressor.add(SimpleRNN(units=45))
regressor.add(Dropout(0.15))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=30, batch_size=32)

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(X_train, y_train, epochs=30, batch_size=32)

real_values = df_test.loc[:, ["Open"]].values
dataset_total = pd.concat((df["Open"], df_test["Open"]), axis=0)
inputs = dataset_total[len(df) - len(df_test) - timesteps:].values.reshape(-1, 1)
inputs = scaler.transform(inputs)
inputs.shape

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i - timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

plt.plot(real_values, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
