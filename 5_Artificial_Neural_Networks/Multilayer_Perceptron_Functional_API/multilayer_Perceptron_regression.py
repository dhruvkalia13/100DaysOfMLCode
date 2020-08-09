import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.python import keras
import tensorflow as tf
import os
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

# Importing data
from tensorflow.python.layers.normalization import BatchNormalization

df = pd.read_csv("../Multilayer_Perceptron_Sequential_API/Dataset/water.csv")
df = df.loc[1:1000, ['T_degC', 'Salnty', 'Depthm']]
df.head()

# Creating train and test dataset
msk = np.random.rand(len(df)) < 0.8
train_raw = df[msk]
test_raw = df[~msk]

# Fixing missing values issue and imputing
imputer = SimpleImputer()
train = pd.DataFrame(imputer.fit_transform(train_raw))
train.columns = df.columns
train.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY', 'Depthm': 'DEPTH'}, inplace=True)
train = train.reindex(columns={'SALINITY', 'DEPTH', 'TEMP'})
print(train.head())

train_x = train[['SALINITY', 'DEPTH']]
train_y = train[['TEMP']]

# Splitting into train and test dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(train_x, train_y, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[2]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(BatchNormalization())
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model


keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model)
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])

model = build_model(n_hidden=rnd_search_cv.best_params_['n_hidden'], n_neurons=rnd_search_cv.best_params_['n_neurons'],
                    learning_rate=rnd_search_cv.best_params_['learning_rate'], input_shape=[2])

# Fitting
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))

# Evaluating
mse_test = model.evaluate(X_test, y_test)

root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid), callbacks=[tensorboard_cb])
mse_test = model.evaluate(X_test, y_test)

# # Predicting
pred = model.predict(X_test)

# Accuracy measures
print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - np.asanyarray(y_test))))
print("Residual sum of squares (MSE): %.2f" % np.mean((pred - np.asanyarray(y_test)) ** 2))
print("R2-score: %.2f" % r2_score(pred, np.asanyarray(y_test)))

# Plotting
pred = model.predict(X_test)
plt.scatter(pd.DataFrame(X_test[:, 0]), y_test, color='blue')
plt.scatter(pd.DataFrame(X_test[:, 0]), pred, color='red')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel("Salinity")
plt.ylabel("Temparature")
plt.show()
