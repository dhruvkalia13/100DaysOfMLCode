import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Importing dataset
df = pd.read_csv("Dataset/iris.csv")

# Visualizing
fig = df[df.Species == 'Iris-setosa'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', color='orange', label='Setosa')
df[df.Species == 'Iris-versicolor'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', color='blue', label='versicolor',
                                                 ax=fig)
df[df.Species == 'Iris-virginica'].plot.scatter(x='PetalLengthCm', y='PetalWidthCm', color='green', label='virginica',
                                                ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title(" Petal Length VS Width")
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()

# Checking if data contains missing values
missing_cols = [col for col in df.columns if df[col].isnull().any()]
print(missing_cols)

# Dependent and Independent variables
X = df.iloc[:, [3, 4]]
y = df.iloc[:, [5]]

# Label encoding in categorical columns
labelled_y = y.copy()
label_encoder = LabelEncoder()
for col in y.columns:
    labelled_y[col] = label_encoder.fit_transform(y[col])

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, labelled_y, test_size=0.2, random_state=42)

# Pipeline to perform feature scaling and modelling
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
])
svm_pipeline.fit(X_train, y_train)

# Prediction and accuracy measurement
y_test_hat = svm_pipeline.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(plot_confusion_matrix(svm_pipeline, X_test, y_test, cmap=plt.cm.Blues,
                            display_labels=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))
