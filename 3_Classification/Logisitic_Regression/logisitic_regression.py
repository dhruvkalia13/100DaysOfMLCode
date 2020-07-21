import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv("Dataset/diabetes.csv")
print(df.columns)

# Checking for columns which have missing values
missing_cols = [col for col in df.columns if df[col].isnull().any()]
print(missing_cols)

# Independent and dependent variables
X = pd.DataFrame(df.iloc[:, 0:8])
y = pd.DataFrame(df.iloc[:, 8])

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modelling
clf = LogisticRegression(max_iter=250)
clf.fit(X_train, y_train['Outcome'])

# Predicting on testing dataset
y_test_hat = clf.predict(X_test)
print(classification_report(y_test, y_test_hat))

# Confusion matrix for testing dataset
plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Diabetic', 'Non-diabetic'])

# Predicting on training dataset
y_train_hat = clf.predict(X_train)
print(classification_report(y_train, y_train_hat))

# Confusion matrix for testing dataset
plot_confusion_matrix(clf, X_train, y_train, cmap=plt.cm.Blues, display_labels=['Diabetic', 'Non-diabetic'])
