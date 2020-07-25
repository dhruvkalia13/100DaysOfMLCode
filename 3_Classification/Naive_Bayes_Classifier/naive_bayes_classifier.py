import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Importing dataset
df = pd.read_csv("../Dataset/gender.csv")
print(df.describe())

# After checking the description of the table, it looks like there are no missing columns and all columns are
# categorical

# Dependent and Independent Variables
X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

# Creating pipeline for handling categorical columns, apply feature scaling on them and modelling through GaussianNB
# later on.

clf = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ("scaler", StandardScaler()),
    ("model", GaussianNB()),
])

clf.fit(X, y)
print(classification_report(y, clf.predict(X)))
print(plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues,
                            display_labels=['Male', 'Female']))
