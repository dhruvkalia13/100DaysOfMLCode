import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix

# Importing dataset
df = pd.read_csv("Dataset/glass.csv")
print(df.describe())

# Dependent and Independent variables
X = df.iloc[:, 0:9]
y = df.iloc[:, 9]

# Analysing
correlation=df.corr()
plt.figure(figsize=(10,10))
sbn.heatmap(correlation,annot=True,cmap=plt.cm.Blues)

# Applying GridSearchCV to find the best hyperparameters
model = RandomForestClassifier()
# Applying GridSearchCV to find the best hyperparameters for doing random forest classification
parameters = [{'n_estimators': [10, 20, 50]}]
clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy")
clf.fit(X, y)
print(clf.best_params_)

# Creating model using best parameters
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X, y)

# Predicting and evaluating its accuracy
y_hat = clf.predict(X)
print(classification_report(y, y_hat))
print(plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues,
                            display_labels=(y.unique())))