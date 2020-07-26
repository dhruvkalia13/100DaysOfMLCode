import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


# Importing dataset
df = pd.read_csv("../SVM_Classifier/Dataset/mushrooms.csv")

# Checking for missing values
missing_values_cols = [col for col in df.columns if df[col].isnull().any()]
print(missing_values_cols)

# Label Encoding to categorical columns
df.head()
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
label_encoder = LabelEncoder()
labelled_df = df.copy()
for col in categorical_cols:
    labelled_df[col] = label_encoder.fit_transform(df[col])

# Checking for corelation between variables
correlation = labelled_df.corr()
plt.figure(figsize=(15, 15))
sbn.heatmap(correlation, annot=True, cmap=plt.cm.Blues)

# As viel-type is not related to any of the columns, removing it
labelled_df.drop('veil-type', axis=1, inplace=True)

# Dependent and Independent variables
y = labelled_df.iloc[:, 0]
X = labelled_df.iloc[:, 1:22]

# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying GridSearchCV to find the best hyperparameters for performing classification
model = DecisionTreeClassifier()
parameters = [{'max_depth': [1,2,3,4,5,6,7,8,9,10]}]
clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy")
clf.fit(X_train, y_train)
print(clf.best_params_)

# Applying the best parameters in our model
clf = DecisionTreeClassifier(max_depth=8)
clf.fit(X_train, y_train)

# Predicting
y_test_hat = clf.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues,
                            display_labels=['Poison', 'No poison']))

# Visualizing
plt.figure(figsize=[20, 10])
tree.plot_tree(clf, rounded= True, filled= True)
