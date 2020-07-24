import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

# Importing dataset
df = pd.read_csv("Dataset/mushrooms.csv")

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

# Applying GridSearchCV to find the best hyperparameters for doing polynomial classification
svm = SVC()
parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100], 'gamma': [1e-3, 1e-4, 1e-5]},
              {'kernel': ['poly'], 'C': [1, 10, 100], 'degree': (2, 3, 4)}
              ]
clf = GridSearchCV(svm, parameters, cv=5, scoring="accuracy")
clf.fit(X_train, y_train)
print(clf.best_params_)

# Applying the best parameters in our model
svc = SVC(kernel='poly', C=10, degree=3)
svc.fit(X_train, y_train)
svc.score(X_test, y_test)

y_test_hat = svc.predict(X_test)
print(classification_report(y_test, y_test_hat))
print(plot_confusion_matrix(svc, X_test, y_test, cmap=plt.cm.Blues, display_labels=['Poison', 'No poison']))
