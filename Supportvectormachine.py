# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing the dataset
veri = pd.read_csv(r"C:\Users\CASPER\OneDrive\Masaüstü\Veri Madenciliği\tümVeriler.csv", delimiter=";", decimal=',')

# Selecting the relevant columns
X = veri.iloc[:, 1:-1].values  # Using the first two features for visualization
y = veri.iloc[:, -1].values  # Dependent variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Hyperparameter Tuning using GridSearchCV for Support Vector Machine
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto']
}

grid_search_svm = GridSearchCV(estimator=SVC(random_state=0), param_grid=param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)

# Print the best parameters found by the grid search for Support Vector Machine
print("Best Parameters (Support Vector Machine):", grid_search_svm.best_params_)

# Get the best Support Vector Machine classifier from the grid search
best_classifier_svm = grid_search_svm.best_estimator_

# Predicting the test set results with Support Vector Machine
y_pred_svm = best_classifier_svm.predict(X_test)

# Calculate and print metrics for Support Vector Machine
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='binary')
recall_svm = recall_score(y_test, y_pred_svm, average='binary')
f1_svm = f1_score(y_test, y_pred_svm, average='binary')

print(f"Support Vector Machine - Improved Accuracy: {accuracy_svm:.2f}")
print(f"Support Vector Machine - Improved Precision: {precision_svm:.2f}")
print(f"Support Vector Machine - Improved Recall: {recall_svm:.2f}")
print(f"Support Vector Machine - Improved F1 Score: {f1_svm:.2f}")

