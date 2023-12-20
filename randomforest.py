# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

from sklearn.ensemble import RandomForestClassifier

# Hyperparameter Tuning using GridSearchCV for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Print the best parameters found by the grid search for Random Forest
print("Best Parameters (Random Forest):", grid_search_rf.best_params_)

# Get the best Random Forest classifier from the grid search
best_classifier_rf = grid_search_rf.best_estimator_

# Predicting the test set results with Random Forest
y_pred_rf = best_classifier_rf.predict(X_test)

# Calculate and print metrics for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='binary')
recall_rf = recall_score(y_test, y_pred_rf, average='binary')
f1_rf = f1_score(y_test, y_pred_rf, average='binary')

print(f"Random Forest - Improved Accuracy: {accuracy_rf:.2f}")
print(f"Random Forest - Improved Precision: {precision_rf:.2f}")
print(f"Random Forest - Improved Recall: {recall_rf:.2f}")
print(f"Random Forest - Improved F1 Score: {f1_rf:.2f}")
