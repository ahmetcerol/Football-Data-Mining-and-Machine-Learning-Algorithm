# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing the dataset
veri = pd.read_csv(r"C:\Users\CASPER\OneDrive\Masaüstü\Veri Madenciliği\tümVeriler.csv", delimiter=";", decimal=',')

# Selecting the relevant columns
X = veri.iloc[:, 1:-1].values  # Using the first two features for visualization
y = veri.iloc[:, -1].values  # Dependent variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

# Robust Scaling
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

# Fitting Naive Bayes to the Training set with Min-Max Scaling
classifier_minmax = GaussianNB()
classifier_minmax.fit(X_train_minmax, y_train)

# Predicting the test set results
y_pred_nb_minmax = classifier_minmax.predict(X_test_minmax)

# Calculate and print metrics for Naive Bayes with Min-Max Scaling
accuracy_nb_minmax = accuracy_score(y_test, y_pred_nb_minmax)
precision_nb_minmax = precision_score(y_test, y_pred_nb_minmax, average='binary')
recall_nb_minmax = recall_score(y_test, y_pred_nb_minmax, average='binary')
f1_nb_minmax = f1_score(y_test, y_pred_nb_minmax, average='binary')

print("Naive Bayes with Min-Max Scaling:")
print(f"Accuracy: {accuracy_nb_minmax:.2f}")
print(f"Precision: {precision_nb_minmax:.2f}")
print(f"Recall: {recall_nb_minmax:.2f}")
print(f"F1 Score: {f1_nb_minmax:.2f}")

# Fitting Naive Bayes to the Training set with Robust Scaling
classifier_robust = GaussianNB()
classifier_robust.fit(X_train_robust, y_train)

# Predicting the test set results
y_pred_nb_robust = classifier_robust.predict(X_test_robust)

# Calculate and print metrics for Naive Bayes with Robust Scaling
accuracy_nb_robust = accuracy_score(y_test, y_pred_nb_robust)
precision_nb_robust = precision_score(y_test, y_pred_nb_robust, average='binary')
recall_nb_robust = recall_score(y_test, y_pred_nb_robust, average='binary')
f1_nb_robust = f1_score(y_test, y_pred_nb_robust, average='binary')

print("\nNaive Bayes with Robust Scaling:")
print(f"Accuracy: {accuracy_nb_robust:.2f}")
print(f"Precision: {precision_nb_robust:.2f}")
print(f"Recall: {recall_nb_robust:.2f}")
print(f"F1 Score: {f1_nb_robust:.2f}")
