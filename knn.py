# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importing the dataset
veri = pd.read_csv(r"C:\Users\CASPER\OneDrive\Masaüstü\Veri Madenciliği\tümVeriler.csv", delimiter=";", decimal=',')

# Selecting the relevant columns
X = veri.iloc[:, 1:-1].values  # Using the first two features for visualization
y = veri.iloc[:, -1].values  # Dependent variable

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standard Scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

# Fitting K-Nearest Neighbors to the Training set with Standard Scaling
classifier_scaled = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier_scaled.fit(X_train_scaled, y_train)

# Predicting the test set results
y_pred_knn_scaled = classifier_scaled.predict(X_test_scaled)

# Calculate and print metrics for K-Nearest Neighbors with Standard Scaling
accuracy_knn_scaled = accuracy_score(y_test, y_pred_knn_scaled)
precision_knn_scaled = precision_score(y_test, y_pred_knn_scaled, average='binary')
recall_knn_scaled = recall_score(y_test, y_pred_knn_scaled, average='binary')
f1_knn_scaled = f1_score(y_test, y_pred_knn_scaled, average='binary')

print("K-Nearest Neighbors with Standard Scaling:")
print(f"Accuracy: {accuracy_knn_scaled:.2f}")
print(f"Precision: {precision_knn_scaled:.2f}")
print(f"Recall: {recall_knn_scaled:.2f}")
print(f"F1 Score: {f1_knn_scaled:.2f}")
