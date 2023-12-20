# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

# Standard Scaling
sc_X = StandardScaler()
X_train_scaled = sc_X.fit_transform(X_train)
X_test_scaled = sc_X.transform(X_test)

# Decision Tree with Min-Max Scaling
classifier_minmax = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_minmax.fit(X_train_minmax, y_train)
y_pred_dt_minmax = classifier_minmax.predict(X_test_minmax)

# Decision Tree with Robust Scaling
classifier_robust = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_robust.fit(X_train_robust, y_train)
y_pred_dt_robust = classifier_robust.predict(X_test_robust)

# Decision Tree with Standard Scaling
classifier_scaled = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_scaled.fit(X_train_scaled, y_train)
y_pred_dt_scaled = classifier_scaled.predict(X_test_scaled)

# Calculate and print metrics for Decision Tree with Min-Max Scaling
accuracy_dt_minmax = accuracy_score(y_test, y_pred_dt_minmax)
precision_dt_minmax = precision_score(y_test, y_pred_dt_minmax, average='binary')
recall_dt_minmax = recall_score(y_test, y_pred_dt_minmax, average='binary')
f1_dt_minmax = f1_score(y_test, y_pred_dt_minmax, average='binary')

print("Decision Tree with Min-Max Scaling:")
print(f"Accuracy: {accuracy_dt_minmax:.2f}")
print(f"Precision: {precision_dt_minmax:.2f}")
print(f"Recall: {recall_dt_minmax:.2f}")
print(f"F1 Score: {f1_dt_minmax:.2f}")

# Calculate and print metrics for Decision Tree with Robust Scaling
accuracy_dt_robust = accuracy_score(y_test, y_pred_dt_robust)
precision_dt_robust = precision_score(y_test, y_pred_dt_robust, average='binary')
recall_dt_robust = recall_score(y_test, y_pred_dt_robust, average='binary')
f1_dt_robust = f1_score(y_test, y_pred_dt_robust, average='binary')

print("\nDecision Tree with Robust Scaling:")
print(f"Accuracy: {accuracy_dt_robust:.2f}")
print(f"Precision: {precision_dt_robust:.2f}")
print(f"Recall: {recall_dt_robust:.2f}")
print(f"F1 Score: {f1_dt_robust:.2f}")

# Calculate and print metrics for Decision Tree with Standard Scaling
accuracy_dt_scaled = accuracy_score(y_test, y_pred_dt_scaled)
precision_dt_scaled = precision_score(y_test, y_pred_dt_scaled, average='binary')
recall_dt_scaled = recall_score(y_test, y_pred_dt_scaled, average='binary')
f1_dt_scaled = f1_score(y_test, y_pred_dt_scaled, average='binary')

print("\nDecision Tree with Standard Scaling:")
print(f"Accuracy: {accuracy_dt_scaled:.2f}")
print(f"Precision: {precision_dt_scaled:.2f}")
print(f"Recall: {recall_dt_scaled:.2f}")
print(f"F1 Score: {f1_dt_scaled:.2f}")
