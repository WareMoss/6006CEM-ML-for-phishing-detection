import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

# Load the dataset
df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=0, low_memory=False)
df = df.drop('FILENAME', axis=1)  # Removes unimportant column

# Define categorical columns for One-Hot Encoding
categorical_columns = ['URL', 'Domain', 'TLD', 'Title']

# Initialize OneHotEncoder with sparse output
encoder = OneHotEncoder(sparse_output=True, drop='first', handle_unknown='ignore')

# Apply the encoder to the categorical columns
X_sparse = encoder.fit_transform(df[categorical_columns])

# Separate the numerical columns (excluding the target 'label')
numerical_columns = df.drop(columns=categorical_columns + ['label']).values

# Combine the sparse matrix (categorical features) with numerical columns
X_combined = hstack([X_sparse, numerical_columns])

# Define target variable
y = df['label']

# Split the dataset into training and testing sets (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Additional evaluation metrics (optional)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
