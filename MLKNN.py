from preprocessing import X_train, y_train, X_test, y_test
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

def main():

    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)
    y_train_1d = pd.Series(y_train.squeeze())
    y_test_1d = pd.Series(y_test.squeeze())
    # Convert X_train and X_test into DataFrames, and y_train and y_test into Series
    # This ensures that we can work with the data using pandas for easier manipulation and scaling

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)
    # Initialize StandardScaler for feature scaling
    # Scaling the features ensures that all variables have the same scale, preventing any one feature from dominating

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train_1d) 
    # Fit the model using the scaled training data
    # Initialize the K-Nearest Neighbors model with 5 neighbors (k=5)
    # KNN is sensitive to the choice of k, so this is a basic starting point

    train_score = model.score(X_train_scaled, y_train_1d)
    test_score = model.score(X_test_scaled, y_test_1d)
    y_pred = model.predict(X_test_scaled)
    accuracy_KNN = accuracy_score(y_test_1d, y_pred)
    # Evaluate the model on both training and testing datasets
    # Using the score() method to calculate accuracy for both training and test data

    print(f"Training score: {train_score:.3f}")
    print(f"Test score: {test_score:.3f}")
    print(f"Accuracy of KNN: {accuracy_KNN:.3f}")
    # Print the training and test scores, as well as the accuracy of the KNN model
    
    k_values = range(1, 21)
    train_accuracies = []
    test_accuracies = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, y_train_1d)
        train_accuracies.append(model.score(X_train_scaled, y_train_1d))
        test_accuracies.append(model.score(X_test_scaled, y_test_1d))
    # Hyperparameter tuning: Evaluating the performance of different k values
    # Loop over a range of k values (1 to 20) and store the accuracies for both train and test datasets

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, train_accuracies, label="Train Accuracy", marker='o')  
    plt.plot(k_values, test_accuracies, label="Test Accuracy", marker='o')  
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Neighbors (k) for KNN')
    plt.legend()  
    # Add legend to differentiate between train and test accuracy
    plt.show() 
    # Plot the accuracy for different k values to visually find the optimal k

    return "K-Nearest Neighbors", y_test_1d, y_pred, accuracy_KNN

model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN = main()
