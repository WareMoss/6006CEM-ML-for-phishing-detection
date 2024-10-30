from imports import *

def main():
    # Ensure X_train and X_test are DataFrames
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, pd.Series) else X_train
    X_test_df = pd.DataFrame(X_test) if isinstance(X_test, pd.Series) else X_test

    # Ensure y_train and y_test are 1D Series
    y_train_1d = y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    y_test_1d = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)

    # Initialize KNN classifier
    k = 5  # You can tune this hyperparameter
    model = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    model.fit(X_train_scaled, y_train_1d)

    # Evaluate the model
    train_score = model.score(X_train_scaled, y_train_1d)
    test_score = model.score(X_test_scaled, y_test_1d)

    #print("Training score: {:.3f}".format(train_score))
    #print("Test score: {:.3f}".format(test_score))

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Convert predictions and true labels to DataFrames for consistency
    y_test_KNN = pd.DataFrame(y_test_1d, columns=["True_Label"])
    y_pred_KNN = pd.DataFrame(y_pred, columns=["Predicted_Label"])

    # Calculate accuracy
    accuracy_KNN = accuracy_score(y_test_1d, y_pred)

    # Display results
    #print(f"Accuracy of KNN: {accuracy_KNN:.3f}")
    
    # You can also print the confusion matrix and classification report if needed
    #cm = confusion_matrix(y_test_1d, y_pred)
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #plt.title("Confusion Matrix for KNN")
    #plt.xlabel("Predicted")
    #plt.ylabel("True")
    #plt.show()

    # Return results for further processing or evaluation
    model_name_KNN = "K-Nearest Neighbors"
    return model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN

# Call the main function
model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN = main()
