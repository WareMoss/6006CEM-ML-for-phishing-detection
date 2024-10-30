from imports import *

def main():
    # Ensure X_train and X_test are DataFrames
    X_train_df = pd.DataFrame(X_train) if isinstance(X_train, pd.Series) else X_train
    X_test_df = pd.DataFrame(X_test) if isinstance(X_test, pd.Series) else X_test

    # Ensure y_train and y_test are 1D Series
    y_train_1d = y_train.squeeze() if isinstance(y_train, pd.DataFrame) else y_train
    y_test_1d = y_test.squeeze() if isinstance(y_test, pd.DataFrame) else y_test

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_df), columns=X_train_df.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_df), columns=X_test_df.columns)

    # Train the model with scaled data
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_scaled, y_train_1d)  # Fit the model with 1D y_train

    train_score = model.score(X_train_scaled, y_train_1d)
    test_score = model.score(X_test_scaled, y_test_1d)

    #print("Training score: {:.3f}".format(train_score))
    #print("Test score: {:.3f}".format(test_score))

    # Predictions as a 1D Series
    y_pred = model.predict(X_test_scaled)

    model_name_LR = "Logistic Regression"
    accuracy_LR = accuracy_score(y_test_1d, y_pred)

    return model_name_LR, y_test_1d, y_pred, accuracy_LR

# Call the main function
model_name_LR, y_test_LR, y_pred_LR, accuracy_LR = main()
