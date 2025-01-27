from preprocessing import X_train, y_train, X_test, y_test
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    rf_model = RandomForestClassifier(random_state=42)
    # Initialize the Random Forest model
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Define Stratified K-Folds cross-validation with 5 folds
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=stratified_kfold, scoring='accuracy', n_jobs=4, verbose=2)
    # Use cross_val_score to evaluate the model with cross-validation
    # This will run the model 5 times, each time with a different train-test split
    
    print(f"Cross-validation scores for each fold: {cv_scores}")
    # Print the individual fold scores
    
    print(f"Mean cross-validation score: {np.mean(cv_scores):.3f}")
    print(f"Standard deviation of cross-validation score: {np.std(cv_scores):.3f}")
    # Calculate and print the mean and standard deviation of the cross-validation scores
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    # Fit the model using the full training data to evaluate the final model performance
    
    accuracy_rf = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy_rf:.3f}")
    # Calculate and print the accuracy score on the test set
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Define K-Fold cross-validation (5 splits)

    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='accuracy', n_jobs=4, verbose=2)
    # Use cross_val_score to evaluate the model with K-Fold

    print(f"Cross-validation scores for each fold: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores):.3f}")
    print(f"Standard deviation of cross-validation score: {np.std(cv_scores):.3f}")
    # Print the individual fold scores

accuracy_rf = main()

if __name__ == "__main__":
    main()
