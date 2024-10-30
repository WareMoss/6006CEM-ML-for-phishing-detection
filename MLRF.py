from imports import *
from sklearn.model_selection import GridSearchCV

def main():
    # Define the hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100],  # Fewer trees
        'max_depth': [None, 10],    # Fewer max depth options
        'min_samples_split': [2, 5],  # Fewer options
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']  # Keeping just one option
    }

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier()

    # Set up Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                               scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    # n_estimators represents the number of decision trees in the forest, higher number of trees improves performance but increases computational cost 
    # max_depth is how far eacch tree is able to go within the 'forest' 
    # deeper trees can capture more complex data but risks overfitting, value of none means each tree will grow until only one class in each leaf node 
    # min_samples_split is the min number of samples required to split each node, higher values prevent tress from growing too complex and help control overfitting 
    # by forcing nodes to have more samples before they split 
    # min_samples_leaf is the minimum  number of samples required to be in a leaf node, setting higher values can reduce overfitting by keeping the leaves broader 
    # and reauiring each leaf to contain a min amount of data 
    # max_features is the number of features to consdier when looking for the best split at each node
    # auto means the model uses the square root of the number of features while sqrt is similar but more optemised for speed in some implementations

    # Fit Grid Search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and model from grid search
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Print the best parameters
    print("Best Hyperparameters:", best_params)

    # Make predictions using the best model
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy_rf = accuracy_score(y_test, y_pred)
    model_name = "Random Forest (Tuned)"

    # Retain the true labels and predictions
    y_test_rf = y_test
    y_pred_rf = y_pred

    # Get feature importances from the best model
    feature_importances = best_model.feature_importances_
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(len(feature_importances))]
    indices = np.argsort(feature_importances)[::-1]

    # Produce a ranking of all the features in the dataset based on importance
    print("Feature ranking:")
    for f in range(len(feature_names)):
        print(f"{f + 1}. {feature_names[indices[f]]} ({feature_importances[indices[f]]})")

    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Importances for {model_name}")
    plt.bar(range(len(feature_names)), feature_importances[indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.show()

    return y_test_rf, y_pred_rf, accuracy_rf, model_name

# Run the main function
y_test_rf, y_pred_rf, accuracy_rf, model_name = main()
