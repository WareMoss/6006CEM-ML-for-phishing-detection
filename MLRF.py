from preprocessing import X_train, y_train, X_test, y_test
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
def main():
    # Define the hyperparameter grid for tuning
    # hyperparameters are settings which influence a models performance and are set before the training process
    # for the random forest model, they influence the number of trees made
    # gridsearchcv systematically tests a range of hyper parameter combinations to find which model performs best 
    # here the parameter grid is definced, this contains different values to try
    # it uses k fold cross validation where the data is split into different sets of training and testing values multiple times
    '''hyperparametergrid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, 15],  # Limiting tree depth
        'min_samples_split': [2, 5, 10],  # Increasing minimum samples required to split
        'min_samples_leaf': [1, 2, 4],  # Increasing minimum samples in leaf nodes
        'max_features': ['sqrt', 'log2'],  # Limiting the features considered at each split
    }'''
    # best hyperperams when ran with low ammount of data
    # Best Hyperparameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
    # best when ran with more:
    # Best Hyperparameters: {'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    
    hyperparametergrid = {
        'n_estimators': [ 100],
        'max_depth': [5],
        'min_samples_split': [2],
        'min_samples_leaf': [1], 
        'max_features': ['sqrt'],
    }
    # these hyper parameters are now tuned to what the results from previous tests were  
    # Limiting the features considered at each split
    # n_estimators represents the number of decision trees in the forest, higher number of trees improves performance but increases computational cost 
    # max_depth is how far each tree is able to go within the forest
    # deeper trees can capture more complex data but risks overfitting, value of none means each tree will grow until only one class in each leaf node 
    # min_samples_split is the min number of samples required to split each node, higher values prevent tress from growing too complex and help control overfitting 
    # by forcing nodes to have more samples before they split 
    # min_samples_leaf is the minimum  number of samples required to be in a leaf node, setting higher values can reduce overfitting by keeping the leaves broader 
    # and requiring each leaf to contain a min amount of data 
    # max_features is the number of features to consdier when looking for the best split at each node
    # sqrt means the model will randomly select a subset of features equal to the square root of the total number of features in the dataset

    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Set up GridSearchCV with Stratified K-Fold cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=hyperparametergrid, 
                               scoring='accuracy', cv=3, n_jobs=4, verbose=2)
    
    # Estimator: refers to the model to be optemised, in this case its the random forest model
    # hyperparamgrid: the dictionary of hyperperameters and their possibilities
    # scoring=accuracy: the model is evaluated based on the accuracy of each hyperperameter combination
    # cv=3: this means the number of cross validation folds to use
    # 3 means teh data will be split into 3 folds, 2 for training and 1 for validation 
    # cross validations goal is to improve the models robustness by evaluating it on each subset of data, helping to prevent overfitting
    # n_jobs: specifies the number of cpu cores to use for computation, setting to -1 would use all available cores 
    # verbose: amount of information printed, 1 is minimal progress indicators and status updates 
    
    grid_search.fit(X_train, y_train)
    # Fit Grid Search to the training data

    tunedparams = grid_search.best_params_
    bestmodel = grid_search.best_estimator_
    # Get the best parameters and model from grid search

    print("Best Hyperparameters:", tunedparams)
    # Print the best parameters

    y_pred = bestmodel.predict(X_test)
    # Make predictions using the best model found

    accuracy_rf = accuracy_score(y_test, y_pred)
    # Calculate accuracy
    model_name = "Random Forest (Tuned)"

    y_test_rf = y_test
    y_pred_rf = y_pred
    # Retain the true labels and predictions

    feature_importances = bestmodel.feature_importances_
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(len(feature_importances))]
    indices = np.argsort(feature_importances)[::-1]
    # Get feature importances from the best model

    top_n = 20
    top_indices = indices[:top_n]
    top_feature_names = np.array(feature_names)[top_indices]
    top_feature_importances = feature_importances[top_indices]
    # this is to display just the top 20 most important columns of the dataset

    print("Top 20 Feature ranking:")
    for f in range(len(top_feature_importances)):
        print(f"{f + 1}. {top_feature_names[f]} ({top_feature_importances[f]})")
    # Plotting the top 20 feature importances
    
    plt.figure(figsize=(12, 6))
    plt.title(f"Top {top_n} Feature Importances for {model_name}")
    plt.bar(range(len(top_feature_names)), top_feature_importances, align="center")
    plt.xticks(range(len(top_feature_names)), top_feature_names, rotation=90)
    plt.xlim([-1, len(top_feature_names)])
    plt.show()
    # graph to visually show how important each feature is 
    return y_test_rf, y_pred_rf, accuracy_rf, model_name

y_test_rf, y_pred_rf, accuracy_rf, model_name = main()

if __name__ == "__main__":
    main()
# https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
# dataset used
# https://youtu.be/v6VJ2RO66Ag?si=f3WzuX3WBG2UBvQU
# source for explaining the RF model
