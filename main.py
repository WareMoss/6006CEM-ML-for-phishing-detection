from eval import main as eval
def main():
    print("Starting data cleaning:")
    from cleaningcsv import main as cleaningcsv
    cleaningcsv()  # Calls the main function in cleaningcsv.py
    print("Data cleaning Completed.")
    
    print("Starting data preprocessing:")
    from preprocessing import main as preprocess
    preprocess()  # Calls the main function in preprocessing.py
    print("Data preprocessing completed.")
    
    print("Running Random Forest Model with hyperparameter tuning and gridsearch cv:")
    from MLRF import model_name as model_name_rf, y_test_rf, y_pred_rf, accuracy_rf, main as mlrf
    mlrf()  # Calls the main function in MLRF.py
    eval(model_name_rf, y_test_rf, y_pred_rf, accuracy_rf)
    print("Random Forest Model Completed.")

    print("Running Random Forest Model with Kfold cross validation:")
    from CrossVal import main as cv_main
    cv_main()  # Calls the main function in MLRF.py
    print("Random Forest Model Cross Evaluation Completed.")

    print("Running Logistic Regression Model:")
    from MLLR import model_name_LR, y_test_LR, y_pred_LR, accuracy_LR, main as mllr
    mllr()  # Calls the main function in MLLR.py
    eval(model_name_LR, y_test_LR, y_pred_LR, accuracy_LR)
    print("Logistic Regression Model Completed.")

    print("Running K-Nearest Neighbors Model:")
    from MLKNN import model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN, main as mlknn
    mlknn()  # Calls the main function in MLKNN.py
    eval(model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN)
    print("K-Nearest Neighbors Model Completed.")
if __name__ == "__main__":
    main()
