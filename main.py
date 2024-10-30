# main.py
from imports import *
def main():
    print("Starting data cleaning...")
    cleaningcsv_main()  # Calls the main function in cleaningcsv.py
    print("Data cleaning completed.")
    
    print("Starting data preprocessing...")
    preprocess_main()  # Calls the main function in preprocessing.py
    print("Data preprocessing completed.")
    
    print("Running Random Forest Model...")
    mlrf_main()  # Calls the main function in MLRF.py
    eval_main(model_name_rf, y_test_rf, y_pred_rf, accuracy_rf)
    print("Random Forest Model completed.")

    print("Running Logistic Regression Model...")
    mllr_main()  # Calls the main function in MLLR.py
    eval_main(model_name_LR, y_test_LR, y_pred_LR, accuracy_LR)
    print("Logistic Regression Model completed.")

    print("Running K-Nearest Neighbors Model...")
    #MLKNN_main()  # Calls the main function in MLKNN.py
    #eval_main(model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN)
    print("K-Nearest Neighbors Model completed.")

if __name__ == "__main__":
    main()