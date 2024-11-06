from cleaningcsv import main as cleaningcsv_main
from preprocessing import main as preprocess_main
from MLRF import model_name as model_name_rf, y_test_rf, y_pred_rf, accuracy_rf, main as mlrf_main
from MLKNN import model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN, main as MLKNN_main
from MLLR import model_name_LR, y_test_LR, y_pred_LR, accuracy_LR, main as mllr_main
from eval import main as eval_main
from CrossVal import main as cv_main
def main():
    print("Starting data cleaning...")
    cleaningcsv_main()  # Calls the main function in cleaningcsv.py
    print("Data cleaning Completed.")
    
    print("Starting data preprocessing...")
    preprocess_main()  # Calls the main function in preprocessing.py
    print("Data preprocessing completed.")
    
    print("Running Random Forest Model...")
    mlrf_main()  # Calls the main function in MLRF.py
    eval_main(model_name_rf, y_test_rf, y_pred_rf, accuracy_rf)
    print("Random Forest Model Completed.")

    print("Running Cross Evaluation Of The Random Forest Model...")
    cv_main()  # Calls the main function in MLRF.py
    print("Random Forest Model Cross Evaluation Completed.")

    print("Running Logistic Regression Model...")
    mllr_main()  # Calls the main function in MLLR.py
    eval_main(model_name_LR, y_test_LR, y_pred_LR, accuracy_LR)
    print("Logistic Regression Model Completed.")

    print("Running K-Nearest Neighbors Model...")
    MLKNN_main()  # Calls the main function in MLKNN.py
    eval_main(model_name_KNN, y_test_KNN, y_pred_KNN, accuracy_KNN)
    print("K-Nearest Neighbors Model Completed.")

if __name__ == "__main__":
    main()
