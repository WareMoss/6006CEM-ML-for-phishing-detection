from imports import *

def main(model_name, y_test, y_pred, accuracy, feature_importances=None, feature_names=None):
    # evaluate a models performance 
    # model attributes are passed in from file 
    # goal is to reduce amount of code which is re used in multiple files
    print(f"Evaluating Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    # Print model name and accuracy and what model is being evaluated
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    # Confusion Matrix of the model
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)
    # Classification Report of the model
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Phishing (0)', 'Phishing (1)'], 
                yticklabels=['Not Phishing (0)', 'Phishing (1)'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()
    # Plot Confusion Matrix of the model being used


