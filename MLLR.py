from preprocessing import X_train, y_train, X_test, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

def main():
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # scales the data
    # logistic regression uses grqadient based optemisation to find the best fit model 
    # having the features on the same scale prevents dominance of certain features
    # fit_transform: calculates scaling parameters based on X_train data 
    # dont use fit on test as transform on its own just applies the training parameters to the test data
    # without modifying the values, ensuring fair model evaluation and preventing data leakage 

    model = LogisticRegression(max_iter=10000, class_weight='balanced',random_state=42)
    model.fit(X_train_scaled, y_train)
    # Train the model with scaled data
    # max_iter=10000: the max number of iterations the optemisation algorithm will run 
    # the algorithm changes its parameters to minimize the loss function and improve predicitons 

    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    # produce testing and training score for the model
    print("Training score: {:.3f}".format(train_score))
    print("Test score: {:.3f}".format(test_score))
    y_pred = model.predict(X_test_scaled)

    model_name_LR = "Logistic Regression"
    accuracy_LR = accuracy_score(y_test, y_pred)
    # produce accuracy score

    return model_name_LR, y_test, y_pred, accuracy_LR

model_name_LR, y_test_LR, y_pred_LR, accuracy_LR = main()

if __name__ == "__main__":
    main()
# https://youtu.be/U1omz0B9FTw?si=yHB2A5_SstA-CU3S
# video used to explain LR machine learning model
