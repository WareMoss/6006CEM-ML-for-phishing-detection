import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
def main():
    df = pd.read_csv('PhiUSIIL_Phishing_URL_DatasetU.csv', header=0, low_memory=False) # nrows=5000 to show it works
    # reads in the cleaned dataset as df using pandas
    X = df.drop('label', axis=1)
    # Training dataset only contains the values we want to find a link between, label is the answer to if a site was phishing or not
    # hiding this avoids overfitting where the model simply memorizes the dataset instead of looking for a pattern
    # or data leakage where the model knows the values
    y = df['label']
    # 1 = legit 0 = phishing
    # y contains the answers to compare the results of the model against
    if 'TLD_101' not in df.columns:
        # if the onehotencoded column isnt in the dataset, perform onehotencoding on the TLD column
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids multicollinearity
        # Apply the encoder to all values in the TLD column
        # this turns the catagorical data into an integer to be processed
        # so that the Ml algorithm can work
        TLD_encoded = encoder.fit_transform(X[['TLD']])
        # this selects the HTTTLDPS columnm from the dataset and turns it into its own dataframe
        # this is because onehotencoder expects a dataframe as an input
        # fit_transform() is a method that learns the unique catagories present in hte HTTPS column, these are represented as 1 and 0
        # then the data is transformed from catagorical into one-hot encoded format
        # this creates new binary columns (0 or 1) for each unique catagory
        TLD_df = pd.DataFrame(TLD_encoded, columns=encoder.get_feature_names_out(['TLD']))
        # this converts the now one hot encoded array TLD_encoded into a dataframe and names the new columns TLD_...
        X = X.drop('TLD', axis=1)
        # drop original row
        X = pd.concat([X.reset_index(drop=True), TLD_df.reset_index(drop=True)], axis=1)
        # combines the old dataset and the new dataset
        df_combined = pd.concat([X, y], axis=1)
        df_combined.to_csv('PhiUSIIL_Phishing_URL_DatasetU.csv', index=False)
        # saves the results into the cleaned csv file
    else:
        # if the data has already been encoded
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)
        # split the data into training 75% and testing 25%
        # X_train is the features the model is trying to find a relationship between to be trained on
        # X_test is 'unseen' so the model will use what its learnt in train to try and predict the label value for this set of data
        # y_train is the label feature so the model can learn what the expected output should be 
        # y_test is the features for the X_test dataset and the predictions can be compared to the actual values 
        return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = main()
if __name__ == "__main__":
    main()
