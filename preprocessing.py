from imports import *
def main():
    df = pd.read_csv('PhiUSIIL_Phishing_URL_DatasetU.csv', header=0, low_memory=False)
    # reads in the cleaned dataset
    X = df.drop('label', axis=1)
    # Training dataset only contains the values we want to find a link between, label is the answer to if a site was phishing or not
    # hiding this  avoids overfitting where the model simply memorizes the dataset instead of looking for a pattern
    y = df['label']
    # 1 = legit 0 = phishing
    # y contains the answers to compare the results of the model against
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids multicollinearity
    # Apply the encoder to the HTTPS column
    # this turns the catagorical data into an integer to be processed
    https_encoded = encoder.fit_transform(df[['HTTPS']])
    # this selects the HTTPS columnm from the dataset and turns it into its own dataframe
    # this is because onehotencoder expects a dataframe as an input
    # fit_transform() is a method that learns the unique catagories present in hte HTTPS column, these are represented as 1 and 0
    # then the data is transformed from catagorical into one-hot encoded format
    # this creates new binary columns (0 or 1) for each unique catagory
    https_df = pd.DataFrame(https_encoded, columns=encoder.get_feature_names_out(['HTTPS']))
    # print(https_df)
    # this converts the now one hot encoded array https_encoded into a dataframe and names the new column HTTPS
    # Drop the original HTTPS column from the DataFrame
    X = X.drop('HTTPS', axis=1)
    # print(X)
    # drops the old HTTPS column from the dataset
    X = pd.concat([X.reset_index(drop=True), https_df.reset_index(drop=True)], axis=1)
    # print(X['HTTPS_https'])
    # this print statement confirms that the column has correctly been added to the X data frame
    # reset_index(drop=True) this prevents the old index from being addes as a new column in the dataframe, ensures proper alignment 
    # # https_df.reset_index(drop=True) does the same
    # Combine the original DataFrame with the new one-hot encoded DataFrame
    # axis=1 means the concatination should be done column wise
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train is the features the model is trying to find a relationship between to be trained on
    # X_test is 'unseen' so the model will use what its learnt in train to try and predict the label value for this set of data
    # y_train is the label feature so the model can learn what the expected output should be 
    # y_test is the features for the X_test dataset and the predictions can be compared to the actual values 
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = main()