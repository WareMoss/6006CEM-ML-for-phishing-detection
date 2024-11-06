
import os
import pandas as pd
def main():
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=0, low_memory=False)
    if not os.path.exists(r'D:\Uni Work\6006CEM - Machine Learning and Related Applications\assessment\PhiUSIIL_Phishing_URL_DatasetU.csv'):
       # check if this file exists, as it is the processed file 
       # if the file doesnt exist process the original dataset
       # if the file exists, skip the processing step saving time 
        top_20_features = [
        'URLSimilarityIndex', 'NoOfExternalRef', 'NoOfImage', 'LineOfCode', 'NoOfSelfRef',
        'HasSocialNet', 'NoOfCSS', 'NoOfJS', 'HasCopyrightInfo',
        'NoOfOtherSpecialCharsInURL', 'IsHTTPS', 'URLLength', 'HasDescription', 'NoOfDegitsInURL',
        'NoOfLettersInURL', 'HasSubmitButton', 'URLTitleMatchScore', 'HasTitle', 'SpacialCharRatioInURL', 'TLD', 'label'
        ]
        # from previous runs I have determined that these are the top 20 most important features in the dataset, as well as label
        dfclean = df[top_20_features]
        if dfclean.isna().sum().sum() > 0:
            print("NaN values detected in the dataset.")
            dfclean = dfclean.dropna()
            # drops nan rows as the dataset is 230k rows so a few rows wont be missed
            print("NaN count per column:")
            print(dfclean.isna().sum())
        # store the most important features as a new dataset, trimming down unecessary columns
        dfclean.to_csv('PhiUSIIL_Phishing_URL_DatasetU.csv', index=False)
        # creates new CSV file for the new dataset columns
if __name__ == "__main__":
    main()
