from imports import *

def main():
    df = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=0, low_memory=False)
    df['HTTPS'] = df['URL'].apply(lambda x: 'https' if str(x).lower().startswith('https') else 'http')
    # apply is a pandas method that applys a function to each element in a series, used here to clean the data
    # lambda function takes x as an input, which is the URL, as apply iterates over the values in the dataframe 
    # str(x).lower() converts x to a string type and converts the string to lowercase, this allows the comparison to be case sensitive 
    # startswith() checks if the URL starts with HTTPS and if not returns false giving a http otherwise its true and gives a https
    # note: could have just changed it straight to 1 and 0 as its either https or http but to demonstrate my ability to use 
    # one hot encoding I did this
    categorical_columns = ['URL', 'Domain', 'TLD', 'Title', 'FILENAME']
    df = df.drop(categorical_columns, axis=1)
    # axis=1 means the whole column
    # these are all the columns that contain catagorical data, they could be transfered into numerical values using one hot encoding
    # I have done this but it took about 3000s to compile and run while giving the same answer
    # so remove them
    column_names = df.columns.tolist()
    print(df.head())
    print(column_names)
    df.to_csv('PhiUSIIL_Phishing_URL_DatasetU.csv', index=False)
    # Save the updated DataFrame back to the Excel file (or a new file)