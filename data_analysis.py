import pandas as pd

# Load data from CSV file
data = pd.read_csv('data.csv')

# Print the number of rows as the number of entries
print("Number of entries: ", data.shape[0])

# Print the number of columns as the number of features
print("Number of features: ", data.shape[1])

# Print the number of null values and their occurrence in each column
print("Null values:")
print(data.isnull().sum())

# Print the top 5 rows of data
print("Top 5 rows:")
print(data.head())

# Print the datatypes in the various columns
print("Datatypes:")
print(data.dtypes)

# Print the number of unique values and their occurrence in each column
print("Unique values:")
for column in data.columns:
    print(column, ": ", data[column].nunique())