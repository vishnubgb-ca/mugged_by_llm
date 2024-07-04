import pandas as pd

# Load data from CSV file
df = pd.read_csv('data.csv')

# Print the number of rows as the number of entries
print("Number of entries: ", df.shape[0])

# Print the number of columns as the number of features
print("Number of features: ", df.shape[1])

# Print the number of null values and their occurrence in each column
print("Null values:")
print(df.isnull().sum())

# Print the top 5 rows of data
print("Top 5 rows of data:")
print(df.head())

# Print the datatypes in the various columns
print("Datatypes:")
print(df.dtypes)

# Print the number of unique values and their occurrence in each column
print("Unique values:")
for column in df.columns:
    print(column, ": ", df[column].nunique())