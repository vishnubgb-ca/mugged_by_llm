import os
import boto3
import pandas as pd
from botocore.exceptions import NoCredentialsError

def connect_to_s3_and_load_data():
    # Get the credentials from environment variables
    aws_access_key_id = os.getenv('aws_access_key_id')
    aws_secret_access_key = os.getenv('aws_secret_access_key')
    bucket = os.getenv('bucket')
    region = os.getenv('region')
    key = os.getenv('key')

    # Create a session using your credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region
    )

    # Create an S3 resource object using the session
    s3 = session.resource('s3')

    # Load the data from S3
    try:
        obj = s3.Bucket(bucket).Object(key).get()
        data = pd.read_csv(obj['Body'])

        # Print the top 5 rows of data
        print(data.head())

        # Print data extraction successful
        print('Data extraction successful')

        # Store the dataframe as data.csv file
        data.to_csv('data.csv', index=False)

    except NoCredentialsError:
        print('No AWS credentials found')

# Call the function
connect_to_s3_and_load_data()