#!/usr/bin/env python3
"""
Check S3 bucket contents
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

def check_s3_bucket():
    s3 = boto3.client('s3', 
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
        region_name='us-west-2'
    )

    response = s3.list_objects_v2(Bucket='hackaton-stepan-data', Prefix='stepan-pdfs/')
    if 'Contents' in response:
        print(f'Found {len(response["Contents"])} files in S3 bucket:')
        for obj in response['Contents'][:10]:  # Show first 10
            print(f'  - {obj["Key"]} ({obj["Size"]} bytes)')
    else:
        print('No files found in S3 bucket')

if __name__ == "__main__":
    check_s3_bucket()
