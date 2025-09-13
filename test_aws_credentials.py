#!/usr/bin/env python3
"""
Test AWS credentials and S3 access
"""

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def test_aws_credentials():
    """Test AWS credentials and S3 access"""
    print("Testing AWS Credentials...")
    print("=========================")
    
    try:
        # Initialize S3 client with credentials from environment
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name='us-east-1'
        )
        
        print("✅ AWS credentials loaded successfully")
        
        # Test listing buckets
        print("\nTesting S3 access...")
        response = s3_client.list_buckets()
        buckets = [bucket['Name'] for bucket in response['Buckets']]
        
        print(f"✅ Found {len(buckets)} S3 buckets:")
        for bucket in buckets:
            print(f"  - {bucket}")
        
        # Check if our target bucket exists
        target_bucket = "hackaton-stepan-data"
        if target_bucket in buckets:
            print(f"\n✅ Target bucket '{target_bucket}' found!")
            
            # Test bucket access
            try:
                s3_client.head_bucket(Bucket=target_bucket)
                print(f"✅ Access to bucket '{target_bucket}' confirmed")
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '403':
                    print(f"❌ Access denied to bucket '{target_bucket}'")
                    print("   You may need additional permissions or the bucket policy may restrict access")
                else:
                    print(f"❌ Error accessing bucket: {e}")
                return False
        else:
            print(f"\n❌ Target bucket '{target_bucket}' not found")
            print("   Available buckets:")
            for bucket in buckets:
                print(f"     - {bucket}")
            return False
            
    except NoCredentialsError:
        print("❌ AWS credentials not found in .env file")
        print("   Please check your .env file contains:")
        print("   AWS_ACCESS_KEY_ID=your_access_key")
        print("   AWS_SECRET_ACCESS_KEY=your_secret_key")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_aws_credentials()
    if success:
        print("\n🎉 AWS credentials and S3 access are working correctly!")
    else:
        print("\n💥 AWS credentials or S3 access test failed!")
