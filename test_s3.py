#!/usr/bin/env python3
"""
Test S3 functionality for Stepan crawler
"""

from stepan_crawler import S3Handler
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_s3_connection():
    """Test S3 connection and upload functionality"""
    print("Testing S3 Connection...")
    print("=======================")
    
    # Test S3 handler
    s3_handler = S3Handler(
        bucket_name="hackaton-stepan-data",
        region="us-east-1",
        prefix="test-stepan-pdfs/"
    )
    
    if not s3_handler.is_available():
        print("❌ S3 connection failed!")
        print("Please check:")
        print("  1. AWS credentials are configured")
        print("  2. S3 bucket 'hackaton-stepan-data' exists")
        print("  3. You have write permissions to the bucket")
        return False
    
    print("✅ S3 connection successful!")
    
    # Test upload with a small file
    test_file = Path("test_downloads/Stepan20CSDS20US20Canada_2021-07-29.pdf")
    if test_file.exists():
        print(f"Testing upload of: {test_file.name}")
        success = s3_handler.upload_file(test_file)
        if success:
            print("✅ Test upload successful!")
            return True
        else:
            print("❌ Test upload failed!")
            return False
    else:
        print("⚠️ Test file not found. Run test_download.py first.")
        return False

if __name__ == "__main__":
    success = test_s3_connection()
    if success:
        print("\n🎉 S3 functionality is working correctly!")
    else:
        print("\n💥 S3 functionality test failed!")
