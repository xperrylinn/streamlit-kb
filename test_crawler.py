#!/usr/bin/env python3
"""
Test script for Stepan.com Web Crawler
=====================================

This script tests the crawler with a limited scope to verify functionality.
"""

import sys
import os
from stepan_crawler import StepanWebCrawler

def test_crawler():
    """Test the crawler with limited scope."""
    print("Testing Stepan.com Web Crawler")
    print("=============================")
    print()
    
    # Test configuration - limited scope for testing
    config = {
        'base_url': 'https://www.stepan.com/',
        'max_workers': 2,  # Reduced for testing
        'delay': 2.0,      # Longer delay for testing
        'max_depth': 2,    # Limited depth for testing
        'download_dir': 'test_stepan_pdfs',
        'verify_ssl': False,  # Disabled for development
        'respect_robots': False,  # Ignore robots.txt for testing
        's3_bucket': 'hackaton-stepan-data',  # Enable S3 upload
        's3_region': 'us-east-1'
    }
    
    print("Test Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Create and run crawler
        crawler = StepanWebCrawler(**config)
        crawler.start_crawling()
        
        # Print test results
        print("\nTest Results:")
        print(f"  Pages visited: {crawler.stats['total_pages_visited']}")
        print(f"  PDFs found: {crawler.stats['pdfs_found']}")
        print(f"  PDFs downloaded: {crawler.stats['pdfs_downloaded']}")
        print(f"  PDFs uploaded to S3: {crawler.stats['pdfs_uploaded_to_s3']}")
        print(f"  Errors: {crawler.stats['errors']}")
        print(f"  Test directory: {crawler.download_dir}")
        if crawler.s3_bucket:
            print(f"  S3 bucket: {crawler.s3_bucket}")
        
        if crawler.stats['errors'] == 0:
            print("\n✅ Test completed successfully!")
        else:
            print(f"\n⚠️ Test completed with {crawler.stats['errors']} errors")
            
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_crawler()
    sys.exit(0 if success else 1)
