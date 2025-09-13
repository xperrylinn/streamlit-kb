#!/usr/bin/env python3
"""
Debug script to test crawler S3 functionality
"""

from stepan_crawler import StepanWebCrawler
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s')

def test_crawler_s3():
    """Test crawler with S3 enabled"""
    print("Testing Crawler with S3...")
    print("=========================")
    
    # Create crawler with S3 enabled
    crawler = StepanWebCrawler(
        base_url='https://www.stepan.com/',
        max_workers=2,
        delay=1.0,
        max_depth=2,
        download_dir='debug_stepan_pdfs',
        verify_ssl=False,
        respect_robots=False,
        s3_bucket='hackaton-stepan-data',
        s3_region='us-west-2'  # Use the region from .env
    )
    
    print(f"S3 Handler available: {crawler.s3_handler.is_available() if crawler.s3_handler else False}")
    print(f"S3 Bucket: {crawler.s3_bucket}")
    print(f"S3 Region: {crawler.s3_region}")
    
    # Start crawling
    try:
        crawler.start_crawling()
        
        print("\nFinal Statistics:")
        print(f"  Pages visited: {crawler.stats['total_pages_visited']}")
        print(f"  PDFs found: {crawler.stats['pdfs_found']}")
        print(f"  PDFs downloaded: {crawler.stats['pdfs_downloaded']}")
        print(f"  PDFs uploaded to S3: {crawler.stats['pdfs_uploaded_to_s3']}")
        print(f"  Errors: {crawler.stats['errors']}")
        
    except Exception as e:
        print(f"Error during crawling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_crawler_s3()
