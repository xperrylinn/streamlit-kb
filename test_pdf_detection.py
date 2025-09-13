#!/usr/bin/env python3
"""
Test PDF detection and S3 upload
"""

from stepan_crawler import StepanWebCrawler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s')

def test_pdf_detection():
    """Test PDF detection and S3 upload"""
    print("Testing PDF Detection and S3 Upload...")
    print("=====================================")
    
    # Create crawler with S3 enabled
    crawler = StepanWebCrawler(
        base_url='https://www.stepan.com/',
        max_workers=1,
        delay=1.0,
        max_depth=1,
        download_dir='test_pdf_detection',
        verify_ssl=False,
        respect_robots=False,
        s3_bucket='hackaton-stepan-data',
        s3_region='us-west-2'
    )
    
    print(f"S3 Handler available: {crawler.s3_handler.is_available() if crawler.s3_handler else False}")
    
    # Test with a known PDF URL
    test_pdf_url = "https://www.stepan.com/content/dam/stepan-dot-com/webdam/website-product-documents/literature/food-nutrition-pharmaceutical/NEOBEEBrochure.pdf"
    
    print(f"\nTesting with PDF URL: {test_pdf_url}")
    
    # Manually call _crawl_page to test PDF detection
    try:
        result = crawler._crawl_page(test_pdf_url, depth=0)
        print(f"Result: {result}")
        
        print(f"\nStatistics after test:")
        print(f"  PDFs found: {crawler.stats['pdfs_found']}")
        print(f"  PDFs downloaded: {crawler.stats['pdfs_downloaded']}")
        print(f"  PDFs uploaded to S3: {crawler.stats['pdfs_uploaded_to_s3']}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_detection()
