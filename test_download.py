#!/usr/bin/env python3
"""
Test PDF download functionality
"""

from stepan_crawler import PDFHandler
from pathlib import Path

def test_pdf_download():
    """Test downloading a single PDF"""
    print("Testing PDF download functionality...")
    
    # Create PDF handler
    pdf_handler = PDFHandler(Path("test_downloads"), verify_ssl=False)
    
    # Test URL (a known PDF from Stepan.com)
    test_url = "https://www.stepan.com/content/dam/stepan-dot-com/images/Stepan%20CSDS%20US%20Canada_2021-07-29.pdf"
    
    print(f"Testing download of: {test_url}")
    
    try:
        success = pdf_handler.download_pdf(test_url)
        if success:
            print("✅ PDF download successful!")
            
            # Check if file exists
            pdf_files = list(Path("test_downloads").glob("*.pdf"))
            print(f"✅ Found {len(pdf_files)} PDF file(s) in test_downloads/")
            for pdf_file in pdf_files:
                print(f"  - {pdf_file.name} ({pdf_file.stat().st_size} bytes)")
        else:
            print("❌ PDF download failed!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_pdf_download()
