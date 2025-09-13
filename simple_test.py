#!/usr/bin/env python3
"""
Simple test to verify SSL and connection issues
"""

import requests
import ssl
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_connection():
    """Test basic connection to Stepan.com"""
    print("Testing connection to Stepan.com...")
    
    try:
        # Test with SSL verification disabled
        response = requests.get('https://www.stepan.com/', verify=False, timeout=10)
        print(f"✅ Success! Status code: {response.status_code}")
        print(f"✅ Content length: {len(response.content)} bytes")
        print(f"✅ Content type: {response.headers.get('content-type', 'unknown')}")
        
        # Check if there are any PDF links
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = soup.find_all('a', href=lambda x: x and x.lower().endswith('.pdf'))
        print(f"✅ Found {len(pdf_links)} PDF links")
        
        for i, link in enumerate(pdf_links[:5]):  # Show first 5
            print(f"  {i+1}. {link.get('href')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Connection test successful!")
    else:
        print("\n💥 Connection test failed!")
