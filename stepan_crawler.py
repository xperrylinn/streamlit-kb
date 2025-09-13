#!/usr/bin/env python3
"""
Stepan.com Web Crawler
======================

A multithreaded web crawler designed to systematically visit all pages of the 
Stepan.com website, identify PDF files, and download them for further processing.

Features:
- Multithreaded crawling with configurable worker count
- PDF detection and download with validation
- Comprehensive error handling and logging
- Respectful crawling with rate limiting
- Progress tracking and reporting

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
import json
import csv
import logging
import threading
import queue
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from datetime import datetime
from typing import Set, Dict, List, Optional, Tuple
import hashlib

import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3

# AWS S3 integration
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("Warning: boto3 not available. S3 upload functionality will be disabled.")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class S3Handler:
    """Handles S3 upload operations for PDF files."""
    
    def __init__(self, bucket_name: str, region: str = None, prefix: str = "stepan-pdfs/"):
        """
        Initialize S3 handler.
        
        Args:
            bucket_name: Name of the S3 bucket
            region: AWS region
            prefix: S3 key prefix for uploaded files
        """
        self.bucket_name = bucket_name
        self.region = region or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)
        
        if not S3_AVAILABLE:
            self.logger.warning("S3 functionality disabled - boto3 not available")
            self.s3_client = None
            return
        
        try:
            # Initialize S3 client with credentials from environment
            self.s3_client = boto3.client(
                's3', 
                region_name=self.region,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                aws_session_token=os.getenv('AWS_SESSION_TOKEN')
            )
            
            # Test connection
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.info(f"✅ Connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            self.logger.error("❌ AWS credentials not found in .env file. Please check:")
            self.logger.error("   - AWS_ACCESS_KEY_ID is set in .env")
            self.logger.error("   - AWS_SECRET_ACCESS_KEY is set in .env")
            self.s3_client = None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                self.logger.error(f"❌ S3 bucket '{bucket_name}' not found")
            elif error_code == '403':
                self.logger.error(f"❌ Access denied to S3 bucket '{bucket_name}'")
                self.logger.error("   Please check:")
                self.logger.error("   - Bucket exists and is accessible")
                self.logger.error("   - Your AWS credentials have s3:GetObject and s3:PutObject permissions")
                self.logger.error("   - Bucket policy allows your AWS account access")
            else:
                self.logger.error(f"❌ S3 error: {e}")
            self.s3_client = None
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize S3 client: {e}")
            self.s3_client = None
    
    def upload_file(self, local_file_path: Path, s3_key: str = None) -> bool:
        """
        Upload a file to S3.
        
        Args:
            local_file_path: Path to the local file
            s3_key: S3 key (optional, will generate from filename if not provided)
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.s3_client:
            self.logger.warning("S3 client not available - skipping upload")
            return False
        
        try:
            # Generate S3 key if not provided
            if not s3_key:
                s3_key = f"{self.prefix}{local_file_path.name}"
            
            # Upload file
            self.s3_client.upload_file(
                str(local_file_path),
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'ContentType': 'application/pdf',
                    'Metadata': {
                        'source': 'stepan-crawler',
                        'uploaded_at': datetime.now().isoformat(),
                        'original_filename': local_file_path.name
                    }
                }
            )
            
            self.logger.info(f"✅ Uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"❌ S3 upload failed for {local_file_path.name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error uploading {local_file_path.name}: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if S3 functionality is available."""
        return self.s3_client is not None


class StepanWebCrawler:
    """
    Main crawler class for Stepan.com website.
    
    Handles URL discovery, PDF detection, and multithreaded crawling
    with comprehensive error handling and logging.
    """
    
    def __init__(self, base_url: str = "https://www.stepan.com/", 
                 max_workers: int = 5, delay: float = 1.0, 
                 max_depth: int = 10, download_dir: str = "stepan_pdfs",
                 verify_ssl: bool = False, respect_robots: bool = True,
                 s3_bucket: str = None, s3_region: str = "us-east-1"):
        """
        Initialize the web crawler.
        
        Args:
            base_url: Starting URL for crawling
            max_workers: Number of concurrent threads
            delay: Delay between requests in seconds
            max_depth: Maximum crawl depth
            download_dir: Directory to save downloaded PDFs
            verify_ssl: Whether to verify SSL certificates
            respect_robots: Whether to respect robots.txt (default: True)
            s3_bucket: S3 bucket name for uploading PDFs (optional)
            s3_region: AWS region for S3 bucket (default: us-east-1)
        """
        self.base_url = base_url
        self.domain = self._extract_domain(base_url)
        self.max_workers = max_workers
        self.delay = delay
        self.max_depth = max_depth
        self.download_dir = Path(download_dir)
        self.verify_ssl = verify_ssl
        self.respect_robots = respect_robots
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        
        # Thread-safe data structures
        self.visited_urls: Set[str] = set()
        self.url_queue: queue.Queue = queue.Queue()
        self.pdf_urls: Set[str] = set()
        self.lock = threading.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_pages_visited': 0,
            'pdfs_found': 0,
            'pdfs_downloaded': 0,
            'pdfs_uploaded_to_s3': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None,
            'total_size_downloaded': 0
        }
        
        # Setup directories and logging
        self._setup_directories()
        self._setup_logging()
        self._setup_ssl_context()
        self._setup_session()
        
        # Initialize PDF handler
        self.pdf_handler = PDFHandler(self.download_dir / "downloads", verify_ssl=self.verify_ssl)
        
        # Initialize S3 handler
        if self.s3_bucket:
            self.s3_handler = S3Handler(self.s3_bucket, self.s3_region)
            if self.s3_handler.is_available():
                self.logger.info(f"✅ S3 upload enabled for bucket: {self.s3_bucket}")
            else:
                self.logger.warning("⚠️ S3 upload disabled - check AWS credentials and bucket access")
        else:
            self.s3_handler = None
            self.logger.info("ℹ️ S3 upload disabled - no bucket specified")
        
        # Robots.txt handling
        if self.respect_robots:
            self.robots_parser = self._load_robots_txt()
        else:
            self.robots_parser = None
            self.logger.warning("Robots.txt checking disabled. Use with caution.")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for scope validation."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _setup_directories(self):
        """Create necessary directories for downloads, logs, and reports."""
        self.download_dir.mkdir(exist_ok=True)
        (self.download_dir / "downloads").mkdir(exist_ok=True)
        (self.download_dir / "logs").mkdir(exist_ok=True)
        (self.download_dir / "reports").mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Configure logging for the crawler."""
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_file = self.download_dir / "logs" / f"crawler_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s [%(threadName)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_ssl_context(self):
        """Setup SSL context for the application."""
        if not self.verify_ssl:
            import urllib3
            # Disable SSL warnings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            self.logger.warning("SSL verification disabled. Use with caution in production.")
        else:
            self.logger.info("SSL verification enabled.")
    
    def _setup_session(self):
        """Configure HTTP session with appropriate headers and settings."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configure SSL verification
        self.session.verify = self.verify_ssl
        
        # Configure retry strategy
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def _load_robots_txt(self) -> Optional[RobotFileParser]:
        """Load and parse robots.txt file."""
        try:
            robots_url = urljoin(self.base_url, "/robots.txt")
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            self.logger.info(f"Loaded robots.txt from {robots_url}")
            return rp
        except Exception as e:
            self.logger.warning(f"Could not load robots.txt: {e}")
            return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is within the allowed domain scope."""
        try:
            parsed = urlparse(url)
            return (parsed.netloc == urlparse(self.domain).netloc and 
                    parsed.scheme in ['http', 'https'])
        except Exception:
            return False
    
    def _normalize_url(self, url: str, base_url: str) -> str:
        """Convert relative URLs to absolute URLs."""
        try:
            # Handle relative URLs
            if url.startswith('//'):
                url = f"https:{url}"
            elif url.startswith('/'):
                url = urljoin(self.domain, url)
            elif not url.startswith(('http://', 'https://')):
                url = urljoin(base_url, url)
            
            # Remove fragments and normalize
            parsed = urlparse(url)
            normalized = urlunparse((
                parsed.scheme, parsed.netloc, parsed.path,
                parsed.params, parsed.query, None
            ))
            return normalized
        except Exception as e:
            self.logger.warning(f"Error normalizing URL {url}: {e}")
            return url
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        if not self.respect_robots or not self.robots_parser:
            return True
        try:
            return self.robots_parser.can_fetch('*', url)
        except Exception:
            return True
    
    def _crawl_page(self, url: str, depth: int = 0) -> List[str]:
        """
        Crawl a single page and extract links and PDFs.
        
        Args:
            url: URL to crawl
            depth: Current crawl depth
            
        Returns:
            List of new URLs found on the page
        """
        if depth > self.max_depth:
            return []
        
        # Check if we can fetch this URL
        if not self._can_fetch(url):
            self.logger.warning(f"Robots.txt disallows fetching: {url}")
            return []
        
        try:
            self.logger.info(f"Visiting: {url} (depth: {depth})")
            
            # Check if this URL is a PDF file
            if url.lower().endswith('.pdf'):
                self.logger.info(f"Found PDF: {url}")
                with self.lock:
                    if url not in self.pdf_urls:  # Only process new PDFs
                        self.pdf_urls.add(url)
                        self.stats['pdfs_found'] += 1
                        
                        # Download PDF immediately
                        try:
                            success, pdf_filename = self.pdf_handler.download_pdf(url)
                            if success:
                                self.stats['pdfs_downloaded'] += 1
                                self.logger.info(f"✅ Downloaded: {url}")
                                
                                # Upload to S3 if enabled
                                if self.s3_handler and self.s3_handler.is_available():
                                    try:
                                        # Use the actual filename from download
                                        local_file_path = self.download_dir / "downloads" / pdf_filename
                                        
                                        if local_file_path.exists():
                                            s3_success = self.s3_handler.upload_file(local_file_path)
                                            if s3_success:
                                                self.stats['pdfs_uploaded_to_s3'] += 1
                                                self.logger.info(f"☁️ Uploaded to S3: {pdf_filename}")
                                            else:
                                                self.logger.warning(f"⚠️ S3 upload failed: {pdf_filename}")
                                        else:
                                            self.logger.warning(f"⚠️ Downloaded file not found: {local_file_path}")
                                    except Exception as e:
                                        self.logger.error(f"❌ S3 upload error for {pdf_filename}: {e}")
                                        with self.lock:
                                            self.stats['errors'] += 1
                        except Exception as e:
                            self.logger.error(f"❌ Download error for {url}: {e}")
                            with self.lock:
                                self.stats['errors'] += 1
                
                return []  # No links to follow from PDF files
            
            # Make request with timeout and SSL verification setting
            response = self.session.get(url, timeout=30, verify=self.verify_ssl)
            response.raise_for_status()
            
            # Update statistics
            with self.lock:
                self.stats['total_pages_visited'] += 1
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract all links
            new_urls = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                normalized_url = self._normalize_url(href, url)
                
                if (self._is_valid_url(normalized_url) and 
                    normalized_url not in self.visited_urls):
                    new_urls.append(normalized_url)
            
            # Look for PDF links and download them immediately
            pdf_links = self._find_pdf_links(soup, url)
            for pdf_url in pdf_links:
                with self.lock:
                    if pdf_url not in self.pdf_urls:  # Only process new PDFs
                        self.pdf_urls.add(pdf_url)
                        self.stats['pdfs_found'] += 1
                        self.logger.info(f"Found PDF: {pdf_url}")
                        
                        # Download PDF immediately
                        try:
                            success, pdf_filename = self.pdf_handler.download_pdf(pdf_url)
                            if success:
                                self.stats['pdfs_downloaded'] += 1
                                self.logger.info(f"✅ Downloaded: {pdf_url}")
                                
                                # Upload to S3 if enabled
                                if self.s3_handler and self.s3_handler.is_available():
                                    try:
                                        # Use the actual filename from download
                                        local_file_path = self.download_dir / "downloads" / pdf_filename
                                        
                                        if local_file_path.exists():
                                            s3_success = self.s3_handler.upload_file(local_file_path)
                                            if s3_success:
                                                self.stats['pdfs_uploaded_to_s3'] += 1
                                                self.logger.info(f"☁️ Uploaded to S3: {pdf_filename}")
                                            else:
                                                self.logger.warning(f"⚠️ S3 upload failed: {pdf_filename}")
                                        else:
                                            self.logger.warning(f"⚠️ Downloaded file not found: {local_file_path}")
                                    except Exception as s3_error:
                                        self.logger.error(f"❌ S3 upload error for {pdf_url}: {s3_error}")
                                        self.stats['errors'] += 1
                            else:
                                self.logger.warning(f"❌ Failed to download: {pdf_url}")
                        except Exception as e:
                            self.logger.error(f"Error downloading {pdf_url}: {e}")
                            self.stats['errors'] += 1
            
            # Rate limiting
            time.sleep(self.delay)
            
            return new_urls
            
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout for {url}")
            with self.lock:
                self.stats['errors'] += 1
        except requests.exceptions.HTTPError as e:
            self.logger.warning(f"HTTP error {e.response.status_code} for {url}")
            with self.lock:
                self.stats['errors'] += 1
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed for {url}: {e}")
            with self.lock:
                self.stats['errors'] += 1
        except Exception as e:
            self.logger.error(f"Unexpected error crawling {url}: {e}")
            with self.lock:
                self.stats['errors'] += 1
        
        return []
    
    def _find_pdf_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find all PDF links on the current page."""
        pdf_links = []
        
        # Look for direct PDF links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                pdf_url = self._normalize_url(href, base_url)
                if self._is_valid_url(pdf_url):
                    pdf_links.append(pdf_url)
        
        # Look for PDF links in iframes, embeds, etc.
        for tag in soup.find_all(['iframe', 'embed', 'object']):
            src = tag.get('src') or tag.get('data')
            if src and src.lower().endswith('.pdf'):
                pdf_url = self._normalize_url(src, base_url)
                if self._is_valid_url(pdf_url):
                    pdf_links.append(pdf_url)
        
        return pdf_links
    
    def _worker_thread(self):
        """Worker thread function for processing URLs from the queue."""
        while True:
            try:
                # Get URL and depth from queue
                url, depth = self.url_queue.get(timeout=1)
                
                # Mark URL as visited
                with self.lock:
                    if url in self.visited_urls:
                        self.url_queue.task_done()
                        continue
                    self.visited_urls.add(url)
                
                # Crawl the page
                new_urls = self._crawl_page(url, depth)
                
                # Add new URLs to queue
                for new_url in new_urls:
                    if new_url not in self.visited_urls:
                        self.url_queue.put((new_url, depth + 1))
                
                self.url_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Worker thread error: {e}")
                with self.lock:
                    self.stats['errors'] += 1
                self.url_queue.task_done()
    
    def start_crawling(self):
        """Start the multithreaded crawling process."""
        self.logger.info("Starting Stepan.com web crawler...")
        self.stats['start_time'] = datetime.now()
        
        # Add starting URL to queue
        self.url_queue.put((self.base_url, 0))
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit worker threads
            futures = [executor.submit(self._worker_thread) 
                      for _ in range(self.max_workers)]
            
            # Monitor progress
            self._monitor_progress()
            
            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Worker thread failed: {e}")
        
        # Generate final report
        self._generate_report()
        
        self.stats['end_time'] = datetime.now()
        self.logger.info("Crawling completed!")
    
    def _monitor_progress(self):
        """Monitor crawling progress and display statistics."""
        while not self.url_queue.empty():
            time.sleep(5)  # Check every 5 seconds
            with self.lock:
                self.logger.info(
                    f"Progress: {self.stats['total_pages_visited']} pages visited, "
                    f"{self.stats['pdfs_found']} PDFs found, "
                    f"{self.stats['errors']} errors"
                )
    
    def _download_all_pdfs(self):
        """Download all discovered PDF files."""
        self.logger.info(f"Starting download of {len(self.pdf_urls)} PDF files...")
        
        for pdf_url in self.pdf_urls:
            try:
                success, filename = self.pdf_handler.download_pdf(pdf_url)
                if success:
                    with self.lock:
                        self.stats['pdfs_downloaded'] += 1
            except Exception as e:
                self.logger.error(f"Error downloading {pdf_url}: {e}")
                with self.lock:
                    self.stats['errors'] += 1
    
    def _generate_report(self):
        """Generate final crawl report and statistics."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Generate JSON report
        report_data = {
            "crawl_summary": {
                "start_time": self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                "end_time": self.stats['end_time'].isoformat() if self.stats['end_time'] else None,
                "total_pages_visited": self.stats['total_pages_visited'],
                "pdfs_found": self.stats['pdfs_found'],
                "pdfs_downloaded": self.stats['pdfs_downloaded'],
                "pdfs_uploaded_to_s3": self.stats['pdfs_uploaded_to_s3'],
                "errors": self.stats['errors'],
                "total_size_downloaded": f"{self.stats['total_size_downloaded'] / (1024*1024):.2f}MB",
                "s3_bucket": self.s3_bucket if self.s3_bucket else "Not configured"
            },
            "pdf_inventory": []
        }
        
        # Add PDF inventory
        for pdf_file in (self.download_dir / "downloads").glob("*.pdf"):
            try:
                file_size = pdf_file.stat().st_size
                report_data["pdf_inventory"].append({
                    "filename": pdf_file.name,
                    "size": f"{file_size / (1024*1024):.2f}MB",
                    "download_time": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Error getting file info for {pdf_file}: {e}")
        
        # Save JSON report
        json_file = self.download_dir / "reports" / f"crawl_summary_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save CSV inventory
        csv_file = self.download_dir / "reports" / f"pdf_inventory_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'size', 'download_time'])
            writer.writeheader()
            writer.writerows(report_data["pdf_inventory"])
        
        self.logger.info(f"Reports saved to {self.download_dir / 'reports'}")


class PDFHandler:
    """Handles PDF detection, downloading, and validation."""
    
    def __init__(self, download_dir: Path, max_size: int = 50 * 1024 * 1024, verify_ssl: bool = True):
        """
        Initialize PDF handler.
        
        Args:
            download_dir: Directory to save PDFs
            max_size: Maximum file size in bytes (50MB default)
            verify_ssl: Whether to verify SSL certificates
        """
        self.download_dir = download_dir
        self.max_size = max_size
        self.verify_ssl = verify_ssl
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith('.pdf')
    
    def download_pdf(self, url: str) -> tuple[bool, str]:
        """
        Download a PDF file with validation.
        
        Args:
            url: URL of the PDF to download
            
        Returns:
            Tuple of (success: bool, filename: str)
        """
        try:
            self.logger.info(f"Downloading PDF: {url}")
            
            # Make HEAD request to check file size and type
            head_response = requests.head(url, timeout=30, allow_redirects=True, verify=self.verify_ssl)
            head_response.raise_for_status()
            
            # Check content type
            content_type = head_response.headers.get('content-type', '').lower()
            if 'application/pdf' not in content_type:
                self.logger.warning(f"Not a PDF file: {url} (content-type: {content_type})")
                return False, ""
            
            # Check file size
            content_length = head_response.headers.get('content-length')
            if content_length and int(content_length) > self.max_size:
                self.logger.warning(f"File too large: {url} ({int(content_length)} bytes)")
                return False, ""
            
            # Generate filename
            filename = self._generate_filename(url)
            file_path = self.download_dir / filename
            
            # Download the file
            response = requests.get(url, timeout=60, stream=True, verify=self.verify_ssl)
            response.raise_for_status()
            
            # Save file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Validate downloaded PDF
            if self._validate_pdf(file_path):
                file_size = file_path.stat().st_size
                self.logger.info(f"Successfully downloaded: {filename} ({file_size} bytes)")
                return True, filename
            else:
                self.logger.warning(f"Downloaded file is not a valid PDF: {filename}")
                file_path.unlink()  # Remove invalid file
                return False, ""
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False, ""
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {url}: {e}")
            return False, ""
    
    def _generate_filename(self, url: str) -> str:
        """Generate a safe filename for the PDF."""
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename, generate one
        if not filename or not filename.endswith('.pdf'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"document_{url_hash}.pdf"
        
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        # Handle duplicates
        counter = 1
        original_filename = filename
        while (self.download_dir / filename).exists():
            name, ext = os.path.splitext(original_filename)
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        return filename
    
    def _get_downloaded_filename(self, url: str) -> str:
        """Get the actual filename that was used for download."""
        # Extract filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename, generate one
        if not filename or not filename.endswith('.pdf'):
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"document_{url_hash}.pdf"
        
        # Sanitize filename
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        
        # Check if file exists with this name, if not, find the actual filename
        if not (self.download_dir / filename).exists():
            # Look for files that start with the same name
            base_name = os.path.splitext(filename)[0]
            for file_path in self.download_dir.glob(f"{base_name}*.pdf"):
                if file_path.is_file():
                    return file_path.name
        
        return filename
    
    def _validate_pdf(self, file_path: Path) -> bool:
        """Validate that the downloaded file is a valid PDF."""
        try:
            # Check file size
            if file_path.stat().st_size == 0:
                return False
            
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    return False
            
            # Try to read the file to ensure it's not corrupted
            with open(file_path, 'rb') as f:
                content = f.read(1024)  # Read first 1KB
                if b'PDF' not in content:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating PDF {file_path}: {e}")
            return False


def main():
    """Main function to run the crawler."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stepan.com Web Crawler')
    parser.add_argument('--verify-ssl', action='store_true', 
                       help='Enable SSL certificate verification (default: disabled)')
    parser.add_argument('--workers', type=int, default=5,
                       help='Number of worker threads (default: 5)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Maximum crawl depth (default: 10)')
    parser.add_argument('--output-dir', default='stepan_pdfs',
                       help='Output directory for downloads (default: stepan_pdfs)')
    parser.add_argument('--ignore-robots', action='store_true',
                       help='Ignore robots.txt restrictions (use with caution)')
    parser.add_argument('--s3-bucket', default='hackaton-stepan-data',
                       help='S3 bucket name for uploading PDFs (default: hackaton-stepan-data)')
    parser.add_argument('--s3-region', default='us-east-1',
                       help='AWS region for S3 bucket (default: us-east-1)')
    parser.add_argument('--no-s3', action='store_true',
                       help='Disable S3 upload (files will only be saved locally)')
    
    args = parser.parse_args()
    
    print("Stepan.com Web Crawler")
    print("=====================")
    print()
    
    # Configuration
    config = {
        'base_url': 'https://www.stepan.com/',
        'max_workers': args.workers,
        'delay': args.delay,
        'max_depth': args.max_depth,
        'download_dir': args.output_dir,
        'verify_ssl': args.verify_ssl,
        'respect_robots': not args.ignore_robots,
        's3_bucket': None if args.no_s3 else args.s3_bucket,
        's3_region': args.s3_region
    }
    
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create and run crawler
    crawler = StepanWebCrawler(**config)
    
    try:
        crawler.start_crawling()
        
        # Print final statistics
        print("\nCrawl Summary:")
        print(f"  Pages visited: {crawler.stats['total_pages_visited']}")
        print(f"  PDFs found: {crawler.stats['pdfs_found']}")
        print(f"  PDFs downloaded: {crawler.stats['pdfs_downloaded']}")
        if crawler.s3_bucket:
            print(f"  PDFs uploaded to S3: {crawler.stats['pdfs_uploaded_to_s3']}")
            print(f"  S3 bucket: {crawler.s3_bucket}")
        print(f"  Errors: {crawler.stats['errors']}")
        print(f"  Download directory: {crawler.download_dir}")
        
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user.")
    except Exception as e:
        print(f"\nCrawling failed: {e}")
        logging.error(f"Crawling failed: {e}")


if __name__ == "__main__":
    main()
