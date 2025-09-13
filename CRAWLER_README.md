# Stepan.com Web Crawler

A multithreaded web crawler designed to systematically visit all pages of the Stepan.com website, identify PDF files, and download them for further processing.

## Features

- **Multithreaded Crawling**: Uses ThreadPoolExecutor for concurrent page processing
- **PDF Detection & Download**: Automatically finds and downloads PDF files
- **S3 Integration**: Automatically uploads PDFs to AWS S3 bucket
- **Respectful Crawling**: Honors robots.txt and implements rate limiting
- **Comprehensive Logging**: Detailed logs of all operations
- **Error Handling**: Robust error handling with retry logic
- **Progress Tracking**: Real-time progress monitoring
- **Report Generation**: JSON and CSV reports of crawl results

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements_crawler.txt
   ```

2. **Configure AWS Credentials** (for S3 upload):
   ```bash
   # Option 1: AWS CLI
   aws configure
   
   # Option 2: Environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export AWS_DEFAULT_REGION=us-east-1
   
   # Option 3: IAM role (if running on EC2)
   # No additional configuration needed
   ```

3. **Verify Installation**:
   ```bash
   python stepan_crawler.py --help
   ```

## Usage

### Basic Usage

```bash
# Basic usage (SSL verification disabled for development)
python stepan_crawler.py

# With SSL verification enabled (for production)
python stepan_crawler.py --verify-ssl

# With custom settings
python stepan_crawler.py --workers 10 --delay 0.5 --max-depth 15 --output-dir my_downloads

# With S3 upload (default bucket: hackaton-stepan-data)
python stepan_crawler.py --workers 5 --delay 1.0

# With custom S3 bucket
python stepan_crawler.py --s3-bucket my-custom-bucket --s3-region us-west-2

# Disable S3 upload (local files only)
python stepan_crawler.py --no-s3
```

### Command Line Options

- `--verify-ssl`: Enable SSL certificate verification (default: disabled)
- `--workers N`: Number of worker threads (default: 5)
- `--delay N`: Delay between requests in seconds (default: 1.0)
- `--max-depth N`: Maximum crawl depth (default: 10)
- `--output-dir DIR`: Output directory for downloads (default: stepan_pdfs)
- `--s3-bucket BUCKET`: S3 bucket name for uploading PDFs (default: hackaton-stepan-data)
- `--s3-region REGION`: AWS region for S3 bucket (default: us-east-1)
- `--no-s3`: Disable S3 upload (files will only be saved locally)

### Configuration

The crawler can be configured by modifying the `crawler_config.json` file or by editing the configuration in the `main()` function.

### Key Configuration Options

- `max_workers`: Number of concurrent threads (default: 5)
- `delay`: Delay between requests in seconds (default: 1.0)
- `max_depth`: Maximum crawl depth (default: 10)
- `max_file_size`: Maximum PDF file size in bytes (default: 50MB)
- `download_dir`: Directory to save downloaded files (default: "stepan_pdfs")

## Output Structure

```
stepan_pdfs/
├── downloads/           # Downloaded PDF files
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── logs/               # Crawler logs
│   └── crawler_2024-01-15.log
└── reports/            # Generated reports
    ├── crawl_summary_2024-01-15_10-30-00.json
    └── pdf_inventory_2024-01-15_10-30-00.csv
```

## Reports

### JSON Report
Contains comprehensive crawl statistics and PDF inventory:
```json
{
  "crawl_summary": {
    "start_time": "2024-01-15T10:00:00Z",
    "end_time": "2024-01-15T11:30:00Z",
    "total_pages_visited": 150,
    "pdfs_found": 25,
    "pdfs_downloaded": 23,
    "errors": 2,
    "total_size_downloaded": "125.4MB"
  },
  "pdf_inventory": [...]
}
```

### CSV Report
Contains a simple inventory of all downloaded PDF files with metadata.

## Error Handling

The crawler handles various error conditions:

- **HTTP Errors**: 404, 403, 500, timeout errors
- **Network Issues**: Connection timeouts, DNS failures
- **File Errors**: Invalid PDFs, disk space issues
- **Rate Limiting**: Automatic backoff for rate-limited requests

## Logging

Logs are written to both console and file (`stepan_pdfs/logs/crawler_YYYY-MM-DD.log`).

Log levels:
- `INFO`: Normal operations
- `WARNING`: Non-fatal issues
- `ERROR`: Fatal errors

## Ethical Considerations

The crawler is designed to be respectful:

- **Robots.txt Compliance**: Checks and respects robots.txt
- **Rate Limiting**: Implements delays between requests
- **User-Agent**: Uses descriptive user-agent string
- **Resource Conservation**: Limits concurrent requests

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**: 
   - **Problem**: `[SSL: CERTIFICATE_VERIFY_FAILED]` errors on macOS
   - **Solution**: Use `--verify-ssl` flag or set `verify_ssl=False` in config
   - **Note**: SSL verification is disabled by default for development

2. **Permission Errors**: Ensure write permissions for download directory
3. **Network Timeouts**: Increase timeout values in configuration
4. **Memory Issues**: Reduce max_workers for large crawls
5. **PDF Validation Failures**: Check file integrity and format

### Debug Mode

Enable debug logging by modifying the log level in the configuration:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Performance Tips

1. **Adjust Thread Count**: More threads = faster crawling, but more resource usage
2. **Tune Delays**: Lower delays = faster crawling, but may trigger rate limiting
3. **Monitor Resources**: Watch CPU and memory usage during large crawls
4. **Disk Space**: Ensure sufficient space for downloaded files

## Legal Notice

This crawler is for educational and research purposes. Users are responsible for:

- Complying with website terms of service
- Respecting copyright and intellectual property rights
- Following applicable laws and regulations
- Using downloaded content appropriately

## Support

For issues or questions:

1. Check the logs for error messages
2. Verify network connectivity and permissions
3. Review configuration settings
4. Ensure all dependencies are installed correctly

## License

This project is provided as-is for educational purposes. Use responsibly and in accordance with applicable laws and website terms of service.
