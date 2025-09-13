"""
GenAI Bedrock Knowledgebase Application
=====================================

This Streamlit application provides a Q&A interface for a local knowledgebase system that:
1. Automatically syncs PDF documents from S3 bucket hackaton-stepan-data/stepan-pdfs
2. Processes PDF documents and converts them to text files
3. Converts them into searchable vector embeddings using AWS Bedrock
4. Allows users to ask questions about the documents using natural language
5. Provides AI-powered answers with source citations

Key Components:
- S3 Integration: Automatically syncs PDFs from AWS S3 bucket
- Automatic PDF Processing: Extracts text from PDFs on app startup
- Vector Storage: Uses ChromaDB to store document embeddings locally
- AI Integration: Uses AWS Bedrock Nova Micro LLM for question answering
- Web Interface: Streamlit provides an easy-to-use Q&A interface

Prerequisites:
- AWS credentials configured with Bedrock and S3 access
- AWS CLI installed and configured
- Python packages: streamlit, langchain, PyPDF2, chromadb, boto3
- .env file with AWS configuration (optional)
- Access to S3 bucket: hackaton-stepan-data/stepan-pdfs

How to Use:
1. Ensure AWS credentials are configured
2. Run: streamlit run app.py
3. App automatically syncs PDFs from S3 and processes them
4. Wait for automatic processing to complete
5. Ask questions about the documents in your knowledgebase
6. Get AI-powered answers with source citations
"""

# --- IMPORTS ---
# Core Python libraries
import os
import shutil
import stat
import time
import tempfile
import atexit

# Streamlit for web interface
import streamlit as st

# Environment and PDF processing
from dotenv import load_dotenv
import PyPDF2

# LangChain components for AI and document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_aws import ChatBedrockConverse
from langchain.chains import RetrievalQA

# AWS S3 for file management
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# --- ENVIRONMENT SETUP ---
# Load environment variables from .env file (AWS credentials, etc.)
load_dotenv()

# --- CONFIGURATION CONSTANTS ---
# Directory where PDF documents are stored
DATA_DIR = "stepan_pdfs/downloads"
# Directory where processed text files are stored
PROCESSED_DIR = "data"

# S3 Configuration
S3_BUCKET_NAME = "hackaton-stepan-data"
S3_PREFIX = "stepan-pdfs/"
AWS_REGION_S3 = "us-west-2"  # S3 region

# Create a temporary directory for ChromaDB that gets cleaned up automatically
# This avoids permission issues and keeps the workspace clean
TEMP_DIR = tempfile.mkdtemp(prefix="knowledgebase_chromadb_")
CHROMA_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Register cleanup function to remove temp directory when app exits
# def cleanup_temp_dir():
#     """Clean up temporary directory when application exits"""
#     try:
#         if os.path.exists(TEMP_DIR):
#             shutil.rmtree(TEMP_DIR, ignore_errors=True)
#             print(f"✅ Cleaned up temporary directory: {TEMP_DIR}")
#     except Exception as e:
#         print(f"⚠️ Could not clean up temporary directory: {e}")

# atexit.register(cleanup_temp_dir)

# AWS Bedrock model IDs - these are the AI models we'll use
AWS_BEDROCK_EMBEDDING_MODEL_ID = (
    "amazon.titan-embed-text-v2:0"  # For converting text to vectors
)
AWS_BEDROCK_LLM_MODEL_ID = "us.amazon.nova-micro-v1:0"  # For answering questions
AWS_REGION = "us-west-2"  # AWS region where Bedrock is available

# --- AI COMPONENTS INITIALIZATION ---
# Initialize the embedding model (converts text to numerical vectors for similarity search)
embedding_model = BedrockEmbeddings(
    region_name="us-west-2", model_id=AWS_BEDROCK_EMBEDDING_MODEL_ID
)

# Initialize text splitter (breaks documents into smaller chunks for processing)
# chunk_size=1000: Each chunk is ~1000 characters (increased for better context)
# chunk_overlap=100: Overlapping chunks to maintain context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# --- PDF PROCESSING FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using PyPDF2
    Handles both encrypted and unencrypted PDFs

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Extracted text content, or None if extraction fails
    """
    text = ""

    try:
        # Open PDF file in binary read mode
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)

            # Check if PDF is encrypted
            if pdf_reader.is_encrypted:
                print(f"🔒 PDF is encrypted: {pdf_path}")
                # Try to decrypt with empty password (common for many PDFs)
                try:
                    pdf_reader.decrypt("")
                    print(f"✅ Successfully decrypted: {pdf_path}")
                except Exception as decrypt_error:
                    print(f"❌ Failed to decrypt {pdf_path}: {decrypt_error}")
                    return None

            # Loop through each page in the PDF
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"  # Add double newline between pages
                    else:
                        print(f"⚠️ No text found on page {page_num + 1} of {pdf_path}")
                except Exception as page_error:
                    print(
                        f"⚠️ Error extracting text from page {page_num + 1} of {pdf_path}: {page_error}"
                    )
                    continue

    except Exception as e:
        print(f"❌ Error extracting text from PDF {pdf_path}: {e}")
        return None

    return text.strip()


def convert_pdf_to_text(uploaded_file, filename):
    """
    Convert an uploaded PDF file to text and save it as a .txt file

    Args:
        uploaded_file: Streamlit uploaded file object
        filename (str): Original filename of the uploaded PDF

    Returns:
        tuple: (text_filename, text_length) or (None, 0) if conversion fails
    """
    try:
        # Step 1: Save the uploaded PDF temporarily
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Step 2: Extract text from the temporary PDF
        text = extract_text_from_pdf(temp_pdf_path)

        # Step 3: Clean up the temporary PDF file
        os.remove(temp_pdf_path)

        if text:
            # Step 4: Save extracted text as a .txt file
            text_filename = filename.replace(".pdf", ".txt")
            text_path = os.path.join(DATA_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            return text_filename, len(text)
        else:
            return None, 0

    except Exception as e:
        st.error(f"Error processing PDF {filename}: {e}")
        # Clean up temporary file if something went wrong
        temp_pdf_path = os.path.join(DATA_DIR, f"temp_{filename}")
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)
        return None, 0


# --- DATABASE MANAGEMENT FUNCTIONS ---
def sync_s3_bucket():
    """
    Sync PDF files from S3 bucket to local directory using AWS CLI sync

    Returns:
        tuple: (success, message) - sync status and message
    """
    try:
        # Ensure local directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # Use AWS CLI sync command for efficient syncing
        import subprocess

        s3_uri = f"s3://{S3_BUCKET_NAME}/{S3_PREFIX}"

        print(f"🔄 Syncing files from {s3_uri} to {DATA_DIR}")

        # Run AWS sync command
        result = subprocess.run(
            [
                "aws",
                "s3",
                "sync",
                s3_uri,
                DATA_DIR,
                "--region",
                AWS_REGION_S3,
                "--delete",  # Remove local files that don't exist in S3
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"✅ S3 sync completed successfully")
            print(f"📁 Output: {result.stdout}")
            return True, "S3 sync completed successfully"
        else:
            print(f"❌ S3 sync failed: {result.stderr}")
            return False, f"S3 sync failed: {result.stderr}"

    except subprocess.TimeoutExpired:
        print("❌ S3 sync timed out after 5 minutes")
        return False, "S3 sync timed out"
    except FileNotFoundError:
        print("❌ AWS CLI not found. Please install AWS CLI")
        return False, "AWS CLI not found. Please install AWS CLI"
    except Exception as e:
        print(f"❌ Error during S3 sync: {e}")
        return False, f"Error during S3 sync: {e}"


def check_s3_credentials():
    """
    Check if AWS credentials are properly configured

    Returns:
        bool: True if credentials are valid, False otherwise
    """
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION_S3)
        # Try to list objects to verify credentials
        s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, MaxKeys=1)
        print("✅ AWS credentials verified")
        return True
    except NoCredentialsError:
        print("❌ AWS credentials not found")
        return False
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchBucket":
            print(f"❌ S3 bucket '{S3_BUCKET_NAME}' not found")
        else:
            print(f"❌ AWS credentials error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking AWS credentials: {e}")
        return False


def create_new_chroma_dir():
    """
    Create a new ChromaDB directory in the temp space

    Returns:
        str: Path to the new ChromaDB directory
    """
    # Create a unique subdirectory for this ChromaDB instance
    timestamp = int(time.time())
    new_chroma_dir = os.path.join(TEMP_DIR, f"chroma_db_{timestamp}")
    os.makedirs(new_chroma_dir, exist_ok=True)
    return new_chroma_dir


def safe_remove_directory(directory):
    """
    Safely remove a directory with proper error handling
    With temp directories, this should be much simpler and more reliable

    Args:
        directory (str): Path to directory to remove

    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(directory):
        return True

    try:
        # Since we're using temp directories, removal should be straightforward
        shutil.rmtree(directory, ignore_errors=True)
        return not os.path.exists(directory)
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
        return False


def create_vectorstore_batch(docs, embedding_model, persist_directory, batch_size=50):
    """
    Create vectorstore in batches to handle large document collections

    Args:
        docs: List of document chunks
        embedding_model: The embedding model to use
        persist_directory: Directory to save the vectorstore
        batch_size: Number of documents to process in each batch

    Returns:
        Chroma: The created vectorstore
    """
    print(
        f"📁 Creating vectorstore with {len(docs)} documents in batches of {batch_size}"
    )

    # Process documents in batches
    vectorstore = None
    total_batches = (len(docs) + batch_size - 1) // batch_size

    for i in range(0, len(docs), batch_size):
        batch_num = (i // batch_size) + 1
        batch_docs = docs[i : i + batch_size]

        print(
            f"📊 Processing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)"
        )

        try:
            if vectorstore is None:
                # Create the first batch
                vectorstore = Chroma.from_documents(
                    batch_docs, embedding_model, persist_directory=persist_directory
                )
                print(f"✅ Created initial vectorstore with batch {batch_num}")
            else:
                # Add to existing vectorstore
                vectorstore.add_documents(batch_docs)
                print(f"✅ Added batch {batch_num} to vectorstore")

            # Add a small delay to prevent overwhelming the system
            time.sleep(0.1)

        except Exception as e:
            print(f"❌ Error processing batch {batch_num}: {e}")
            # If we have a vectorstore, continue; otherwise, this is critical
            if vectorstore is None:
                raise e
            continue

    return vectorstore


# --- CORE KNOWLEDGEBASE FUNCTION ---
def process_pdfs_to_text():
    """
    Process all PDFs from the stepan_pdfs/downloads directory and convert them to text files
    Handles encrypted PDFs and provides detailed progress feedback

    Returns:
        tuple: (processed_count, total_count, encrypted_count, failed_count) - processing statistics
    """
    # Ensure directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Get all PDF files from the downloads directory
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        return 0, 0, 0, 0

    processed_count = 0
    encrypted_count = 0
    failed_count = 0

    print(f"📁 Processing {len(pdf_files)} PDF files from {DATA_DIR}")

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            # Check if text file already exists
            text_filename = pdf_file.replace(".pdf", ".txt")
            text_path = os.path.join(PROCESSED_DIR, text_filename)

            if os.path.exists(text_path):
                # Skip if already processed
                processed_count += 1
                if i % 50 == 0:  # Progress update every 50 files
                    print(
                        f"📊 Progress: {i}/{len(pdf_files)} (skipped existing: {text_filename})"
                    )
                continue

            # Process PDF to text
            pdf_path = os.path.join(DATA_DIR, pdf_file)
            text = extract_text_from_pdf(pdf_path)

            if text and text.strip():
                # Save as text file
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(text)
                processed_count += 1
                if i % 50 == 0:  # Progress update every 50 files
                    print(f"📊 Progress: {i}/{len(pdf_files)} (processed: {pdf_file})")
            else:
                print(f"⚠️ No text extracted from: {pdf_file}")
                failed_count += 1

        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            failed_count += 1
            continue

    print(
        f"✅ Processing complete: {processed_count} successful, {failed_count} failed"
    )
    return processed_count, len(pdf_files), encrypted_count, failed_count


def reindex_knowledgebase():
    """
    Re-index the knowledgebase by processing all documents in the processed directory
    This is the core function that:
    1. Reads all text files from the processed directory
    2. Splits them into chunks
    3. Converts chunks to vector embeddings
    4. Stores embeddings in ChromaDB for similarity search

    Returns:
        bool: True if successful, False otherwise
    """

    # Step 1: Process PDFs to text first
    processed_count, total_count, encrypted_count, failed_count = process_pdfs_to_text()
    print(
        f"📁 Processed {processed_count}/{total_count} PDF files ({failed_count} failed)"
    )

    # Step 2: Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Step 3: Process all documents in the processed directory
    docs = []  # List to store all document chunks
    processed_files = []  # List to track which files were processed
    max_docs = 1000  # Limit to prevent memory issues

    # Loop through all files in the processed directory
    for filename in os.listdir(PROCESSED_DIR):
        if filename.endswith(".txt") or filename.endswith(".md"):
            try:
                # Read the file content
                with open(
                    os.path.join(PROCESSED_DIR, filename), "r", encoding="utf-8"
                ) as f:
                    text = f.read()

                    if text.strip():  # Only process non-empty files
                        # Split text into chunks using the text splitter
                        splits = text_splitter.create_documents([text])
                        docs.extend(splits)
                        processed_files.append(filename)

                        # Limit the number of documents to prevent memory issues
                        if len(docs) >= max_docs:
                            print(
                                f"⚠️ Limiting to {max_docs} document chunks to prevent memory issues"
                            )
                            break
                    else:
                        print(f"⚠️ Skipping empty file: {filename}")

            except Exception as e:
                print(f"❌ Error reading file {filename}: {e}")
                continue

    # Step 4: Check if we have any documents to process
    if not docs:
        print("❌ No valid documents found to index.")
        return False

    try:
        print(f"📁 Found {len(processed_files)} file(s) in processed directory")

        # Step 5: Create a new ChromaDB directory in temp space
        new_chroma_dir = create_new_chroma_dir()
        print(f"✅ Created new vectorstore directory: {new_chroma_dir}")

        # Step 6: Create new vectorstore with embeddings using batch processing
        print(f"📁 Creating new vectorstore in temporary directory")
        # This is where the magic happens - documents are converted to vectors in batches
        st.session_state.vectorstore = create_vectorstore_batch(
            docs,  # The document chunks
            embedding_model,  # The embedding model (converts text to vectors)
            new_chroma_dir,  # Where to save the vector database (temp dir)
            batch_size=100,  # Process 100 documents at a time
        )
        print(f"✅ Created new vectorstore in {new_chroma_dir}")

        # Step 7: Update global variable to point to new location
        globals()["CHROMA_DIR"] = new_chroma_dir

        # Step 8: Show success messages
        print(f"✅ Processed {len(processed_files)} files")
        print(f"✅ Created {len(docs)} document chunks")
        print(f"✅ Database saved to temporary directory")

        # Step 9: Test the vectorstore to make sure it works
        try:
            test_results = st.session_state.vectorstore.similarity_search("test", k=1)
            print(f"✅ Vectorstore test successful - found {len(test_results)} results")
        except Exception as e:
            print(f"❌ Vectorstore test failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Error creating vectorstore: {e}")
        print("Please check your AWS credentials and Bedrock access.")
        return False


# --- STREAMLIT APPLICATION SETUP ---
# Configure the main page
PAGE_TITLE = "⚗️🧪Chemical Finder"
st.set_page_config(page_title=PAGE_TITLE)
st.title(PAGE_TITLE)
st.subheader("AI-powred Chemical Search and Discovery Application Using RAG and AWS")
st.write(
    "Example: I'm looking for surfactants that have a viscosity around 4000 cps and dispersible in water."
)

# Initialize vectorstore loading state
if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False

# --- AUTOMATIC INITIALIZATION ---
# Automatically sync from S3, process PDFs and create knowledgebase on app startup
if not st.session_state.vectorstore_loaded:
    # Step 1: Check AWS credentials
    st.info("🔐 Checking AWS credentials...")
    if not check_s3_credentials():
        st.error("❌ AWS credentials not configured. Please check your AWS setup.")
        st.session_state.vectorstore_loaded = False
    else:
        # Step 2: Sync files from S3
        st.info("🔄 Syncing PDF files from S3 bucket...")
        sync_success, sync_message = sync_s3_bucket()

        if not sync_success:
            st.error(f"❌ S3 sync failed: {sync_message}")
            st.session_state.vectorstore_loaded = False
        else:
            st.success(f"✅ {sync_message}")

            # Step 3: Check if we have processed files
            os.makedirs(PROCESSED_DIR, exist_ok=True)
            existing_files = [
                f for f in os.listdir(PROCESSED_DIR) if f.endswith((".txt", ".md"))
            ]

            if existing_files and len(existing_files) > 0:
                # We have processed files, try to load existing vectorstore or create a new one
                st.info(
                    f"📁 Found {len(existing_files)} processed files. Initializing knowledgebase..."
                )

                try:
                    success = reindex_knowledgebase()
                    if success:
                        st.session_state.vectorstore_loaded = True
                        st.session_state.vectorstore_initialized = True
                        st.success("✅ Knowledgebase initialized successfully!")
                        print(
                            "✅ Knowledgebase initialized successfully during startup"
                        )
                    else:
                        st.session_state.vectorstore_loaded = False
                        st.error("❌ Failed to initialize knowledgebase")
                        print("⚠️ Failed to initialize knowledgebase during startup")
                except Exception as e:
                    st.session_state.vectorstore_loaded = False
                    st.error(f"❌ Error during initialization: {str(e)}")
                    print(f"❌ Error during initialization: {e}")
            else:
                # No processed files, need to process PDFs first
                st.info("🔄 Processing PDFs and initializing knowledgebase...")

                try:
                    success = reindex_knowledgebase()
                    if success:
                        st.session_state.vectorstore_loaded = True
                        st.session_state.vectorstore_initialized = True
                        st.success("✅ Knowledgebase initialized successfully!")
                        print(
                            "✅ Knowledgebase initialized successfully during startup"
                        )
                    else:
                        st.session_state.vectorstore_loaded = False
                        st.error("❌ Failed to initialize knowledgebase")
                        print("⚠️ Failed to initialize knowledgebase during startup")
                except Exception as e:
                    st.session_state.vectorstore_loaded = False
                    st.error(f"❌ Error during initialization: {str(e)}")
                    print(f"❌ Error during initialization: {e}")

# --- MAIN Q&A INTERFACE ---
st.header("💬 Ask Questions")

# Check if vectorstore is loaded and functional
if "vectorstore" not in st.session_state or not st.session_state.vectorstore_loaded:
    st.error("❌ Failed to initialize knowledgebase. Please check:")
    st.write("• AWS credentials are properly configured")
    st.write("• AWS S3 and Bedrock access is enabled")
    st.write("• S3 bucket hackaton-stepan-data/stepan-pdfs is accessible")

    # Show current PDF count
    os.makedirs(DATA_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    st.info(f"📁 Found {len(pdf_files)} PDF file(s) in stepan_pdfs/downloads/")

    # Manual sync and retry buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Sync from S3"):
            with st.spinner("Syncing files from S3..."):
                sync_success, sync_message = sync_s3_bucket()
            if sync_success:
                st.success(f"✅ {sync_message}")
                st.rerun()
            else:
                st.error(f"❌ {sync_message}")

    with col2:
        if st.button("🔄 Retry Initialization"):
            with st.spinner("Re-initializing knowledgebase..."):
                success = reindex_knowledgebase()
            if success:
                st.success("✅ Knowledgebase initialized successfully!")
                st.session_state.vectorstore_loaded = True
                st.rerun()
            else:
                st.error(
                    "❌ Initialization failed. Please check the error messages above."
                )
else:
    # Show sync status
    st.success("✅ Knowledgebase is loaded and ready!")

    # Q&A Interface when knowledgebase is loaded
    try:
        # Step 1: Set up retriever (top 3 chunks)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Step 2: Initialize the Bedrock LLM (completion-style)
        from langchain_community.llms import Bedrock

        llm = ChatBedrockConverse(
            region_name=AWS_REGION, model_id=AWS_BEDROCK_LLM_MODEL_ID
        )

        # Step 3: Text input for question
        question = st.text_input("Enter your question:")

        if st.button("Generate Answer") and question:
            with st.spinner("Retrieving and generating answer..."):
                # Retrieve relevant documents
                docs = retriever.get_relevant_documents(question)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Build prompt
                prompt = f"""
You are an AI assistant helping answer questions based on the provided context.

Context:
{context}
"""

                messages = [
                    (
                        "system",
                        prompt,
                    ),
                    (
                        "human",
                        question,
                    ),
                ]

                # Generate completion
                answer = llm.invoke(messages)

                # Display answer
                st.markdown("### 💬 Answer")
                st.write(answer.content)

                # Display sources
                st.markdown("### 📄 Retrieved Context")
                for i, doc in enumerate(docs):
                    st.markdown(f"**Source {i + 1}:**")
                    st.write(doc.page_content[:500] + "...")
    except Exception as e:
        st.error(f"Error during question answering: {e}")
        st.info("Please re-index your knowledgebase.")
        st.session_state.vectorstore_loaded = False
