import os
import logging
import atexit
import warnings
from coderag.index import clear_index, add_to_index, save_index, load_index_if_exists, file_exists_in_index, remove_file_from_index
from coderag.embeddings import generate_embeddings
from coderag.config import WATCHED_DIR
from coderag.monitor import start_monitoring, should_ignore_path
from utils.file_tools import get_file_hash

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

def incremental_reindex():
    """Perform incremental reindexing - only process new or changed files."""
    logging.info("Starting incremental reindexing...")
    
    # Try to load existing index
    index_loaded = load_index_if_exists()
    
    files_processed = 0
    files_skipped = 0
    files_updated = 0
    
    # Collect all Python files first
    python_files = []
    for root, _, files in os.walk(WATCHED_DIR):
        if should_ignore_path(root):
            logging.info(f"Ignoring directory: {root}")
            continue

        for file in files:
            filepath = os.path.join(root, file)
            if should_ignore_path(filepath):
                continue

            if file.endswith(".py"):
                python_files.append(filepath)
    
    logging.info(f"Found {len(python_files)} Python files to check")
    
    for filepath in python_files:
        try:
            # Get current file hash
            current_hash = get_file_hash(filepath)
            if current_hash is None:
                continue
            
            relative_filepath = os.path.relpath(filepath, WATCHED_DIR)
            filename = os.path.basename(filepath)
            
            # Check if file exists in index and if it has changed
            existing_entry = file_exists_in_index(relative_filepath)
            
            if existing_entry and existing_entry.get('hash') == current_hash:
                # File hasn't changed, skip embedding
                logging.info(f"Skipping unchanged file: {relative_filepath}")
                files_skipped += 1
                continue
            
            # File is new or changed - process it
            if existing_entry:
                logging.info(f"Processing changed file: {relative_filepath}")
            else:
                logging.info(f"Processing new file: {relative_filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                full_content = f.read()

            embeddings = generate_embeddings(full_content)
            if embeddings is not None:
                # If file existed before, remove old entry first
                if existing_entry:
                    remove_file_from_index(relative_filepath)
                    files_updated += 1
                else:
                    files_processed += 1
                
                # Add new entry with hash
                add_to_index(embeddings, full_content, filename, filepath, current_hash)
            else:
                logging.warning(f"Failed to generate embeddings for {filepath}")
                
        except Exception as e:
            logging.error(f"Error processing file {filepath}: {e}")

    save_index()
    logging.info(f"Incremental reindexing completed. New files: {files_processed}, Updated files: {files_updated}, Skipped files: {files_skipped}")

def full_reindex():
    """Perform a full reindex of the entire codebase."""
    logging.info("Starting full reindexing of the codebase...")
    
    # Clear existing index
    clear_index()
    
    files_processed = 0
    for root, _, files in os.walk(WATCHED_DIR):
        if should_ignore_path(root):
            logging.info(f"Ignoring directory: {root}")
            continue

        for file in files:
            filepath = os.path.join(root, file)
            if should_ignore_path(filepath):
                logging.info(f"Ignoring file: {filepath}")
                continue

            if file.endswith(".py"):
                logging.info(f"Processing file: {filepath}")
                try:
                    current_hash = get_file_hash(filepath)
                    if current_hash is None:
                        continue
                        
                    with open(filepath, 'r', encoding='utf-8') as f:
                        full_content = f.read()

                    embeddings = generate_embeddings(full_content)
                    if embeddings is not None:
                        add_to_index(embeddings, full_content, file, filepath, current_hash)
                    else:
                        logging.warning(f"Failed to generate embeddings for {filepath}")
                    files_processed += 1
                except Exception as e:
                    logging.error(f"Error processing file {filepath}: {e}")

    save_index()
    logging.info(f"Full reindexing completed. {files_processed} files processed.")

def main():
    # Check if we should do full or incremental reindex
    # You can add a command line argument or config option to force full reindex
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        # Force full reindex
        full_reindex()
    else:
        # Try incremental reindex first
        incremental_reindex()

    # Start monitoring the directory for changes
    start_monitoring()

if __name__ == "__main__":
    main()