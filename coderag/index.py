import os
import faiss
import numpy as np
import logging
from coderag.config import EMBEDDING_DIM, FAISS_INDEX_FILE, WATCHED_DIR

index = faiss.IndexFlatL2(EMBEDDING_DIM)
metadata = []

def clear_index():
    """Delete the FAISS index and metadata files if they exist, and reinitialize the index."""
    global index, metadata
    
    # Delete the FAISS index file
    if os.path.exists(FAISS_INDEX_FILE):
        os.remove(FAISS_INDEX_FILE)
        logging.info(f"Deleted FAISS index file: {FAISS_INDEX_FILE}")

    # Delete the metadata file
    metadata_file = "metadata.npy"
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
        logging.info(f"Deleted metadata file: {metadata_file}")

    # Reinitialize the FAISS index and metadata
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata = []
    logging.info("FAISS index and metadata cleared and reinitialized.")

def load_index_if_exists():
    """Load existing index and metadata if they exist."""
    global index, metadata
    
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists("metadata.npy"):
        try:
            index = faiss.read_index(FAISS_INDEX_FILE)
            with open("metadata.npy", "rb") as f:
                metadata = np.load(f, allow_pickle=True).tolist()
            logging.info(f"Loaded existing index with {index.ntotal} entries")
            return True
        except Exception as e:
            logging.error(f"Error loading existing index: {e}")
            # Reinitialize if loading fails
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            metadata = []
            return False
    else:
        # Initialize empty index
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = []
        logging.info("No existing index found, starting fresh")
        return False

def file_exists_in_index(relative_filepath):
    """Check if a file already exists in the index and return its metadata."""
    global metadata
    
    for entry in metadata:
        if entry.get('filepath') == relative_filepath:
            return entry
    return None

def remove_file_from_index(relative_filepath):
    """Remove a file's entries from the index (for updates)."""
    global index, metadata
    
    # Find indices to remove
    indices_to_remove = []
    new_metadata = []
    
    for i, entry in enumerate(metadata):
        if entry.get('filepath') == relative_filepath:
            indices_to_remove.append(i)
        else:
            new_metadata.append(entry)
    
    if not indices_to_remove:
        logging.info(f"No existing entries found for {relative_filepath}")
        return
    
    logging.info(f"Removing {len(indices_to_remove)} existing entries for {relative_filepath}")
    
    # Since FAISS doesn't support direct removal, we need to rebuild the index
    # Create new index
    new_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    
    # Add all vectors except the ones we want to remove
    vectors_to_keep = []
    for i in range(len(metadata)):
        if i not in indices_to_remove:
            vector = index.reconstruct(i).reshape(1, -1)
            vectors_to_keep.append(vector)
    
    # Add vectors to new index
    if vectors_to_keep:
        all_vectors = np.vstack(vectors_to_keep)
        new_index.add(all_vectors)
    
    # Update global variables
    index = new_index
    metadata = new_metadata
    
    logging.info(f"Index rebuilt with {index.ntotal} entries remaining")

def add_to_index(embeddings, full_content, filename, filepath, file_hash=None):
    """Add embeddings to the index with optional file hash for change detection."""
    global index, metadata

    if embeddings.shape[1] != index.d:
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match FAISS index dimension {index.d}")

    # Convert absolute filepath to relative path
    relative_filepath = os.path.relpath(filepath, WATCHED_DIR)

    index.add(embeddings)
    
    entry = {
        "content": full_content,
        "filename": filename,
        "filepath": relative_filepath
    }
    
    # Add file hash if provided
    if file_hash:
        entry["hash"] = file_hash
    
    metadata.append(entry)

def save_index():
    """Save the FAISS index and metadata to disk."""
    if index.ntotal > 0:  # Only save if there are entries
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open("metadata.npy", "wb") as f:
            np.save(f, metadata)
        logging.info(f"Saved index with {index.ntotal} entries")
    else:
        logging.warning("No entries to save in index")

def load_index():
    """Load the FAISS index and metadata from disk."""
    global index, metadata
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open("metadata.npy", "rb") as f:
        metadata = np.load(f, allow_pickle=True).tolist()
    return index

def get_metadata():
    """Get the current metadata."""
    return metadata

def retrieve_vectors(n=5):
    """Retrieve the first n vectors from the index."""
    if index.ntotal == 0:
        return np.array([])
    
    n = min(n, index.ntotal)
    vectors = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    return vectors

def inspect_metadata(n=5):
    """Inspect the first n metadata entries."""
    metadata = get_metadata()
    print(f"Inspecting the first {n} metadata entries:")
    for i, data in enumerate(metadata[:n]):
        print(f"Entry {i}:")
        print(f"Filename: {data['filename']}")
        print(f"Filepath: {data['filepath']}")
        if 'hash' in data:
            print(f"Hash: {data['hash']}")
        print(f"Content: {data['content'][:100]}...")  # Show the first 100 characters
        print()

def get_index_stats():
    """Get statistics about the current index."""
    return {
        "total_entries": index.ntotal,
        "total_files": len(set(entry['filepath'] for entry in metadata)),
        "embedding_dimension": index.d
    }