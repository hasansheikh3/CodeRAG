import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# === Environment Variables ===
# OpenAI API key and model settings (loaded from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")  # Default to ada-002
VOYAGEAI_EMBEDDING_MODEL = os.getenv("VOYAGEAI_EMBEDDING_MODEL", "voyage-code-3")  # Default to VoyageAI embedding model
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4")  # Default to GPT-4
VOYAGEAI_API_KEY = os.getenv("VOYAGEAI_API_KEY")  # Optional, for VoyageAI integration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Optional, for Gemini integration
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-exp-03-07")  # Default to Gemini embedding model
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")  # Default to Gemini chat model

# Embedding dimension (from .env or fallback)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))  # Default to 1536 if not in .env

# Project directory (from .env)
# Path to FAISS index (from .env or fallback)

# === Project-Specific Configuration ===
# Define the root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "faiss_index.bin")
METADATA_FILE = os.path.join(PROJECT_ROOT, "data", "metadata.npy")

DATA_DIR = os.path.dirname(FAISS_INDEX_FILE)
os.makedirs(DATA_DIR, exist_ok=True)

WATCHED_DIR = PROJECT_ROOT

# Additional directories to ignore during indexing (these can remain static)
IGNORE_PATHS = [
    os.path.join(WATCHED_DIR, ".venv"),
    os.path.join(WATCHED_DIR, "node_modules"),
    os.path.join(WATCHED_DIR, "__pycache__"),
    os.path.join(WATCHED_DIR, ".git"),
    os.path.join(WATCHED_DIR, "tests"),
]


