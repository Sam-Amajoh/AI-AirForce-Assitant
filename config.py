from dotenv import load_dotenv
import os
load_dotenv() 
# Hugging Face API token (set this in your environment as HF_TOKEN)
HF_TOKEN = os.getenv("HF_TOKEN")

from pathlib import Path


# Embedding model configuration
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM model configuration
LLM_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

# Directory to store uploaded TO manuals
TO_MANUALS_DIR = Path(__file__).parent / "Documents"


# Chunking parameters
CHUNK_SIZE = 1000       # number of tokens per chunk
CHUNK_OVERLAP = 200     # overlap between chunks to preserve context