import pickle
from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from config import HF_TOKEN, EMBED_MODEL_NAME, LLM_MODEL_NAME, TO_MANUALS_DIR
from chunker import chunk_documents
from dotenv import load_dotenv
load_dotenv()   # now os.environ["HF_TOKEN"] is populated

import os
HF_TOKEN = os.environ["HF_TOKEN"]

# Path to persist the vector store index
INDEX_PATH = Path(__file__).parent / 'vector_index.pkl'

def build_or_load_index() -> VectorStoreIndex:
    """
    Load an existing index if found, otherwise build a new one from PDFs in TO_MANUALS_DIR.
    """
    if INDEX_PATH.exists():
        with open(INDEX_PATH, 'rb') as f:
            index = pickle.load(f)
    else:
        # Ensure the directory exists
        TO_MANUALS_DIR.mkdir(exist_ok=True)
        # Read all PDF manuals
        reader = SimpleDirectoryReader(
            str(TO_MANUALS_DIR),
            file_extractor={'.pdf': PyMuPDFReader()}
        )
        raw_docs = reader.load_data()
        # Chunk documents
        docs = chunk_documents(raw_docs)
        # Create embedding model
        embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME
        )
        # Build the index
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        # Persist it
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(index, f)
    return index

def reload_index_for_files(new_files: List[Path]) -> None:
    """
    Embed and insert newly uploaded PDFs into the existing index, then persist it.
    """
    # 1) (Re)load the existing index from disk (or build a new one if missing)
    index = build_or_load_index()

    # 2) Read *only* the newly uploaded PDFs
    reader = SimpleDirectoryReader(
        input_files=[str(TO_MANUALS_DIR / p.name) for p in new_files],
        file_extractor={".pdf": PyMuPDFReader()}
    )
    raw_docs = reader.load_data()  # no args

    # 3) Chunk the raw documents into manageable pieces
    docs = chunk_documents(raw_docs)

    # 4) Prepare the same embedding model used for the original index
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME
    )

    # 5) Insert each chunk into the index
    for doc in docs:
        index.insert(doc)

    # 6) Persist the updated index back to disk
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
