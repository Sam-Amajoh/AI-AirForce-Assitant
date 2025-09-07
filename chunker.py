from typing import List
from pathlib import Path
from config import CHUNK_SIZE, CHUNK_OVERLAP
from llama_index.core import Document

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a large text string into overlapping chunks.
    Args:
        text: The full text to chunk.
        chunk_size: Number of tokens (approx. words) per chunk.
        chunk_overlap: Number of tokens to overlap between chunks.
    Returns:
        A list of text chunks.
    """
    tokens = text.split()
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be larger than chunk_overlap")
    chunks = []
    stride = chunk_size - chunk_overlap
    for i in range(0, len(tokens), stride):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk_tokens))
        if i + chunk_size >= len(tokens):
            break
    return chunks

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Convert a list of llama_index.core.Document into smaller chunked Documents.
    Args:
        docs: List of original Document objects.
    Returns:
        List of chunked Document objects with updated metadata.
    """
    chunked_docs = []
    for doc in docs:
        # Extract text content; adjust attribute as needed (e.g., doc.text or doc.content)
        text = getattr(doc, "text", None) or getattr(doc, "content", "")
        # Perform chunking
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            # Build metadata copying original metadata if exists
            metadata = {}
            if hasattr(doc, "extra_info") and isinstance(doc.extra_info, dict):
                metadata.update(doc.extra_info)
            if hasattr(doc, "doc_id"):
                metadata["doc_id"] = doc.doc_id
            metadata["chunk_index"] = idx
            # Create new Document for each chunk
            chunked_doc = Document(text=chunk, metadata=metadata)
            chunked_docs.append(chunked_doc)
    return chunked_docs