import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_COLLECTION, CHROMA_PATH, EMBEDDING_MODEL, N_RESULTS

# Embedding function + persistent client initialized once at module load.
# sentence-transformers downloads all-MiniLM-L6-v2 on first use, then caches it.
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)
_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    embedding_function=_ef,
    metadata={"hnsw:space": "cosine"},
)


def get_collection():
    """Return the active ChromaDB collection."""
    return _collection


def reset_collection():
    """Drop and recreate the collection so build_index can rebuild cleanly."""
    global _collection
    try:
        _client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass  # collection didn't exist yet
    _collection = _client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def embed_and_store(chunks):
    """
    Embed chunks and store them in the vector DB.

    Each chunk's source metadata (source_file, doc_type) is stored alongside
    its vector so retrieval can attribute and filter results. chunk_id is the
    unique ChromaDB id.
    """
    if not chunks:
        print("No chunks to store.")
        return

    _collection.add(
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source_file": c["source_file"], "doc_type": c["doc_type"]}
            for c in chunks
        ],
        ids=[c["chunk_id"] for c in chunks],
    )
    print(f"Stored {_collection.count()} chunks in the vector database.")


def retrieve(query, n_results=N_RESULTS, doc_type=None):
    """
    Semantic top-k retrieval over the collection (cosine distance).

    Optional `doc_type` applies a metadata filter (e.g. "Lease Contract") to
    restrict the search to one document category.

    Returns a list of dicts ordered most- to least-relevant:
      - "text"        : chunk text
      - "source_file" : originating filename
      - "doc_type"    : document type
      - "distance"    : cosine distance (lower = more similar)
    """
    if _collection.count() == 0:
        return []

    query_kwargs = {
        "query_texts": [query],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if doc_type is not None:
        query_kwargs["where"] = {"doc_type": doc_type}

    results = _collection.query(**query_kwargs)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text": text,
            "source_file": meta.get("source_file", "?"),
            "doc_type": meta.get("doc_type", "?"),
            "distance": distance,
        }
        for text, meta, distance in zip(documents, metadatas, distances)
    ]
