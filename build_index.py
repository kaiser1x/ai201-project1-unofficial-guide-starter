"""
One-shot index builder: load documents -> chunk -> embed -> store in ChromaDB.

Run this once after changing documents/, chunking, or the embedding model:

    python build_index.py

It resets the collection first, so it is safe to re-run — no duplicate ids.
"""
from ingest import load_documents
from chunk import chunk_documents
from retriever import embed_and_store, get_collection, reset_collection


def main():
    print("Resetting collection...")
    reset_collection()

    documents = load_documents()
    chunks = chunk_documents(documents)
    embed_and_store(chunks)

    print(f"\n✅ Index built: {get_collection().count()} chunks stored.")


if __name__ == "__main__":
    main()
