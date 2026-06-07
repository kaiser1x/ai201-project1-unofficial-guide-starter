import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP

# RecursiveCharacterTextSplitter tries separators in order, falling back to
# finer ones only when a chunk is still too big. Ordering paragraph -> line ->
# sentence -> word keeps tabular/bulleted rows in lease, tax, and inspection
# docs from being severed mid-row whenever a coarser boundary fits.
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_documents(documents):
    """
    Split loaded documents into overlapping chunks for embedding.

    `documents` is the list returned by ingest.load_documents(). Each chunk
    propagates the source metadata intact so retrieval can attribute results:
      - "text"        : the chunk text (str)
      - "source_file" : original filename, passed through unchanged
      - "doc_type"    : document type, passed through unchanged
      - "chunk_id"    : unique id, e.g. "lease-agreement-simple-form_0"
    """
    chunks = []
    for doc in documents:
        prefix = os.path.splitext(doc["source_file"])[0].lower().replace(" ", "_")
        pieces = _splitter.split_text(doc["text"])

        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "text": piece,
                "source_file": doc["source_file"],
                "doc_type": doc["doc_type"],
                "chunk_id": f"{prefix}_{i}",
            })

    print(f"Produced {len(chunks)} chunk(s) from {len(documents)} document(s).")
    return chunks


if __name__ == "__main__":
    from ingest import load_documents

    docs = chunk_documents(load_documents())
    # Preview the first chunk of each document so boundaries are inspectable.
    seen = set()
    for c in docs:
        if c["source_file"] not in seen:
            seen.add(c["source_file"])
            print(f"\n[{c['doc_type']}] {c['chunk_id']}")
            print(f"  {c['text'][:120]!r}")
