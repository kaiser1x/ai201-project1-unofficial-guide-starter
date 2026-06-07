import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Milestone 5) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# --- Embeddings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Vector store ---
CHROMA_COLLECTION = "unofficial_guide"
CHROMA_PATH = "./chroma_db"

# --- Documents ---
DOCS_PATH = "./documents"

# --- Chunking (blueprint-locked) ---
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

# --- Retrieval ---
N_RESULTS = 4
