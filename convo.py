import os
import argparse
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

# --- Configuration ---
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = "models_cache"

def create_faiss_index(source_path: str, index_path: str):
    """
    Reads a source file (.csv or .txt), normalizes embeddings, and creates a FAISS vector index.

    Args:
        source_path (str): The path to the source data file.
        index_path (str): The directory where the FAISS index will be saved.
    """
    documents = []
    print(f"Reading source file from: {source_path}")

    if not os.path.exists(source_path):
        print(f"❌ ERROR: Source file '{source_path}' not found.")
        return

    # Intelligently handle different file types
    if source_path.endswith('.csv'):
        df = pd.read_csv(source_path)
        # Convert each CSV row into a Document object
        for index, row in df.iterrows():
            content = ". ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            documents.append(Document(page_content=content, metadata={"source": source_path, "row": index}))
        print(f"Loaded {len(documents)} documents from CSV.")

    elif source_path.endswith('.txt'):
        with open(source_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        # Split text into smaller chunks for better semantic search
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(text_content)
        documents = [Document(page_content=t, metadata={"source": source_path}) for t in texts]
        print(f"Split text file into {len(documents)} documents.")
    else:
        print(f"❌ ERROR: Unsupported file type for '{source_path}'. Please use .csv or .txt.")
        return

    if not documents:
        print("⚠️ WARNING: No documents were generated from the source file.")
        return

    print("Initializing embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_DIR,
        # 💡 CORE FIX: Normalize embeddings to ensure scores are between 0 and 1.
        # This makes the vector math equivalent to Cosine Similarity.
        model_kwargs={'device': 'cpu'}, # Use CPU, can be changed to 'cuda' if GPU is available
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"Creating FAISS vector index at '{index_path}'...")
    # FAISS will now use normalized vectors, solving the score issue.
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(index_path)
    print(f"✅ Index saved successfully to: '{index_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a FAISS vector index from a source file (.csv or .txt).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- COMMAND-LINE ARGUMENT FIX ---
    # Define the names for the arguments properly.
    parser.add_argument("source_file", help="Path to the source data file (e.g., 'data_files/conversations_pt.csv').")
    parser.add_argument("index_directory", help="Name of the directory to save the new index (e.g., 'company_faiss_index').")
    
    args = parser.parse_args()
    
    # Call the function with the correctly parsed arguments
    create_faiss_index(args.source_file, args.index_directory)
    
    print("\n--- Process Complete ---")