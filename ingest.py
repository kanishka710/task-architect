import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Initialize the Embedding Model (Runs on your CPU)
# This model turns "words" into a list of numbers
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize ChromaDB (Local Storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_papers")

def process_pdfs(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")
            reader = PdfReader(os.path.join(folder_path, filename))
            
            # Extract text page by page
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    # Generate a unique ID for this chunk
                    chunk_id = f"{filename}_page_{i}"
                    
                    # Convert text to numbers (Vector)
                    vector = embed_model.encode(text).tolist()
                    
                    # Store in our Database
                    collection.add(
                        ids=[chunk_id],
                        embeddings=[vector],
                        documents=[text],
                        metadatas=[{"source": filename, "page": i}]
                    )
    print("Ingestion Complete!")

if __name__ == "__main__":
    process_pdfs("./papers")