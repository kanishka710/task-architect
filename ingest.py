import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize the Embedding Model (Runs on your CPU)
# This model turns "words" into a list of numbers
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Initialize ChromaDB (Local Storage)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="research_papers")

# Create the splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

def process_pdfs(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            print(f"Processing {filename}...")
            reader = PdfReader(os.path.join(folder_path, filename))


            # Extract text from each page and create chunks
            for i, page in enumerate(reader.pages):
                text = page.extract_text()

                if not text or not text.strip():
                    continue  # Skip empty or unreadable pages

                # Break the page into overlapping chunks
                chunks = text_splitter.split_text(text)

                for j, chunk in enumerate(chunks):
                    # Unique ID for every single piece
                    chunk_id = f"{filename}_p{i}_c{j}"

                    vector = embed_model.encode(chunk).tolist()

                    collection.add(
                        ids=[chunk_id],
                        embeddings=[vector],
                        documents=[chunk],
                        metadatas=[{"source": filename, "page": i}]
                    )
                    
    print("Ingestion Complete!")

if __name__ == "__main__":
    process_pdfs("./papers")