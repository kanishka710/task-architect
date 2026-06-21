import hashlib
from pathlib import Path

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

def iter_pdf_paths(folder_path: str):
    folder = Path(folder_path)
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() == ".pdf" and file_path.is_file():
            yield file_path


def build_chunk_id(source_name: str, page_number: int, chunk_number: int, chunk_text: str) -> str:
    digest_input = f"{source_name}:{page_number}:{chunk_number}:{chunk_text}".encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()
    return f"{source_name}_p{page_number}_c{chunk_number}_{digest[:12]}"


def process_pdf(file_path: Path) -> int:
    print(f"Processing {file_path.name}...")
    reader = PdfReader(str(file_path))
    chunk_count = 0

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        for chunk_number, chunk_text in enumerate(text_splitter.split_text(text)):
            chunk_id = build_chunk_id(file_path.name, page_number, chunk_number, chunk_text)
            vector = embed_model.encode(chunk_text).tolist()

            collection.upsert(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk_text],
                metadatas=[{"source": file_path.name, "page": page_number}],
            )
            chunk_count += 1

    return chunk_count


def process_pdfs(folder_path):
    total_chunks = 0

    for pdf_path in iter_pdf_paths(folder_path):
        total_chunks += process_pdf(pdf_path)

    print(f"Ingestion Complete! Stored {total_chunks} chunks.")

if __name__ == "__main__":
    process_pdfs("./papers")