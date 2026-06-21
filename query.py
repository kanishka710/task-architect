import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
import os
from dotenv import load_dotenv

# 1. SETUP
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to our existing ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="research_papers")


def retrieve_context(question: str, top_k: int = 2):
    query_vector = embed_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return documents, metadatas


def build_grounded_prompt(context_text: str, question: str) -> str:
    return f"""
    You are a Research Assistant. Use the provided context from research papers
    to answer the question. If the answer isn't in the context, say you don't know.

    CONTEXT:
    {context_text}

    QUESTION:
    {question}

    ANSWER:
    """


def print_sources(sources):
    print("\n--- SOURCES ---")
    for source in sources:
        print(f"File: {source['source']}, Page: {source['page']}")

def ask_librarian(question: str):
    documents, sources = retrieve_context(question)

    if not documents:
        print("\n--- LIBRARIAN'S ANSWER ---")
        print("No relevant documents were found in the library.")
        return

    context_text = "\n".join(documents)
    prompt = build_grounded_prompt(context_text, question)

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    print("\n--- LIBRARIAN'S ANSWER ---")
    print(response.text)
    print_sources(sources)

if __name__ == "__main__":
    user_query = input("What do you want to know from your library? ")
    ask_librarian(user_query)