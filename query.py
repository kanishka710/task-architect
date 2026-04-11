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
collection = db_client.get_collection(name="research_papers")

def ask_librarian(question: str):
    # Step A: Turn the question into a vector
    query_vector = embed_model.encode(question).tolist()
    
    # Step B: Retrieve the top 2 most relevant chunks from Chroma
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2
    )
    
    context_text = "\n".join(results['documents'][0])
    sources = results['metadatas'][0]

    # Step C: The "Grounded" Prompt
    prompt = f"""
    You are a Research Assistant. Use the provided context from research papers 
    to answer the question. If the answer isn't in the context, say you don't know.
    
    CONTEXT:
    {context_text}
    
    QUESTION:
    {question}
    
    ANSWER:
    """

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    print("\n--- LIBRARIAN'S ANSWER ---")
    print(response.text)
    print("\n--- SOURCES ---")
    for source in sources:
        print(f"File: {source['source']}, Page: {source['page']}")

if __name__ == "__main__":
    user_query = input("What do you want to know from your library? ")
    ask_librarian(user_query)