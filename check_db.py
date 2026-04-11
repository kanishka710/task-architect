import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
# List all collections to make sure we didn't have a typo
print("Collections:", client.list_collections())

collection = client.get_collection(name="research_papers")
# Count how many items are inside
print("Total documents in collection:", collection.count())

# Peek at the first item to see what the text looks like
if collection.count() > 0:
    print("Sample data:", collection.peek(1)['documents'])