import chromadb


def inspect_collection(collection_name: str = "research_papers"):
    client = chromadb.PersistentClient(path="./chroma_db")
    print("Collections:", client.list_collections())

    collection = client.get_collection(name=collection_name)
    item_count = collection.count()
    print("Total documents in collection:", item_count)

    if item_count > 0:
        sample = collection.peek(1)
        print("Sample data:", sample["documents"])

if __name__ == "__main__":
    inspect_collection()