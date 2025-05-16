from qdrant_client import QdrantClient


def main():
    # Initialize the client
    client = QdrantClient(host="localhost", port=6333)

    print("Attempting to connect to Qdrant and list collections...")

    try:
        # Get a list of all collections
        collections = client.get_collections()
        print("Successfully connected to Qdrant.")
        if collections.collections:
            print("Existing collections:")
            for collection in collections.collections:
                print(f"- {collection.name}")
        else:
            print("No collections found.")
    except Exception as e:
        print(f"Could not connect to Qdrant or an error occurred: {e}")
        print(
            "Please ensure Qdrant is running. You can start it with: docker-compose up -d"
        )


if __name__ == "__main__":
    main()
