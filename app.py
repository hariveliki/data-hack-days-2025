from collection_manager import CollectionManager


def main():
    manager = CollectionManager()

    try:
        collections = manager.list_collections()
        print("\nExisting collections:")
        if collections:
            for collection in collections:
                print(f"- {collection}")
        else:
            print("No collections found.")

    except Exception as e:
        print(f"Could not connect to Qdrant or an error occurred: {e}")
        print(
            "Please ensure Qdrant is running. You can start it with: docker-compose up -d"
        )


if __name__ == "__main__":
    main()
