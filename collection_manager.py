from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional, List


class CollectionManager:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize the collection manager with Qdrant client."""
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, collection_name: str, vector_size: int = 1536) -> bool:
        """
        Create a new collection with specified parameters.

        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the vectors to be stored

        Returns:
            bool: True if collection was created successfully, False otherwise
        """
        try:
            vector_config = models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            )

            self.client.create_collection(
                collection_name=collection_name, vectors_config=vector_config
            )
            print(f"Successfully created collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error creating collection {collection_name}: {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection by name.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if collection was deleted successfully, False otherwise
        """
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"Successfully deleted collection: {collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        List all existing collections.

        Returns:
            List[str]: List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise
        """
        return collection_name in self.list_collections()
