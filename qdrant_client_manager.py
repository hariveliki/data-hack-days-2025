from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional, List, Dict


class QdrantClientManager:
    DENSE_VECTOR_SIZE = 1024
    SPARSE_VECTOR_SIZE = 256
    MATRYOSHKA_64_SIZE = 64
    MATRYOSHKA_128_SIZE = 128
    MATRYOSHKA_256_SIZE = 256

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize the Qdrant client manager."""
        self.client = QdrantClient(host=host, port=port)
        self.TEXT_VECTOR_NAME = "text_dense"
        self.MATRYOSHKA_64 = "matryoshka-64dim"
        self.MATRYOSHKA_128 = "matryoshka-128dim"
        self.MATRYOSHKA_256 = "matryoshka-256dim"
        self.SPARSE_VECTOR_NAME = "text_sparse"

    def create_collection(
        self, collection_name: str, vector_size: int = DENSE_VECTOR_SIZE
    ) -> bool:
        """
        Create a new collection with hybrid search configuration.

        Args:
            collection_name: Name of the collection to create
            vector_size: Size of the dense vectors to be stored (defaults to DENSE_VECTOR_SIZE)

        Returns:
            bool: True if collection was created successfully, False otherwise
        """
        try:
            vectors_config = {
                self.TEXT_VECTOR_NAME: models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                ),
                # Use uint8 for faster retrieval in initial stages
                self.MATRYOSHKA_64: models.VectorParams(
                    size=self.MATRYOSHKA_64_SIZE,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8
                        )
                    ),
                ),
                self.MATRYOSHKA_128: models.VectorParams(
                    size=self.MATRYOSHKA_128_SIZE,
                    distance=models.Distance.COSINE,
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8
                        )
                    ),
                ),
                self.MATRYOSHKA_256: models.VectorParams(
                    size=self.MATRYOSHKA_256_SIZE, distance=models.Distance.COSINE
                ),
                self.SPARSE_VECTOR_NAME: models.VectorParams(
                    size=self.SPARSE_VECTOR_SIZE, distance=models.Distance.DOT
                ),
            }

            # Note: For even better search quality, consider adding a late interaction model
            # (like ColBERT) for reranking. This would require additional vector configuration
            # and would be used in the final reranking step of the search pipeline.

            self.client.create_collection(
                collection_name=collection_name, vectors_config=vectors_config
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

    def hybrid_search(
        self,
        collection_name: str,
        query_embedding: List[float],
        sparse_query: List[float],
        limit: int = 5,
    ) -> List[Dict]:
        """
        Perform hybrid search using both dense and sparse vectors.
        Implements a multi-stage search pipeline with:
        1. Fast initial retrieval using quantized Matryoshka vectors
        2. Dense vector search for semantic matching
        3. Sparse vector search for keyword matching
        4. RRF fusion to combine results from all branches

        Args:
            collection_name: Name of the collection to search in
            query_embedding: Dense vector embedding of the query
            sparse_query: Sparse vector representation of the query
            limit: Maximum number of results to return

        Returns:
            List[Dict]: List of search results
        """
        try:
            # Prepare matryoshka vectors
            matryoshka_query_64 = query_embedding[:64]
            matryoshka_query_128 = query_embedding[:128]
            matryoshka_query_256 = query_embedding[:256]

            # Matryoshka branch (multi-stage with oversampling)
            # Start with 64-dim for fast initial retrieval, then refine with larger dimensions
            matryoshka_prefetch = models.Prefetch(
                prefetch=[
                    models.Prefetch(
                        prefetch=[
                            models.Prefetch(
                                query=matryoshka_query_64,
                                using=self.MATRYOSHKA_64,
                                limit=200,  # Oversample for better recall
                            ),
                        ],
                        query=matryoshka_query_128,
                        using=self.MATRYOSHKA_128,
                        limit=100,  # Reduce candidates for next stage
                    )
                ],
                query=matryoshka_query_256,
                using=self.MATRYOSHKA_256,
                limit=50,  # Further reduce for final stage
            )

            # Dense branch for semantic search
            dense_prefetch = models.Prefetch(
                query=query_embedding,
                using=self.TEXT_VECTOR_NAME,
                limit=limit * 2,  # Oversample for better recall
            )

            # Sparse branch for keyword matching
            sparse_prefetch = models.Prefetch(
                query=sparse_query,
                using=self.SPARSE_VECTOR_NAME,
                limit=limit * 2,  # Oversample for better recall
            )

            # Fusion of all branches using RRF (Reciprocal Rank Fusion)
            # RRF is preferred over linear combination as it handles score distributions better
            search_result = self.client.query_points(
                collection_name=collection_name,
                prefetch=[matryoshka_prefetch, dense_prefetch, sparse_prefetch],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )

            return search_result.points

        except Exception as e:
            print(f"Error performing hybrid search: {e}")
            return []
