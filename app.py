from collection_manager import CollectionManager
from data_pipeline import DataPipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    # Delete existing collection if it exists
    collection_name = "motions"
    manager = CollectionManager()
    if manager.collection_exists(collection_name):
        logger.info(f"Deleting existing collection: {collection_name}")
        manager.delete_collection(collection_name)

    # Initialize the data pipeline
    pipeline = DataPipeline(collection_name=collection_name)

    try:
        # Process and store documents
        logger.info("Starting document processing and storage...")
        pipeline.process_and_store()
        logger.info("Document processing and storage completed.")

        # Example search
        query = "Antrag Regierungsrat"
        logger.info(f"Searching for documents similar to: {query}")
        results = pipeline.search_similar(query, limit=5)

        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document ID: {result['metadata']['document_id']}")
            print(f"   Motion ID: {result['metadata']['motion_id']}")
            print(f"   Similarity Score: {result['score']:.4f}")
            print(f"   URL: {result['metadata']['url']}")
            print(f"   Text Preview: {result['metadata']['text'][:200]}...")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(
            "Please ensure Qdrant is running. You can start it with: docker-compose up -d"
        )


if __name__ == "__main__":
    main()
