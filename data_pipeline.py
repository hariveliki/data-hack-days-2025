from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import logging
from pdfplumber_parser import PDFPlumberParser, TextCleaner
from collection_manager import CollectionManager
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Class to handle the complete data pipeline from PDF processing to Qdrant storage."""

    def __init__(self, collection_name: str = "motions", vector_size: int = 384):
        """Initialize the data pipeline with PDF parser and Qdrant collection manager.

        Args:
            collection_name (str): Name of the Qdrant collection
            vector_size (int): Size of the embedding vectors
        """
        self.pdf_parser = PDFPlumberParser()
        self.collection_manager = CollectionManager()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.text_cleaner = TextCleaner()
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Create collection if it doesn't exist
        if not self.collection_manager.collection_exists(collection_name):
            logger.info(
                f"Creating collection {collection_name} with vector size {vector_size}"
            )
            self.collection_manager.create_collection(collection_name, vector_size)
        else:
            logger.info(f"Using existing collection {collection_name}")

    def process_and_store(self, samples: int = 0) -> None:
        """Process PDFs and store the extracted text in Qdrant.

        Args:
            samples (int): Number of samples to process. If 0, process all.
        """
        # Read the filtered motions data
        if samples > 0:
            df = pd.read_csv("data/filtered_motions.csv").head(samples)
        else:
            df = pd.read_csv("data/filtered_motions.csv")

        # Process each document
        for _, row in df.iterrows():
            try:
                # Process the PDF
                result = self.pdf_parser.process_url(
                    row["dokument_url"], str(row["geschaeft_uid"])
                )

                if result["status"] == "success" and result["text"]:
                    # Clean the text
                    cleaned_text = self.text_cleaner.clean_text(result["text"])

                    # Generate embedding
                    embedding = self.embedding_model.encode(cleaned_text)

                    # Verify embedding dimension
                    if len(embedding) != self.vector_size:
                        logger.error(
                            f"Embedding dimension mismatch: got {len(embedding)}, expected {self.vector_size}"
                        )
                        continue

                    # Prepare metadata
                    metadata = {
                        "geschaeft_uid": str(row["geschaeft_uid"]),
                        "dokument_uid": str(row["dokument_uid"]),
                        "url": row["dokument_url"],
                        "text": cleaned_text,
                    }

                    # Generate a UUID for the point ID
                    point_id = str(uuid.uuid4())

                    # Store in Qdrant
                    self.collection_manager.client.upsert(
                        collection_name=self.collection_name,
                        points=[
                            models.PointStruct(
                                id=point_id,
                                vector=embedding.tolist(),
                                payload=metadata,
                            )
                        ],
                    )

                    logger.info(
                        f"Successfully processed and stored document {row['dokument_uid']}"
                    )
                else:
                    logger.warning(f"Failed to process document {row['dokument_uid']}")

            except Exception as e:
                logger.error(
                    f"Error processing document {row['dokument_uid']}: {str(e)}"
                )
                continue

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using a text query.

        Args:
            query (str): The search query
            limit (int): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Search in Qdrant
        search_results = self.collection_manager.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit,
        )

        # Format results
        results = []
        for hit in search_results:
            results.append({"score": hit.score, "metadata": hit.payload})

        return results
