from typing import List, Dict, Any, Optional, Protocol
import pandas as pd
from pathlib import Path
import logging
from src.parsers.pdfminer_parser import PDFMinerParser
from src.db.qdrant_client_manager import QdrantClientManager
from qdrant_client.http import models
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
from rank_bm25 import BM25Okapi
from collections import Counter
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFParser(Protocol):
    """Protocol defining the interface for PDF parsers."""

    def process_url(self, url: str, motion_id: str) -> Dict[str, Any]:
        """Process a PDF URL and return the extracted text."""
        ...


class DataPipeline:
    """Class to handle the complete data pipeline from PDF processing to Qdrant storage."""

    def __init__(
        self,
        collection_name: str = "motions",
        vector_size: int = QdrantClientManager.DENSE_VECTOR_SIZE,
        parser: Optional[PDFParser] = None,
    ):
        """Initialize the data pipeline with PDF parser and Qdrant client manager.

        Args:
            collection_name (str): Name of the Qdrant collection
            vector_size (int): Size of the embedding vectors (defaults to 1024 for BGE-M3)
            parser (Optional[PDFParser]): PDF parser to use. If None, defaults to PDFPlumberParser
        """
        self.pdf_parser = parser if parser is not None else PDFMinerParser()
        self.collection_manager = QdrantClientManager()
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding_model = BGEM3FlagModel(
            "BAAI/bge-m3", use_fp16=True
        )  # Using BGE-M3 with FP16 for better performance
        self.bm25 = None
        self.vocabulary = None
        self.tokenized_corpus = []

        if not self.collection_manager.collection_exists(collection_name):
            logger.info(
                f"Creating collection {collection_name} with vector size {vector_size}"
            )
            self.collection_manager.create_collection(collection_name, vector_size)
        else:
            logger.info(f"Using existing collection {collection_name}")
            # Try to load vocabulary from existing collection
            self._load_vocabulary_from_collection()

    def _load_vocabulary_from_collection(self) -> None:
        """Load vocabulary from existing collection if available."""
        try:
            # Get a sample point from the collection to extract vocabulary size
            points = self.collection_manager.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=False,
                with_vectors=True,
            )[0]

            if points:
                # Get the sparse vector size from the first point
                sparse_vector = points[0].vector.get(
                    self.collection_manager.SPARSE_VECTOR_NAME, []
                )
                if sparse_vector:
                    # Create a dummy vocabulary of the same size
                    self.vocabulary = [""] * len(sparse_vector)
                    logger.info(
                        f"Loaded vocabulary size from collection: {len(self.vocabulary)}"
                    )
        except Exception as e:
            logger.warning(f"Could not load vocabulary from collection: {str(e)}")

    def build_bm25_index(self, documents: List[str]) -> None:
        """Builds a BM25 index and vocabulary from the documents."""
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.vocabulary = self._get_vocabulary()

    def _get_vocabulary(
        self, max_vocab_size: int = QdrantClientManager.SPARSE_VECTOR_SIZE
    ) -> List[str]:
        """Builds a vocabulary from the tokenized corpus."""
        all_tokens = [token for doc in self.tokenized_corpus for token in doc]
        most_common = Counter(all_tokens).most_common(max_vocab_size)
        return [token for token, _ in most_common]

    def process_and_store(self, samples: int = 0) -> None:
        """Process PDFs and store the extracted text in Qdrant with hybrid vectors."""
        if samples > 0:
            df = pd.read_csv("data/filtered_motions.csv").head(samples)
        else:
            df = pd.read_csv("data/filtered_motions.csv")

        self.tokenized_corpus = []
        self.bm25 = None
        expected_vocab_size = QdrantClientManager.SPARSE_VECTOR_SIZE

        processed_documents = []
        for _, row in df.iterrows():
            try:
                result = self.pdf_parser.process_url(
                    row["dokument_url"], str(row["geschaeft_uid"])
                )
                if result["status"] == "success" and result["text"]:
                    cleaned_text = self.pdf_parser.text_cleaner.clean_text(
                        result["text"]
                    )
                    processed_documents.append({"text": cleaned_text, "row": row})
            except Exception as e:
                logger.error(
                    f"Error processing document {row['dokument_uid']}: {str(e)}"
                )
                continue

        if processed_documents:
            self.tokenized_corpus = [
                doc["text"].lower().split() for doc in processed_documents
            ]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self.vocabulary = self._get_vocabulary()
            if len(self.vocabulary) < expected_vocab_size:
                self.vocabulary.extend(
                    [""] * (expected_vocab_size - len(self.vocabulary))
                )
        else:
            logger.warning(
                "No documents were successfully processed for BM25 initialization"
            )
            self.vocabulary = [""] * expected_vocab_size

        for doc in processed_documents:
            try:
                cleaned_text = doc["text"]
                row = doc["row"]

                # Get dense embedding using BGE-M3
                dense_embedding = self.embedding_model.encode(cleaned_text)[
                    "dense_vecs"
                ]

                # Create sparse vector
                sparse_vec = np.zeros(expected_vocab_size)
                if self.bm25 is not None:
                    doc_tokens = cleaned_text.lower().split()
                    for idx, token in enumerate(self.vocabulary):
                        if token and token in doc_tokens:
                            sparse_vec[idx] = self.bm25.idf.get(token, 0.0)

                title = None
                beschluss = None
                if "Deutsch" in row["dokument_ziel_publikation"]:
                    title = row["geschaeft_titel_deutsch"]
                    beschluss = row["geschaeft_beschluss_deutsch"]
                elif row["dokument_ziel_publikation"] == "Französisch":
                    title = row["geschaeft_titel_franz"]
                    beschluss = row["geschaeft_beschluss_franz"]

                metadata = {
                    "geschaeft_uid": str(row["geschaeft_uid"]),
                    "dokument_uid": str(row["dokument_uid"]),
                    "url": row["dokument_url"],
                    "title": title,
                    "beschluss": beschluss,
                    "text": cleaned_text,
                }

                point_id = str(uuid.uuid4())
                self.collection_manager.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector={
                                self.collection_manager.TEXT_VECTOR_NAME: dense_embedding.tolist(),
                                self.collection_manager.MATRYOSHKA_64: dense_embedding[
                                    :64
                                ].tolist(),
                                self.collection_manager.MATRYOSHKA_128: dense_embedding[
                                    :128
                                ].tolist(),
                                self.collection_manager.MATRYOSHKA_256: dense_embedding[
                                    :256
                                ].tolist(),
                                self.collection_manager.SPARSE_VECTOR_NAME: sparse_vec.tolist(),
                            },
                            payload=metadata,
                        )
                    ],
                )

                logger.info(
                    f"Successfully processed and stored document {row['dokument_uid']}"
                )

            except Exception as e:
                logger.error(f"Error storing document {row['dokument_uid']}: {str(e)}")
                continue

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using hybrid search.

        Args:
            query (str): The search query
            limit (int): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata
        """
        # Get dense embedding using BGE-M3
        query_embedding = self.embedding_model.encode(query)["dense_vecs"]

        # Handle sparse vector creation
        if self.vocabulary is None:
            # If vocabulary is not available, create a dummy sparse vector
            sparse_query = [0.0] * self.collection_manager.SPARSE_VECTOR_SIZE
        else:
            query_tokens = query.lower().split()
            sparse_query = np.zeros(len(self.vocabulary))
            for idx, token in enumerate(self.vocabulary):
                if token in query_tokens:
                    sparse_query[idx] = (
                        1.0  # Use simple binary weights when BM25 is not available
                    )

        search_results = self.collection_manager.hybrid_search(
            collection_name=self.collection_name,
            query_embedding=query_embedding.tolist(),
            sparse_query=sparse_query.tolist(),
            limit=limit,
        )

        results = []
        for hit in search_results:
            results.append({"score": hit.score, "metadata": hit.payload})

        return results
