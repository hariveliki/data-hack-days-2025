import argparse
import logging
import traceback
from typing import Union

from src.db.qdrant_client_manager import QdrantClientManager
from src.pipeline.data_pipeline import DataPipeline
from src.parsers.pdfplumber_parser import PDFPlumberParser
from src.parsers.pypdf2_parser import PyPDF2Parser
from src.parsers.pdfminer_parser import PDFMinerParser


def get_parser(parser_type: str):
    parser_map = {
        "pdfplumber": PDFPlumberParser(),
        "pypdf2": PyPDF2Parser(),
        "pdfminer": PDFMinerParser(),
    }
    return parser_map.get(parser_type.lower())


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    query: str = None,
    search_limit: int = 10,
    process: bool = False,
    process_samples: Union[int, str] = "all",
    parser_type: str = "pdfminer",
    collection_name: str = "test",
    delete_collection: bool = False,
):
    manager = QdrantClientManager()
    if delete_collection and manager.collection_exists(collection_name):
        logger.info(f"Deleting existing collection: {collection_name}")
        manager.delete_collection(collection_name)

    parser = get_parser(parser_type)
    if parser is None:
        logger.warning(
            f"Unknown parser type: {parser_type}. Using default PDFPlumberParser."
        )
        parser = PDFPlumberParser()

    pipeline = DataPipeline(collection_name=collection_name, parser=parser)

    try:
        if process:
            logger.info("Starting document processing and storage...")
            if process_samples == "all":
                pipeline.process_and_store()
            else:
                pipeline.process_and_store(process_samples)
            logger.info("Document processing and storage completed.")

        if query:
            logger.info(f"Performing hybrid search for: {query}")
            results = pipeline.search_similar(query, limit=search_limit)

            print(f"\nHybrid Search Results for query {query}")
            for i, result in enumerate(results, 1):
                # print(f"\n{i}. Geschaeft ID: {result['metadata']['geschaeft_uid']}")
                # print(f"   Dokument ID: {result['metadata']['dokument_uid']}")
                # print(f"   Hybrid Search Score: {result['score']:.4f}")
                print(f"\n{i}. Title: {result['metadata']['title']}")
                print(f"   Beschluss: {result['metadata']['beschluss']}")
                print(f"   URL: {result['metadata']['url']}")
                # print(f"   Text Preview: {result['metadata']['text'][:500]}...")

    except Exception as e:
        # add traceback
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        print(
            "Please ensure Qdrant is running. You can start it with: docker-compose up -d"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script.")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query to perform hybrid search on",
    )
    parser.add_argument(
        "--parser_type",
        type=str,
        default="pdfminer",
        help="Parser type to use",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="test",
        help="Collection name to use",
    )
    parser.add_argument(
        "--delete_collection",
        type=bool,
        default=False,
        help="Delete collection if it exists",
    )
    parser.add_argument(
        "--process",
        type=bool,
        default=False,
        help="Process and store documents",
    )
    parser.add_argument(
        "--process_samples",
        type=str,
        default="all",
        help="Process samples to use",
    )
    parser.add_argument(
        "--search_limit",
        type=int,
        default=10,
        help="Search limit",
    )
    args = parser.parse_args()
    main(
        collection_name=args.collection_name,
        delete_collection=args.delete_collection,
        process=args.process,
        process_samples=args.process_samples,
        search_limit=args.search_limit,
        query=args.query,
        parser_type=args.parser_type,
    )
