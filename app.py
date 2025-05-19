from qdrant_client_manager import QdrantClientManager
from data_pipeline import DataPipeline
from pdfplumber_parser import PDFPlumberParser
from pypdf2_parser import PyPDF2Parser
from pdfminer_parser import PDFMinerParser
import logging
import traceback
from typing import Union


def get_parser(parser_type: str):
    parser_map = {
        "pdfplumber": PDFPlumberParser(),
        "pypdf2": PyPDF2Parser(),
        "pdfminer": PDFMinerParser(),
    }
    return parser_map.get(parser_type.lower())


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    collection_name: str,
    delete_collection: bool,
    process: bool,
    search: bool,
    search_limit: int,
    process_samples: Union[int, str],
    query: str,
    parser_type: str,
):
    manager = QdrantClientManager()
    # if delete_collection and manager.collection_exists(collection_name):
    #     logger.info(f"Deleting existing collection: {collection_name}")
    #     manager.delete_collection(collection_name)

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

        if search:
            query = query
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
    query1 = "Erhalt und Förderung einer vielfältigen Tier- und Pflanzenwelt?"
    query2 = "Zugang zu Erwerbstätigkeit für Personen im Asylverfahren?"
    query3 = "Vorkehrungen zur barrierefreien Gestaltung von Einrichtungen?"
    for query in [query1, query2, query3]:
        print("\n")
        main(
            collection_name="motions2",
            delete_collection=False,
            process=False,
            process_samples="all",
            search=True,
            search_limit=6,
            query=query,
            parser_type="pdfminer",
        )
