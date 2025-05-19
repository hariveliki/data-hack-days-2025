import os
import pandas as pd
import requests
from io import BytesIO
import time
from pathlib import Path
import logging
from typing import Optional, Dict, List, Any
import concurrent.futures
from dataclasses import dataclass
import re
from typing import Set
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfparser import PDFSyntaxError
from src.pipeline.text_cleaner import AdvancedTextCleaner

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PDFMinerParser:
    """Class to handle PDF extraction operations using PDFMiner.six."""

    output_dir: Path = "data"
    text_cleaner: AdvancedTextCleaner = AdvancedTextCleaner()

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, url: str) -> Optional[BytesIO]:
        """Download PDF from URL.

        Args:
            url (str): URL of the PDF to download

        Returns:
            Optional[BytesIO]: PDF content as BytesIO object if successful, None otherwise
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_text(self, pdf_content: BytesIO) -> Optional[str]:
        """Extract text from PDF content using PDFMiner.six."""
        try:
            text = pdfminer_extract_text(pdf_content)
            return text if text.strip() else None
        except PDFSyntaxError as e:
            logger.error(f"PDF syntax error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def process_url(self, url: str, motion_id: str) -> Dict:
        """Process a single PDF URL."""
        pdf_content = self.download_pdf(url)
        if not pdf_content:
            return {
                "motion_id": motion_id,
                "status": "failed",
                "text": None,
                "url": url,
            }

        text = self.extract_text(pdf_content)
        if not text:
            return {
                "motion_id": motion_id,
                "status": "failed",
                "text": None,
                "url": url,
            }

        return {"motion_id": motion_id, "status": "success", "text": text, "url": url}

    def save_pdf(self, content: BytesIO, filename: str) -> bool:
        """Save PDF content to a file.

        Args:
            content (BytesIO): The PDF content to save
            filename (str): The name of the file to save (without extension)

        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            output_file = self.output_dir / f"{filename}.pdf"
            with open(output_file, "wb") as f:
                f.write(content.getvalue())
            logger.info(f"Successfully saved PDF to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving PDF {filename}: {str(e)}")
            return False

    def save_content(self, results: List[Any]):
        """Save extraction results to CSV file."""
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/pdf_extraction_results.csv", index=False)
        logger.info(f"Processing complete. Results saved to pdf_extraction_results.csv")

    def clean_extracted_texts(
        self,
        input_dir: str = "extracted_texts",
        output_dir: str = "extracted_texts_cleaned",
    ) -> None:
        """Clean extracted text files using NLP processing pipeline.

        Args:
            input_dir (str): Directory containing the original text files
            output_dir (str): Directory where cleaned files will be saved
        """
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)

            # Create output directory if it doesn't exist
            output_path.mkdir(parents=True, exist_ok=True)

            if not input_path.exists():
                logger.error(f"Input directory {input_path} does not exist")
                return

            processed_files = 0
            cleaned_files = 0

            for file_path in input_path.glob("*.txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                        processed_files += 1

                        # Apply text cleaning to the entire content
                        cleaned_content = self.text_cleaner.clean_text(content)

                        # Write cleaned content to output file
                        output_file = output_path / file_path.name
                        with open(output_file, "w", encoding="utf-8") as out_file:
                            out_file.write(cleaned_content)

                        cleaned_files += 1
                        logger.info(f"Cleaned file: {file_path.name}")

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

            logger.info(
                f"Processing complete. Processed {processed_files} files, cleaned {cleaned_files} files."
            )

        except Exception as e:
            logger.error(f"Error in cleanup pipeline: {str(e)}")

    def process_pdfs(
        self,
        extractor,
        samples: int = 0,
        save_pdf: bool = False,
    ):
        """Process PDFs from the filtered motions CSV file.

        Args:
            samples (int): Number of samples to process. If 0, process all.
            save_pdf (bool): Whether to save the original PDF files.
        """
        if samples > 0:
            df = pd.read_csv("data/filtered_motions.csv").head(samples)
        else:
            df = pd.read_csv("data/filtered_motions.csv")

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(
                    extractor.process_url,
                    row["dokument_url"],
                    str(row["geschaeft_uid"]),
                ): row["geschaeft_uid"]
                for _, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_url):
                motion_id = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Processed motion {motion_id}: {result['status']}")

                    if save_pdf and result["status"] == "success":
                        pdf_content = extractor.download_pdf(result["url"])
                        if pdf_content:
                            extractor.save_pdf(pdf_content, str(motion_id))

                except Exception as e:
                    logger.error(f"Error processing motion {motion_id}: {str(e)}")
                    results.append(
                        {"motion_id": motion_id, "status": "failed", "text": None}
                    )


if __name__ == "__main__":
    parser = PDFMinerParser()
    parser.clean_extracted_texts()
