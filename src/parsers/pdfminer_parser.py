import concurrent.futures
import logging
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import requests
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.pdfparser import PDFSyntaxError

from src.pipeline.text_cleaner import SimpleTextCleaner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PDFMinerParser:
    """Class to handle PDF extraction operations using PDFMiner.six."""

    pdf_dir: Path = Path("data/parsers/pdfminer/pdfs")
    extracted_dir: Path = Path("data/parsers/pdfminer/extracted")
    cleaned_dir: Path = Path("data/parsers/pdfminer/cleaned")
    duplicates_dir: Path = Path("data/parsers/pdfminer/duplicates")
    text_cleaner: SimpleTextCleaner = SimpleTextCleaner()

    def __post_init__(self):
        """Create output directories if they don't exist."""
        if isinstance(self.pdf_dir, str):
            self.pdf_dir = Path(self.pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.extracted_dir, str):
            self.extracted_dir = Path(self.extracted_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.cleaned_dir, str):
            self.cleaned_dir = Path(self.cleaned_dir)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.duplicates_dir, str):
            self.duplicates_dir = Path(self.duplicates_dir)
        self.duplicates_dir.mkdir(parents=True, exist_ok=True)

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
            pdf_content.seek(0)
            text = pdfminer_extract_text(pdf_content)
            return text if text and text.strip() else None
        except PDFSyntaxError as e:
            logger.error(f"PDF syntax error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def process_url(self, url: str, motion_id: str) -> Dict:
        """Process a single PDF URL.
        Returns a dictionary with status, text, url, pdf_content_bytes, and error_message.
        """
        pdf_content_io = self.download_pdf(url)
        if not pdf_content_io:
            return {
                "motion_id": motion_id,
                "status": "failed",
                "text": None,
                "url": url,
                "pdf_content_bytes": None,
                "error_message": "Download failed",
            }

        pdf_bytes = pdf_content_io.getvalue()
        pdf_content_io.seek(0)

        text = self.extract_text(pdf_content_io)
        if not text:
            return {
                "motion_id": motion_id,
                "status": "failed",
                "text": None,
                "url": url,
                "pdf_content_bytes": pdf_bytes,
                "error_message": "Text extraction failed or text is empty",
            }

        return {
            "motion_id": motion_id,
            "status": "success",
            "text": text,
            "url": url,
            "pdf_content_bytes": pdf_bytes,
        }

    def save_pdf(
        self, content: BytesIO, filename: str, target_dir: Optional[Path] = None
    ) -> bool:
        """Save PDF content to a file in the specified directory.

        Args:
            content (BytesIO): The PDF content to save
            filename (str): The name of the file to save (without extension)
            target_dir (Optional[Path]): The directory to save the PDF. Defaults to self.pdf_dir.

        Returns:
            bool: True if save was successful, False otherwise
        """
        save_to_dir = target_dir if target_dir is not None else self.pdf_dir
        output_file = save_to_dir / f"{filename}.pdf"
        try:
            output_file.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure target subdir exists
            with open(output_file, "wb") as f:
                f.write(content.getvalue())
            logger.info(f"Successfully saved PDF to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving PDF {filename} to {output_file}: {str(e)}")
            return False

    def save_content(self, results: List[Any]):
        """Save extraction results to CSV file."""
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/pdf_extraction_results.csv", index=False)
        logger.info(f"Processing complete. Results saved to pdf_extraction_results.csv")

    def clean_extracted_texts(
        self,
    ) -> None:
        """Clean extracted text files using NLP processing pipeline.

        Args:
            input_dir (str): Directory containing the original text files
            output_dir (str): Directory where cleaned files will be saved
        """
        try:
            input_path = Path(self.extracted_dir)
            output_path = Path(self.cleaned_dir)

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

                    # Ensure content is not None or empty before cleaning
                    if content and content.strip():
                        cleaned_content = self.text_cleaner.clean_text(content)

                        output_file = output_path / file_path.name
                        with open(output_file, "w", encoding="utf-8") as out_file:
                            out_file.write(cleaned_content)
                        cleaned_files += 1
                        logger.info(f"Cleaned file: {file_path.name}")
                    else:
                        logger.warning(
                            f"Skipping empty or whitespace-only file: {file_path.name}"
                        )

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")

            logger.info(
                f"Processing complete. Processed {processed_files} files, cleaned {cleaned_files} files."
            )

        except Exception as e:
            logger.error(f"Error in cleanup pipeline: {str(e)}")

    def process_test(
        self,
        samples: int = 0,
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
        seen_texts_content: Set[str] = set()
        motion_id_to_url_map = {
            str(row["geschaeft_uid"]): row["dokument_url"] for _, row in df.iterrows()
        }

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {
                executor.submit(
                    self.process_url,
                    row["dokument_url"],
                    str(row["geschaeft_uid"]),
                ): row["geschaeft_uid"]
                for _, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_url):
                motion_id = future_to_url[future]
                try:
                    result = (
                        future.result()
                    )  # Contains status, text, url, pdf_content_bytes, error_message

                    original_text = result.get("text")
                    pdf_bytes_for_saving = result.get("pdf_content_bytes")

                    if result["status"] == "success" and original_text is not None:
                        cleaned_text_for_dedup = self.text_cleaner.clean_text(
                            original_text
                        )

                        if cleaned_text_for_dedup in seen_texts_content:
                            logger.info(
                                f"Duplicate content detected for motion {motion_id} (URL: {result['url']}) after cleaning. Saving original to duplicates directory."
                            )
                            result["status"] = (
                                "duplicate"  # Update status for results list
                            )

                            if pdf_bytes_for_saving:
                                self.save_pdf(
                                    BytesIO(pdf_bytes_for_saving),
                                    str(motion_id),
                                    target_dir=self.duplicates_dir,
                                )
                                # Save the ORIGINAL text for duplicates
                                self.save_text(
                                    original_text,
                                    str(motion_id),
                                    target_dir=self.duplicates_dir,
                                )
                        else:  # New, unique text (based on cleaned version)
                            seen_texts_content.add(cleaned_text_for_dedup)
                            logger.info(
                                f"Successfully processed motion {motion_id} (URL: {result['url']}). Cleaned text unique."
                            )
                            if pdf_bytes_for_saving:
                                self.save_pdf(
                                    BytesIO(pdf_bytes_for_saving), str(motion_id)
                                )  # Saves to self.pdf_dir
                                # Save the ORIGINAL text for unique entries to extracted_dir
                                self.save_text(
                                    original_text, str(motion_id)
                                )  # Saves to self.extracted_dir

                    elif result["status"] == "failed":
                        logger.error(
                            f"Failed to process motion {motion_id} (URL: {result['url']}). Reason: {result.get('error_message', 'N/A')}"
                        )
                        # Optional: Save PDFs that failed text extraction to a specific directory
                        # if pdf_bytes_for_saving:
                        #     failed_extraction_dir = self.duplicates_dir / "failed_extraction"
                        #     failed_extraction_dir.mkdir(parents=True, exist_ok=True)
                        #     self.save_pdf(BytesIO(pdf_bytes_for_saving), str(motion_id), target_dir=failed_extraction_dir)

                    # Append the original result which includes original_text
                    results.append(result)

                except Exception as e:
                    url_for_error = motion_id_to_url_map.get(
                        str(motion_id), "URL not found for motion_id"
                    )
                    logger.error(
                        f"Error processing future for motion {motion_id} (URL: {url_for_error}): {str(e)}"
                    )
                    results.append(
                        {
                            "motion_id": motion_id,
                            "status": "error_in_future",
                            "text": None,
                            "url": url_for_error,
                            "error_message": str(e),
                        }
                    )

    def save_text(self, text: str, filename: str, target_dir: Optional[Path] = None):
        """Save text to a file in the specified directory."""
        save_to_dir = target_dir if target_dir is not None else self.extracted_dir
        output_file = save_to_dir / f"{filename}.txt"
        try:
            output_file.parent.mkdir(
                parents=True, exist_ok=True
            )  # Ensure target subdir exists
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(f"Successfully saved text to {output_file}")
        except Exception as e:
            logger.error(f"Error saving text {filename} to {output_file}: {str(e)}")


def main():
    parser = PDFMinerParser()
    parser.process_test(samples=5)


if __name__ == "__main__":
    main()
