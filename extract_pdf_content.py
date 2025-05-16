import pandas as pd
import requests
from PyPDF2 import PdfReader
from io import BytesIO
import time
from pathlib import Path
import logging
from typing import Optional, Dict, List
import concurrent.futures
from dataclasses import dataclass

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PDFExtractor:
    """Class to handle PDF extraction operations."""

    output_dir: Path

    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_pdf(self, url: str) -> Optional[BytesIO]:
        """Download PDF from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_text(self, pdf_content: BytesIO) -> Optional[str]:
        """Extract text from PDF content."""
        try:
            pdf_reader = PdfReader(pdf_content)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return None

    def process_url(self, url: str, motion_id: str) -> Dict:
        """Process a single PDF URL."""
        pdf_content = self.download_pdf(url)
        if not pdf_content:
            return {"motion_id": motion_id, "status": "failed", "text": None}

        text = self.extract_text(pdf_content)
        if not text:
            return {"motion_id": motion_id, "status": "failed", "text": None}

        # Save extracted text to file
        output_file = self.output_dir / f"{motion_id}.txt"
        output_file.write_text(text, encoding="utf-8")

        return {"motion_id": motion_id, "status": "success", "text": text}


def main():
    # Create output directory
    output_dir = Path("extracted_texts")
    extractor = PDFExtractor(output_dir)

    # Read the CSV file and take only one sample
    df = pd.read_csv("data/filtered_motions.csv").head(1)
    logger.info(f"Testing with one sample: {df['geschaeft_uid'].iloc[0]}")

    # Process URLs in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(
                extractor.process_url, row["dokument_url"], str(row["geschaeft_uid"])
            ): row["geschaeft_uid"]
            for _, row in df.iterrows()
        }

        for future in concurrent.futures.as_completed(future_to_url):
            motion_id = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(f"Processed motion {motion_id}: {result['status']}")
            except Exception as e:
                logger.error(f"Error processing motion {motion_id}: {str(e)}")
                results.append(
                    {"motion_id": motion_id, "status": "failed", "text": None}
                )

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/pdf_extraction_results.csv", index=False)
    logger.info(f"Processing complete. Results saved to pdf_extraction_results.csv")


if __name__ == "__main__":
    main()
