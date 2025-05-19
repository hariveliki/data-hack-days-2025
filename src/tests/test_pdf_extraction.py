import os
from pathlib import Path
import logging
from typing import Optional, Dict, List, Protocol
import pandas as pd
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFParser(Protocol):
    """Protocol defining the interface for PDF parsers."""

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file.

        Args:
            pdf_path (Path): Path to the PDF file

        Returns:
            Optional[str]: Extracted text if successful, None otherwise
        """
        ...


class PDFPlumberParser:
    """PDF parser implementation using pdfplumber."""

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(
                f"Error extracting text with pdfplumber from {pdf_path}: {str(e)}"
            )
            return None


class PyPDF2Parser:
    """PDF parser implementation using PyPDF2."""

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        try:
            import PyPDF2

            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2 from {pdf_path}: {str(e)}")
            return None


class PDFMinerParser:
    """PDF parser implementation using PDFMiner."""

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        try:
            from pdfminer.high_level import extract_text

            text = extract_text(pdf_path)
            return text
        except Exception as e:
            logger.error(
                f"Error extracting text with PDFMiner from {pdf_path}: {str(e)}"
            )
            return None


class PDFParserFactory:
    """Factory class to create PDF parsers based on the input directory."""

    def __init__(self, input_dir: Path, parser_name: str):
        self.input_dir = input_dir
        self.parser_name = parser_name

    def get_parser(self) -> PDFParser:
        if self.parser_name == "pdfplumber":
            return PDFPlumberParser()
        elif self.parser_name == "pypdf2":
            return PyPDF2Parser()
        elif self.parser_name == "pdfminer":
            return PDFMinerParser()
        else:
            raise ValueError(f"Invalid parser name: {self.parser_name}")


class PDFExtractor:
    """Class to handle PDF extraction operations with configurable parser."""

    def __init__(self, input_dir: Path, output_dir: Path, parser: PDFParser):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.parser = parser
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: Path) -> Dict:
        """Process a single PDF file."""
        try:
            text = self.parser.extract_text(pdf_path)
            if not text:
                return {"pdf_name": pdf_path.name, "status": "failed", "text": None}

            # Save extracted text
            output_file = self.output_dir / f"{pdf_path.stem}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            return {"pdf_name": pdf_path.name, "status": "success", "text": text}
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"pdf_name": pdf_path.name, "status": "failed", "text": None}

    def process_pdfs(self, samples: int = 0) -> None:
        """Process PDFs from the input directory."""
        pdf_files = list(self.input_dir.glob("*.pdf"))

        if samples > 0:
            pdf_files = pdf_files[:samples]

        results = []
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}...")
            result = self.process_pdf(pdf_path)
            results.append(result)
            logger.info(f"Status: {result['status']}")

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "extraction_results.csv", index=False)
        logger.info(f"Processing complete. Results saved to extraction_results.csv")


def main():
    input_dir = Path("data/pdfs")
    output_dir = Path("data/pdfs/extracted")

    parser = PDFParserFactory(input_dir, "pdfminer").get_parser()

    extractor = PDFExtractor(input_dir, output_dir, parser)
    logger.info("Starting PDF extraction test with 5 samples...")
    extractor.process_pdfs(samples=5)
    logger.info(
        "PDF extraction test completed. Check the results in data/pdfs/extracted"
    )


if __name__ == "__main__":
    main()
