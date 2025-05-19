import os
from pathlib import Path
import logging
from typing import Optional, Dict, List, Protocol
import pandas as pd
from abc import ABC, abstractmethod
import re

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextCleaner(Protocol):
    """Protocol defining the interface for text cleaners."""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text
        """
        ...


class SimpleTextCleaner:
    """Minimal text cleaner that only removes line breaks and extra spaces, placing all text on one line."""

    def clean_text(self, text: str) -> str:
        """Clean text by removing line breaks and extra spaces only.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text on a single line
        """
        if not text:
            return ""

        # Replace all line breaks with a single space
        text = re.sub(r"\n+", " ", text)

        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        return text.strip()


class BasicTextCleaner:
    """Basic text cleaner implementation."""

    def clean_text(self, text: str) -> str:
        """Clean and normalize text by handling whitespace and newlines.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text with normalized whitespace
        """
        # Replace multiple newlines with a single space
        text = re.sub(r"\n+", " ", text)

        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        return text.strip()


class AdvancedTextCleaner:
    """Advanced text cleaner implementation with additional cleaning rules."""

    def _remove_specific_patterns(self, text: str) -> str:
        """Remove specific patterns commonly found in parliamentary documents.

        Args:
            text (str): Input text to clean

        Returns:
            str: Text with specific patterns removed
        """
        patterns = [
            r"Page \d+ / \d+",  # Page numbers
            r"Seite \d+ von \d+",  # German page numbers
            r"\d{2}\.\d{2}\.\d{4}",  # Dates in format DD.MM.YYYY
            r"\d{4}\.RRGR\.\d+",  # RRGR numbers
            r"\d{3}-\d{4}",  # Numbers like 049-2022
            r"RRB-Nr\.:?",
            r"Direktion:?",
            r"Klassifizierung:?",
            r"Antrag Regierungsrat:?",
            r"Weitere Unterschriften:?",
            r"Eingereicht am:?",
            r"Geschäftsnummer:?",
            r"Richtlinienmotion:?",
            r"Vorstossart:?",
            r"Vorstoss-Nr\.:?",
            r"https?://\S+",  # URLs starting with http:// or https://
            r"www\.\S+",  # URLs starting with www.
            r"Fraktionsvorstoss:?",
            r"Kommissionsvorstoss:?",
            r"Eingereicht von:?",
            r"Dringlichkeit verlangt:?",
            r"Dringlichkeit gewährt:?",
            r"Nicht klassifiziert",
            r"Auswahl",
            r"Letzte Bearbeitung: \d{2}\.\d{2}\.\d{4}",  # Last modification date
            r"Version: \d+",  # Version number
            r"Dok\.-Nr\.: \d+",  # Document number
            r"Geschäftsnummer: \d{4}\.RRGR\.\d+",  # Business number
            r"Parlamentarischer Vorstoss",
            r"Motion",
            r"Nein",
            r"\bEVP\b",
            r"\bSP\b",
            r"\bGLP\b",
            r"\bCVP\b",
            r"\bEDU\b",
            r"\bFDP\b",
            r"\bSVP\b",
            r"Letzte Bearbeitung",
            r"Verteiler",
            r"Grosser Rat",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        return text

    def clean_text(self, text: str) -> str:
        """Clean and normalize text with advanced rules.

        Args:
            text (str): Input text to clean

        Returns:
            str: Cleaned text with advanced normalization
        """
        # Basic cleaning
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # Remove specific patterns
        text = self._remove_specific_patterns(text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\s.,!?-]", "", text)

        # Normalize whitespace around punctuation
        text = re.sub(r"\s+([.,!?])", r"\1", text)

        return text


class TextCleanerFactory:
    """Factory class to create text cleaners based on the cleaning level."""

    def __init__(self, cleaner_type: str = "basic"):
        self.cleaner_type = cleaner_type

    def get_cleaner(self) -> TextCleaner:
        if self.cleaner_type == "simple":
            return SimpleTextCleaner()
        elif self.cleaner_type == "basic":
            return BasicTextCleaner()
        elif self.cleaner_type == "advanced":
            return AdvancedTextCleaner()
        else:
            raise ValueError(f"Invalid cleaner type: {self.cleaner_type}")


class TextCleanerTester:
    """Class to handle text cleaning operations with configurable cleaner."""

    def __init__(self, input_dir: Path, output_dir: Path, cleaner: TextCleaner):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cleaner = cleaner
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_text(self, text: str, filename: str) -> Dict:
        """Process a single text file."""
        try:
            cleaned_text = self.cleaner.clean_text(text)
            if not cleaned_text:
                return {"filename": filename, "status": "failed", "text": None}

            # Save cleaned text
            output_file = self.output_dir / f"{filename}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            return {"filename": filename, "status": "success", "text": cleaned_text}
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return {"filename": filename, "status": "failed", "text": None}

    def process_texts(self, samples: int = 0) -> None:
        """Process text files from the input directory."""
        text_files = list(self.input_dir.glob("*.txt"))

        if samples > 0:
            text_files = text_files[:samples]

        results = []
        for text_path in text_files:
            logger.info(f"Processing {text_path.name}...")
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read()
            result = self.process_text(text, text_path.stem)
            results.append(result)
            logger.info(f"Status: {result['status']}")

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / "cleaning_results.csv", index=False)
        logger.info(f"Processing complete. Results saved to cleaning_results.csv")


def main():
    # Use the extracted text directory from test_pdf_extraction.py
    input_dir = Path("data/pdfs/extracted")
    output_dir = Path("data/pdfs/cleaned")

    cleaner = TextCleanerFactory("advanced").get_cleaner()

    tester = TextCleanerTester(input_dir, output_dir, cleaner)
    logger.info("Starting text cleaning test with extracted PDF texts...")
    tester.process_texts()  # Process all extracted texts
    logger.info("Text cleaning test completed. Check the results in data/pdfs/cleaned")


if __name__ == "__main__":
    main()
