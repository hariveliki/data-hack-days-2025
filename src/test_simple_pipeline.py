import logging
from pathlib import Path
from src.parsers.pdfminer_parser import PDFMinerParser
from src.pipeline.text_cleaner import SimpleTextCleaner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_simple_cleaner():
    """Test the simple text cleaner directly."""
    cleaner = SimpleTextCleaner()

    # Test with multiline text
    test_text = """This is a test
    with multiple
    
    lines and     extra    spaces.
    
    Let's see how it works."""

    cleaned = cleaner.clean_text(test_text)
    logger.info(f"Original text length: {len(test_text)}")
    logger.info(f"Cleaned text length: {len(cleaned)}")
    logger.info(f"Cleaned text: {cleaned}")

    # Check if the result is on a single line
    assert "\n" not in cleaned, "Text should not contain line breaks"

    # Check if multiple spaces are removed
    assert "  " not in cleaned, "Text should not contain multiple spaces"

    logger.info("Simple cleaner test passed!")


def test_pdf_parser_with_simple_cleaner():
    """Test the PDFMinerParser with SimpleTextCleaner."""
    # Initialize parser with test directories
    test_dir = Path("test_data")
    parser = PDFMinerParser(
        pdf_dir=test_dir / "pdfs",
        extracted_dir=test_dir / "extracted",
        cleaned_dir=test_dir / "cleaned",
        duplicates_dir=test_dir / "duplicates",
    )

    # Ensure we're using SimpleTextCleaner
    assert isinstance(
        parser.text_cleaner, SimpleTextCleaner
    ), "Parser should use SimpleTextCleaner"

    # Process a small number of samples
    logger.info("Processing sample PDFs...")
    parser.process_test(samples=5)

    # Clean the extracted texts
    logger.info("Cleaning extracted texts...")
    parser.clean_extracted_texts()

    logger.info("PDF parser test with simple cleaner completed!")


if __name__ == "__main__":
    logger.info("Testing simple pipeline...")
    test_simple_cleaner()
    test_pdf_parser_with_simple_cleaner()
    logger.info("All tests completed!")
