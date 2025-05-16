import layoutparser as lp
import pdf2image
import os
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from PIL import Image


class LayoutParser:
    def __init__(self, model_name: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x"):
        """
        Initialize the LayoutParser with a specified model name.

        Args:
            model_name (str): The name of the layout analysis model to use
        """
        self.model_name = model_name
        self._model: Optional[lp.Detectron2LayoutModel] = None
        print(f"Initializing LayoutParser with model: {model_name}")

        # Define color map for different element types
        self.color_map = {
            0: "red",  # Text
            1: "blue",  # Title
            2: "green",  # List
            3: "purple",  # Table
            4: "orange",  # Figure
            5: "brown",  # Unknown
        }

    @property
    def model(self) -> lp.Detectron2LayoutModel:
        """
        Lazy loading of the model.

        Returns:
            lp.Detectron2LayoutModel: The loaded layout analysis model
        """
        if self._model is None:
            print("Attempting to load model...")
            try:
                # Using Detectron2LayoutModel directly instead of AutoLayoutModel
                self._model = lp.Detectron2LayoutModel(self.model_name)
                print(f"Model '{self.model_name}' loaded successfully")

            except Exception as e:
                print(f"Error loading model '{self.model_name}': {str(e)}")
                raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")

        return self._model

    def load_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path (str): Path to the PDF file

        Returns:
            List[np.ndarray]: List of page images as numpy arrays
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            print(f"Converting PDF to images: {pdf_path}")
            images = pdf2image.convert_from_path(pdf_path)
            print(f"Successfully converted PDF to {len(images)} images")
            return images
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")

    def extract_layout(self, image: np.ndarray) -> lp.Layout:
        """
        Extract layout from a single page image.

        Args:
            image (np.ndarray): Page image as numpy array

        Returns:
            lp.Layout: Detected layout elements
        """
        try:
            print("Extracting layout from image...")
            layout = self.model.detect(image)
            print(f"Successfully extracted layout with {len(layout)} elements")
            return layout
        except Exception as e:
            print(f"Error extracting layout: {str(e)}")
            raise RuntimeError(f"Failed to extract layout: {str(e)}")

    def visualize_layout(
        self,
        image: np.ndarray,
        layout: lp.Layout,
        output_path: str = None,
        min_confidence: float = 0.5,
    ) -> Figure:
        """
        Visualize the detected layout on the image.

        Args:
            image (np.ndarray): The original page image
            layout (lp.Layout): The detected layout
            output_path (str, optional): Path to save the visualization. If None, just displays it.
            min_confidence (float, optional): Minimum confidence threshold for showing elements

        Returns:
            Figure: The matplotlib figure with visualization
        """
        # Filter layout elements by confidence
        filtered_layout = [block for block in layout if block.score >= min_confidence]

        # Create figure and axis
        height, width, _ = image.shape
        fig_width = 12
        fig_height = fig_width * height / width

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.imshow(image)

        # Add boxes for each layout element
        for block in filtered_layout:
            x1, y1, x2, y2 = block.coordinates
            width = x2 - x1
            height = y2 - y1

            # Get color based on element type
            block_type = block.type
            color = self.color_map.get(block_type, "gray")

            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                alpha=0.7,
            )
            ax.add_patch(rect)

            # Add label with type and confidence
            label = f"Type: {block_type}, Conf: {block.score:.2f}"
            ax.text(
                x1,
                y1 - 5,
                label,
                fontsize=8,
                color="white",
                bbox=dict(facecolor=color, alpha=0.7),
            )

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            print(f"Visualization saved to {output_path}")

        plt.close()
        return fig

    def process_pdf(
        self,
        pdf_path: str,
        save_visualizations: bool = False,
        output_dir: str = "data/pdfs/borders",
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract layouts from all pages.

        Args:
            pdf_path (str): Path to the PDF file
            save_visualizations (bool): Whether to save visualizations of the layouts
            output_dir (str): Directory to save visualizations
            min_confidence (float): Minimum confidence threshold for visualization

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing layout information for each page
        """
        pages = self.load_pdf(pdf_path)
        results = []

        # Create output directory if needed
        if save_visualizations and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get PDF filename without extension
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

        for page_num, page in enumerate(pages):
            try:
                print(f"\nProcessing page {page_num + 1}...")
                layout = self.extract_layout(np.array(page))
                page_result = {
                    "page_number": page_num + 1,
                    "layout": layout,
                    "elements": [
                        {
                            "type": block.type,
                            "coordinates": block.coordinates,
                            "confidence": block.score,
                        }
                        for block in layout
                    ],
                }
                results.append(page_result)
                print(f"Successfully processed page {page_num + 1}")

                # Create visualization if requested
                if save_visualizations:
                    output_path = os.path.join(
                        output_dir, f"{pdf_name}_page_{page_num+1}.png"
                    )
                    self.visualize_layout(
                        np.array(page), layout, output_path, min_confidence
                    )

            except Exception as e:
                print(f"Warning: Failed to process page {page_num + 1}: {str(e)}")
                continue

        return results


def main():
    # Example usage
    parser = LayoutParser()
    pdf_path = "data/pdfs/0dc93c8e14ea49dcb2bff8784dc41a48.pdf"

    try:
        results = parser.process_pdf(
            pdf_path, save_visualizations=True, min_confidence=0.5
        )
        print(f"\nProcessed {len(results)} pages")

        # Print layout information for the first page
        if results:
            first_page = results[0]
            print(
                f"\nLayout elements on page {first_page['page_number']} (showing only top 5):"
            )
            # Sort elements by confidence and show the top 5
            top_elements = sorted(
                first_page["elements"], key=lambda x: x["confidence"], reverse=True
            )[:5]
            for element in top_elements:
                print(f"Type: {element['type']}")
                print(f"Confidence: {element['confidence']:.2f}")
                print(f"Coordinates: {element['coordinates']}")
                print("---")

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")


if __name__ == "__main__":
    main()
