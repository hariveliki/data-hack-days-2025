import layoutparser as lp
import numpy as np
from PIL import Image


def test_model_loading():
    print("Testing model loading...")

    # Try loading the model directly
    try:
        print("Loading model with Detectron2LayoutModel...")
        model = lp.Detectron2LayoutModel("lp://PubLayNet/faster_rcnn_R_50_FPN_3x")
        print(f"Model loaded successfully: {model}")

        # Create a simple test image
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255

        # Try to detect layout
        print("Detecting layout on test image...")
        layout = model.detect(test_image)
        print(f"Layout detection successful! Found {len(layout)} elements")

        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print(
            "\nModel loading SUCCESS! Your layoutparser and Detectron2 are now working properly."
        )
    else:
        print("\nModel loading FAILED. Please check the error message above.")
