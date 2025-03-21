import numpy as np
from PIL import Image, ImageDraw
import cv2


def preprocess_image(img):
    """
    Preprocess the image: convert to grayscale, binarize, and return a binary array.
    """
    # Convert to grayscale
    img = img.convert("L")  # Grayscale
    img_array = np.array(img, dtype=np.float32)

    # Binarize: Text is black (0,0,0), background is white/light gray, noise is medium gray
    # In grayscale, black is 0, white is 255, and noise is around 190-200
    threshold = 50  # Pixels darker than this are considered text
    binary_img = (img_array < threshold).astype(
        np.float32
    )  # 1 for text, 0 for background

    # Visualise binary image
    # plt.imshow(binary_img, cmap="gray")
    # plt.show()

    return binary_img


def find_bounding_box(input_file):
    """
    Find the bounding box of characters in the input image file.
    Returns a tuple of ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
    """
    # Read the input image
    img = Image.open(input_file)
    img_array = np.array(img)

    # Preprocess to get binary image
    binary_img = preprocess_image(img)

    # Find non-zero (character) pixels
    y_indices, x_indices = np.nonzero(binary_img)

    if len(y_indices) == 0:
        return None  # No characters found

    # Get bounding box coordinates
    top_left_y, top_left_x = np.min(y_indices), np.min(x_indices)
    bottom_right_y, bottom_right_x = np.max(y_indices), np.max(x_indices)

    # Ensure coordinates don't go out of bounds
    height, width = binary_img.shape
    top_left_y = max(0, top_left_y)
    top_left_x = max(0, top_left_x)
    bottom_right_y = min(height, bottom_right_y)
    bottom_right_x = min(width, bottom_right_x)

    # # Debug: Visualize using OpenCV
    # img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # cv2.rectangle(
    #     img_cv,
    #     (top_left_x, top_left_y),
    #     (bottom_right_x, bottom_right_y),
    #     (0, 255, 0),
    #     1,  # line width
    # )
    # cv2.imshow("Bounding Box", img_cv)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))


def get_area_of_interest():
    """
    Get the area of interest from the input image file.
    Returns:
        tuple, ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
    """

    bboxes = []
    for i in range(25):
        input_file = f"sampleCaptchas/input/input{i:02d}.jpg"
        bbox = find_bounding_box(input_file)
        bboxes.append(bbox)

    # Check if all bboxes are the same
    for i, bbox in enumerate(bboxes):
        print(f"Bounding box of input{i:02d}: {bbox}")

    return bboxes[0]


if __name__ == "__main__":
    # Example usage
    get_area_of_interest()
