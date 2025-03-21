import numpy as np
from PIL import Image
import os
from collections import defaultdict
import os.path as osp
import matplotlib.pyplot as plt
import cv2

PROTJECT_PATH = osp.dirname(osp.abspath(__file__))
DATA_PATH = osp.join(PROTJECT_PATH, "sampleCaptchas")


class Captcha(object):
    def __init__(self):
        # Define the character set (A-Z, 0-9)
        self.characters = [chr(i) for i in range(65, 91)] + [
            str(i) for i in range(10)
        ]  # A-Z, 0-9
        self.num_chars = len(self.characters)  # 36 characters
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}

        # Image dimensions (from the pixel data)
        self.height = 30
        self.width = 60
        self.num_positions = 5  # 5 characters in the CAPTCHA

        # Fixed character area coordinates with heuristic offsets
        offset1, offset2 = 1, 2
        # Find self.char_area using find_area_of_interest.py
        self.char_area = (
            (5 - offset1, 11 - offset1),
            (47 + offset2, 20 + offset2),
        )  # ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
        self.char_width = (
            self.char_area[1][0] - self.char_area[0][0]
        ) // self.num_positions  # Width per character

        # Placeholder for character templates (to be computed from training data)
        self.templates = {}  # Dictionary: char -> template (numpy array)

        # Load the 25 sample CAPTCHAs and their labels here and compute the templates.
        self._load_templates()

    def _load_templates(self):
        """
        Load the 25 sample CAPTCHAs, segment them, and create templates.
        Process:
        1. For each CAPTCHA image and its label:
           - Load the image and convert to grayscale
           - Binarize the image to isolate the text
           - Segment into 5 regions
           - For each region, extract the character and associate it with the corresponding label
        2. For each character (A-Z, 0-9), average the pixel values across all instances to create a template
        3. Store the templates in self.templates
        """
        # Dictionary to store all instances of each character
        char_instances = defaultdict(list)

        # Process each training sample
        for i in range(25):  # We have 25 training samples
            # Load the image
            img_path = osp.join(DATA_PATH, "input", f"input{i:02d}.jpg")
            if not osp.exists(img_path):
                continue

            # Load the label
            label_path = osp.join(DATA_PATH, "output", f"output{i:02d}.txt")
            if not osp.exists(label_path):
                continue

            with open(label_path, "r") as f:
                label = f.read().strip()

            if len(label) != 5:  # Skip if label is not 5 characters
                continue

            # Load and preprocess the image
            img = Image.open(img_path)
            binary_img = self._preprocess_image(img)

            # Segment the image
            segments = self._segment_image(binary_img)

            # Store each character segment with its label
            for char, segment in zip(label, segments):
                char_instances[char].append(segment)

        # Create templates by averaging all instances of each character
        for char in self.characters:
            if char in char_instances and char_instances[char]:
                # Stack all instances of this character and compute mean
                instances = np.stack(char_instances[char])
                template = np.mean(instances, axis=0)
                # Binarize the template
                self.templates[char] = (template > 0.5).astype(np.float32)
            else:
                # If no instances found, create an empty template
                self.templates[char] = np.zeros(
                    (self.height, self.char_width), dtype=np.float32
                )

    def _preprocess_image(self, img):
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

        # # Debug: visualise binary image with bounding box
        # binary_img_cv = (binary_img * 255).astype(np.uint8)
        # binary_img_cv = cv2.cvtColor(binary_img_cv, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(
        #     binary_img_cv,
        #     (self.char_area[0][0], self.char_area[0][1]),
        #     (self.char_area[1][0], self.char_area[1][1]),
        #     (0, 255, 0),
        #     1,  # line width
        # )
        # cv2.imshow("Binary Image with Bounding Box", binary_img_cv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return binary_img

    def _segment_image(self, binary_img):
        """
        Segment the binary image into 5 regions, each containing one character.
        Uses fixed character area coordinates to extract characters.
        Accounts for 1-pixel spacing between characters.
        """
        segments = []
        # Extract only the area containing characters
        char_area_img = binary_img[
            self.char_area[0][1] : self.char_area[1][1],
            self.char_area[0][0] : self.char_area[1][0],
        ]

        # Calculate character width including spacing
        total_width = self.char_area[1][0] - self.char_area[0][0]
        char_width = (
            total_width - 4
        ) // 5  # Subtract 4 spaces (between 5 chars) and divide by 5 chars

        for i in range(self.num_positions):
            start_x = i * (char_width + 1)  # Add 1 for the spacing
            end_x = start_x + char_width
            segment = char_area_img[:, start_x:end_x]
            segments.append(segment)

        # Debug: visualise segments using subplots
        # plt.figure(figsize=(15, 4))
        # for i, segment in enumerate(segments):
        #     plt.subplot(1, 5, i + 1)
        #     plt.imshow(segment, cmap="gray")
        #     plt.title(f"Segment {i+1}")
        #     plt.axis("off")
        # plt.show()

        return segments

    def _match_template(self, segment):
        """
        Match a segment against all character templates and return the best match.
        """
        best_score = float("inf")
        best_char = None

        for char, template in self.templates.items():
            # Compute the mean squared error between the segment and the template
            if segment.shape != template.shape:
                continue  # Skip if shapes don't match
            score = np.mean((segment - template) ** 2)
            if score < best_score:
                best_score = score
                best_char = char

        return best_char

    def __call__(self, im_path, save_path):
        """
        Inference method: Load the image, predict the CAPTCHA text, and save to file.
        args:
            im_path: Path to the CAPTCHA image (.jpg)
            save_path: Path to save the predicted text
        """
        # Load the image
        img = Image.open(im_path)

        # Preprocess the image
        binary_img = self._preprocess_image(img)

        # Segment the image into 5 regions
        segments = self._segment_image(binary_img)

        # Predict the character in each segment
        predicted_text = ""
        for segment in segments:
            char = self._match_template(segment)
            predicted_text += char

        # Save the predicted text to the save_path
        with open(save_path, "w") as f:
            f.write(predicted_text)

        return predicted_text


def main():
    for i in range(25):
        # Make predictions
        im_path = osp.join(DATA_PATH, "input", f"input{i:02d}.jpg")
        save_path = osp.join(PROTJECT_PATH, f"output{i:02d}.txt")

        captcha_solver = Captcha()
        predicted_text = captcha_solver(im_path, save_path)

        # Load the ground truth label
        img_path = osp.join(DATA_PATH, "input", f"input{i:02d}.jpg")
        label_path = osp.join(
            DATA_PATH,
            "output",
            f"output{osp.basename(img_path).split('.')[0].split('t')[1]}.txt",
        )

        if not osp.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        with open(label_path, "r") as f:
            gt_label = f.read().strip()

        # Check if the predicted text is correct
        if predicted_text != gt_label:
            print(f"{i}: incorrect")
            print(f"Predicted CAPTCHA text: {predicted_text}")
            print(f"GT CAPTCHA text: {gt_label}")


# Example usage (for testing)
if __name__ == "__main__":
    main()
