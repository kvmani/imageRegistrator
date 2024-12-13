# File: test_image_generator.py
import cv2
import numpy as np

class TestImageGenerator:
    def __init__(self, size=(500, 500)):
        self.size = size

    def create_base_image(self):
        """Creates an image with simple geometric shapes."""
        image = np.zeros(self.size + (3,), dtype=np.uint8)
        # Draw a rectangle
        cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)
        # Draw a circle
        cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)
        # Draw a triangle
        pts = np.array([[200, 400], [250, 350], [300, 400]], np.int32)
        cv2.fillPoly(image, [pts], (0, 0, 255))
        return image

    def apply_transformation(self, image, scale=1.0, rotation=0, shift=(0, 0)):
        """Applies scaling, rotation, and shifting to an image."""
        rows, cols, _ = image.shape
        # Create transformation matrix
        M_scale = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, scale)
        transformed = cv2.warpAffine(image, M_scale, (cols, rows))
        M_shift = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        transformed = cv2.warpAffine(transformed, M_shift, (cols, rows))
        return transformed
