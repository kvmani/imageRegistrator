import os
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from skimage.io import imread
from skimage.transform import AffineTransform, warp
from skimage.measure import ransac


# EBSD Image Generator Class
class EBSDImageGenerator:
    def __init__(self, filepath, output_folder):
        self.filepath = filepath
        self.output_folder = output_folder
        self.header = {}
        self.data = None
        self.image = None
        self.validate_file()
        self.read_file()
        self.generate_image()
        self.save_image()

    def validate_file(self):
        if not os.path.isfile(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}. Please verify the file path.")
        print(f"File located: {self.filepath}")

    def read_file(self):
        valid_keys = {"XSTEP", "YSTEP", "NCOLS_ODD", "NROWS"}
        with open(self.filepath, 'r') as f:
            header_lines = [line for line in f if line.startswith('#')]
        for line in header_lines:
            if ':' in line:
                key, value = line[2:].split(':')
                key = key.strip()
                if key in valid_keys:
                    self.header[key] = float(value.strip())

        # Read numeric data without the deprecated argument
        self.data = pd.read_csv(self.filepath, comment='#', delim_whitespace=True, header=None, on_bad_lines='skip')
        print("Header and numeric data successfully loaded.")

    def generate_image(self):
        ncols_odd = int(self.header.get('NCOLS_ODD', 0))
        nrows = int(self.header.get('NROWS', 0))
        if 5 not in self.data.columns:
            raise ValueError("Column 6 (IQ values) not found in data")
        if len(self.data[5]) != nrows * ncols_odd:
            raise ValueError("Data size mismatch: IQ values do not match specified grid dimensions.")
        self.image = self.data[5].values.reshape(nrows, ncols_odd)
        print("EBSD IQ image generated successfully.")

    def save_image(self):
        if self.image is not None:
            output_path = os.path.join(self.output_folder, "ImageAngNi.png")
            norm_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(output_path, norm_image)
            print(f"Image saved to: {output_path}")
        else:
            print("No image generated to save.")


# GUI Class
class ImageRegistrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Registration Tool")

        self.original_image = None
        self.transformed_image = None
        self.registered_image = None
        self.fixed_points = []
        self.moving_points = []
        self.setup_ui()

    def setup_ui(self):
        # Header label
        header_label = tk.Label(self.root, text="Tool For Correlative Microscopy", font=("Arial", 18, "bold"))
        header_label.grid(row=0, column=0, columnspan=4, pady=10)

        # Frames for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10)
        self.fig, self.axs = plt.subplots(1, 4, figsize=(16, 5))
        titles = ["EBSD IQ Image", "LRS Image", "Registered Image", "Superimposed Image"]
        for ax, title in zip(self.axs, titles):
            ax.set_title(title)
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=2, column=0, columnspan=4, pady=10)
        tk.Button(control_frame, text="Load EBSD Data", command=self.load_original_image).grid(row=0, column=0)
        tk.Button(control_frame, text="Load LRS Data", command=self.load_transformed_image).grid(row=0,
                                                                                                          column=1)
        tk.Button(control_frame, text="Register with Affine", command=self.register_with_affine).grid(row=0, column=2)
        tk.Button(control_frame, text="Register with RANSAC", command=self.register_with_ransac).grid(row=0, column=3)

        # Point editing frame
        edit_frame = tk.Frame(self.root)
        edit_frame.grid(row=3, column=0, columnspan=4, pady=10)
        tk.Label(edit_frame, text="Original Image Points:").grid(row=0, column=0)
        tk.Label(edit_frame, text="Transformed Image Points:").grid(row=0, column=1)

        self.original_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.original_points_listbox.grid(row=1, column=0, padx=5)

        self.transformed_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.transformed_points_listbox.grid(row=1, column=1, padx=5)

        tk.Button(edit_frame, text="Delete Selected Point (Original)", command=self.delete_original_point).grid(row=2,
                                                                                                                column=0)
        tk.Button(edit_frame, text="Delete Selected Point (Transformed)", command=self.delete_transformed_point).grid(
            row=2, column=1)

    def on_click(self, event):
        """Handles mouse clicks on Original and Transformed Images."""
        if event.inaxes == self.axs[0]:  # Original Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.fixed_points.append((x, y))
                self.original_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                self.axs[0].scatter(x, y, color='red')
                self.canvas.draw()
        elif event.inaxes == self.axs[1]:  # Transformed Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.moving_points.append((x, y))
                self.transformed_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                self.axs[1].scatter(x, y, color='blue')
                self.canvas.draw()

    def register_with_affine(self):
        if len(self.fixed_points) < 3 or len(self.moving_points) < 3:
            print("At least 3 points are required for affine registration.")
            return
        transform = AffineTransform()
        transform.estimate(np.array(self.moving_points), np.array(self.fixed_points))
        self.apply_transformation(transform)

    def register_with_ransac(self):
        if len(self.fixed_points) < 3 or len(self.moving_points) < 3:
            print("At least 3 points are required for RANSAC registration.")
            return
        model, _ = ransac((np.array(self.moving_points), np.array(self.fixed_points)),
                          AffineTransform, min_samples=3, residual_threshold=2)
        self.apply_transformation(model)

    def apply_transformation(self, transform):
        self.registered_image = warp(self.transformed_image, transform.inverse, output_shape=self.original_image.shape)
        self.axs[2].imshow(self.registered_image, cmap='gray')
        self.axs[2].set_title("Registered Image")

        # Individually normalize both images to [0, 1]
        normalized_original = (self.original_image - self.original_image.min()) / (
                    self.original_image.max() - self.original_image.min())
        normalized_registered = (self.registered_image - self.registered_image.min()) / (
                    self.registered_image.max() - self.registered_image.min())

        # Blend and rescale to 255
        blended = 0.5 * normalized_original + 0.5 * normalized_registered
        blended = (blended * 255).astype(np.uint8)

        self.axs[3].imshow(blended, cmap='gray')
        self.axs[3].set_title("Superimposed Image")
        self.canvas.draw()

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if file_path.endswith('.ang'):
                try:
                    output_folder = os.path.dirname(file_path)
                    ebsd_gen = EBSDImageGenerator(file_path, output_folder)
                    self.original_image = ebsd_gen.image
                    self.axs[0].imshow(self.original_image, cmap='gray')
                    self.axs[0].set_title("EBSD Image")
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error loading EBSD file: {e}")
            else:
                self.original_image = imread(file_path, as_gray=True)
                self.axs[0].imshow(self.original_image, cmap='gray')
                self.axs[0].set_title("Original Image")
                self.canvas.draw()

    def load_transformed_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.transformed_image = imread(file_path, as_gray=True)
            self.axs[1].imshow(self.transformed_image, cmap='gray')
            self.axs[1].set_title("Transformed Image")
            self.canvas.draw()

    def delete_original_point(self):
        selected = self.original_points_listbox.curselection()
        if selected:
            self.original_points_listbox.delete(selected[0])
            self.fixed_points.pop(selected[0])
            self.canvas.draw()

    def delete_transformed_point(self):
        selected = self.transformed_points_listbox.curselection()
        if selected:
            self.transformed_points_listbox.delete(selected[0])
            self.moving_points.pop(selected[0])
            self.canvas.draw()


# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRegistrationTool(root)
    root.mainloop()
