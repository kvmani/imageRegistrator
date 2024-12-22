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
import EBSDImageGenerator


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
        self.zoom_levels = [1.0, 1.0, 1.0, 1.0]  # For each subplot
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
        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Control panel
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=2, column=0, columnspan=4, pady=10)
        tk.Button(control_frame, text="Load EBSD Data", command=self.load_original_image).grid(row=0, column=0)
        tk.Button(control_frame, text="Load LRS Data", command=self.load_transformed_image).grid(row=0, column=1)
        tk.Button(control_frame, text="Register with Affine", command=self.register_with_affine).grid(row=0, column=2)
        tk.Button(control_frame, text="Register with RANSAC", command=self.register_with_ransac).grid(row=0, column=3)

        # Point editing frame
        edit_frame = tk.Frame(self.root)
        edit_frame.grid(row=3, column=0, columnspan=4, pady=10)
        tk.Label(edit_frame, text="EBSD Image Points:").grid(row=0, column=0)
        tk.Label(edit_frame, text="LRS Image Points:").grid(row=0, column=1)

        self.original_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.original_points_listbox.grid(row=1, column=0, padx=5)

        self.transformed_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.transformed_points_listbox.grid(row=1, column=1, padx=5)

        tk.Button(edit_frame, text="Delete Selected Point (Original)", command=self.delete_original_point).grid(row=2, column=0)
        tk.Button(edit_frame, text="Delete Selected Point (Transformed)", command=self.delete_transformed_point).grid(row=2, column=1)

    def on_click(self, event):
        """Handles mouse clicks on Original and Transformed Images."""
        if event.inaxes == self.axs[0]:  # Original Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.fixed_points.append((x, y))
                self.original_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                # Annotate the point with Point ID
                point_id = len(self.fixed_points)  # Use the index as Point ID
                annotation = self.axs[0].scatter(x, y, color='red', zorder=5)
                text = self.axs[0].text(x, y, f"{point_id}", color='red', fontsize=10, zorder=6)
                self.fixed_points[-1] = (x, y, annotation, text)
                self.canvas.draw()
        elif event.inaxes == self.axs[1]:  # Transformed Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.moving_points.append((x, y))
                self.transformed_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                # Annotate the point with Point ID
                point_id = len(self.moving_points)  # Use the index as Point ID
                annotation = self.axs[1].scatter(x, y, color='blue', zorder=5)
                text = self.axs[1].text(x, y, f"{point_id}", color='blue', fontsize=10, zorder=6)
                self.moving_points[-1] = (x, y, annotation, text)
                self.canvas.draw()



    def on_zoom(self, event):
        """Handles zooming in and out on scroll."""
        for ax in self.axs:
            if event.inaxes == ax:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                # Calculate zoom factor
                zoom_factor = 0.2
                if event.button == 'up':  # Zoom in
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min += x_range
                    x_max -= x_range
                    y_min += y_range
                    y_max -= y_range
                elif event.button == 'down':  # Zoom out
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min -= x_range
                    x_max += x_range
                    y_min -= y_range
                    y_max += y_range

                # Ensure no empty canvas regions
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                if self.original_image is not None:
                    x_max = min(x_max, np.array(self.original_image).shape[1])
                    y_max = min(y_max, np.array(self.original_image).shape[0])

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)  # Invert Y-axis for correct display
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
                    ebsd_gen = EBSDImageGenerator.EBSDImageGenerator(file_path, output_folder)
                    self.original_image = ebsd_gen.image
                    self.axs[0].imshow(np.array(self.original_image), cmap='gray')
                    self.axs[0].set_title("EBSD Image")
                    self.canvas.draw()
                except Exception as e:
                    print(f"Error loading EBSD file: {e}")
            else:
                self.original_image = imread(file_path, as_gray=True)
                self.axs[0].imshow(self.original_image, cmap='gray')
                self.axs[0].set_title("Original Image")
                self.canvas.draw()

    def delete_original_point(self):
        selected = self.original_points_listbox.curselection()
        if selected:
            index = selected[0]
            _, _, scatter, text = self.fixed_points[index]  # Retrieve scatter and text handles
            scatter.remove()  # Remove scatter annotation from the plot
            text.remove()  # Remove text annotation from the plot
            self.fixed_points.pop(index)  # Remove the point data
            self.original_points_listbox.delete(index)  # Remove from listbox
            self.canvas.draw()

    def delete_transformed_point(self):
        selected = self.transformed_points_listbox.curselection()
        if selected:
            index = selected[0]
            _, _, scatter, text = self.moving_points[index]  # Retrieve scatter and text handles
            scatter.remove()  # Remove scatter annotation from the plot
            text.remove()  # Remove text annotation from the plot
            self.moving_points.pop(index)  # Remove the point data
            self.transformed_points_listbox.delete(index)  # Remove from listbox
            self.canvas.draw()
    def load_transformed_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.transformed_image = imread(file_path, as_gray=True)
            self.axs[1].imshow(self.transformed_image, cmap='gray')
            self.axs[1].set_title("LRS Image")
            self.canvas.draw()



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRegistrationTool(root)
    root.mainloop()

