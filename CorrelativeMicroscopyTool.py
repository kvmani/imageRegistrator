import os
import numpy as np
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

        # Logger window
        logger_frame = tk.Frame(self.root)
        logger_frame.grid(row=4, column=0, columnspan=4, pady=10)
        self.logger = tk.Text(logger_frame, width=80, height=10, state=tk.DISABLED)
        self.logger.pack()

    def log(self, message):
        """Logs a message to the logger window."""
        self.logger.config(state=tk.NORMAL)
        self.logger.insert(tk.END, message + "\n")
        self.logger.config(state=tk.DISABLED)
        self.logger.see(tk.END)

    def on_click(self, event):
        if event.inaxes == self.axs[0]:  # Original Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.fixed_points.append((x, y))
                self.original_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                self.axs[0].scatter(x, y, color='red', zorder=5)
                self.canvas.draw()

        elif event.inaxes == self.axs[1]:  # Transformed Image
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.moving_points.append((x, y))
                self.transformed_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                self.axs[1].scatter(x, y, color='blue', zorder=5)
                self.canvas.draw()

    def on_zoom(self, event):
        for ax in self.axs:
            if event.inaxes == ax:
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                zoom_factor = 0.2
                if event.button == 'up':
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min += x_range
                    x_max -= x_range
                    y_min += y_range
                    y_max -= y_range
                elif event.button == 'down':
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min -= x_range
                    x_max += x_range
                    y_min -= y_range
                    y_max += y_range

                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                if self.original_image is not None:
                    x_max = min(x_max, self.original_image.shape[1])
                    y_max = min(y_max, self.original_image.shape[0])

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)
                self.canvas.draw()

    def register_with_affine(self):
        if len(self.fixed_points) < 3 or len(self.moving_points) < 3:
            self.log("At least 3 points are required for affine registration.")
            return

        try:
            fixed_points_coords = np.array(self.fixed_points)
            moving_points_coords = np.array(self.moving_points)

            transform = AffineTransform()
            transform.estimate(moving_points_coords, fixed_points_coords)
            self.apply_transformation(transform)

            # Extract rotation and scaling from the transformation matrix
            rotation = np.degrees(np.arctan2(transform.params[1, 0], transform.params[0, 0]))
            scale = np.sqrt(transform.params[0, 0]**2 + transform.params[1, 0]**2)

            self.log(f"Affine Registration Completed.")
            self.log(f"Rotation: {rotation:.2f} degrees")
            self.log(f"Scaling: {scale:.2f}")
        except Exception as e:
            self.log(f"Affine registration error: {e}")

    def register_with_ransac(self):
        if len(self.fixed_points) < 3 or len(self.moving_points) < 3:
            self.log("At least 3 points are required for RANSAC registration.")
            return

        try:
            fixed_points_coords = np.array(self.fixed_points)
            moving_points_coords = np.array(self.moving_points)

            model, _ = ransac((moving_points_coords, fixed_points_coords), AffineTransform, min_samples=3, residual_threshold=2)
            self.apply_transformation(model)

            # Extract rotation and scaling from the transformation matrix
            rotation = np.degrees(np.arctan2(model.params[1, 0], model.params[0, 0]))
            scale = np.sqrt(model.params[0, 0]**2 + model.params[1, 0]**2)

            self.log("RANSAC Registration Completed.")
            self.log(f"Rotation: {rotation:.2f} degrees")
            self.log(f"Scaling: {scale:.2f}")
        except Exception as e:
            self.log(f"RANSAC registration error: {e}")

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image_path = file_path
            if file_path.endswith('.ang'):
                try:
                    output_folder = os.path.dirname(file_path)
                    ebsd_gen = EBSDImageGenerator.EBSDImageGenerator(file_path, output_folder)
                    self.original_image = np.array(ebsd_gen.image)
                    self.axs[0].imshow(self.original_image, cmap='gray')
                    self.canvas.draw()
                    self.log("Loaded EBSD Image.")
                except Exception as e:
                    self.log(f"Error loading EBSD file: {e}")
            else:
                self.original_image = np.array(imread(file_path, as_gray=True))
                self.axs[0].imshow(self.original_image, cmap='gray')
                self.canvas.draw()
                self.log("Loaded Original Image.")

    def load_transformed_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.transformed_image = np.array(imread(file_path, as_gray=True))
            self.axs[1].imshow(self.transformed_image, cmap='gray')
            self.canvas.draw()
            self.log("Loaded Transformed Image.")

    def apply_transformation(self, transform):
        if self.original_image is None or self.transformed_image is None:
            self.log("Error: Load both original and transformed images before registration.")
            return

        try:
            self.registered_image = warp(self.transformed_image, transform.inverse, output_shape=self.original_image.shape)
            self.axs[2].imshow(self.registered_image, cmap='gray')
            self.axs[3].imshow(0.5 * self.original_image + 0.5 * self.registered_image, cmap='gray')
            self.canvas.draw()
            self.export_registered_image()
        except Exception as e:
            self.log(f"Error during transformation: {e}")

    def export_registered_image(self):
        try:
            output_folder = os.path.dirname(self.original_image_path)
            output_path = os.path.join(output_folder, "registeredLRSImage.png")
            cv2.imwrite(output_path, (self.registered_image * 255).astype(np.uint8))
            self.log(f"Registered image saved as: {output_path}")
        except Exception as e:
            self.log(f"Error exporting registered image: {e}")

    def delete_original_point(self):
        selected = self.original_points_listbox.curselection()
        if selected:
            index = selected[0]
            self.fixed_points.pop(index)
            self.original_points_listbox.delete(index)
            self.canvas.draw()

    def delete_transformed_point(self):
        selected = self.transformed_points_listbox.curselection()
        if selected:
            index = selected[0]
            self.moving_points.pop(index)
            self.transformed_points_listbox.delete(index)
            self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRegistrationTool(root)
    root.mainloop()
