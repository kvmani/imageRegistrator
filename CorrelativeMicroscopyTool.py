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
import pandas as pd

# GUI Class
class ImageRegistrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Registration Tool")
        self.root.geometry("1200x800")  # Set window size

        # Images
        self.original_image = None            # EBSD image data (the "fixed" image)
        self.lrs_image_original = None        # Unmodified LRS image data
        self.transformed_image = None         # LRS image currently displayed (the "moving" image)
        self.registered_image = None

        # Lists of picked points
        self.fixed_points = []
        self.moving_points = []

        # Data for zoom (unused in example but included from your code)
        self.zoom_levels = [1.0, 1.0, 1.0, 1.0]  # For each subplot

        # Matrices for LRS data (intensity, waveNumber, shift)
        self.raw_lrs_intensity_matrix = None
        self.raw_lrs_waveNumber_matrix = None
        self.raw_lrs_shift_matrix = None

        self.setup_ui()

    def setup_ui(self):
        # ========== Row 0: Header label ==========
        header_label = tk.Label(self.root, text="Tool For Correlative Microscopy", font=("Arial", 18, "bold"))
        header_label.grid(row=0, column=0, columnspan=4, pady=10)

        # ========== Row 1: Two Sliders at the Top ==========
        # Create a single frame to hold both sliders side by side
        top_sliders_frame = tk.Frame(self.root)
        top_sliders_frame.grid(row=1, column=0, columnspan=4, padx=10, pady=10, sticky="w")

        # --- EBSD Slider Frame ---
        ebsd_slider_frame = tk.Frame(top_sliders_frame)
        ebsd_slider_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(ebsd_slider_frame, text="EBSD Contrast: ").pack(side=tk.LEFT)
        self.contrast_slider_ebsd = tk.Scale(
            ebsd_slider_frame,
            from_=0.1, to=5.0, resolution=0.2,
            orient="horizontal",
            command=self.update_contrast_ebsd,
            length=200  # fixed length so it doesn't span the entire width
        )
        self.contrast_slider_ebsd.set(1.0)
        self.contrast_slider_ebsd.pack(side=tk.LEFT)

        # --- LRS Slider Frame ---
        lrs_slider_frame = tk.Frame(top_sliders_frame)
        lrs_slider_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(lrs_slider_frame, text="LRS Contrast: ").pack(side=tk.LEFT)
        self.contrast_slider_lrs = tk.Scale(
            lrs_slider_frame,
            from_=0.5, to=5.0, resolution=0.2,
            orient="horizontal",
            command=self.update_contrast_lrs,
            length=200
        )
        self.contrast_slider_lrs.set(2.5)
        self.contrast_slider_lrs.pack(side=tk.LEFT)

        # ========== Row 2: Image frame (4 subplots) ==========
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=2, column=0, columnspan=4, padx=10, pady=10, sticky="nsew")
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.fig, self.axs = plt.subplots(1, 4, figsize=(16, 5))
        titles = ["EBSD IQ Image", "LRS Image", "Registered Image", "Superimposed Image"]
        for ax, title in zip(self.axs, titles):
            ax.set_title(title)
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_zoom)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # ========== Row 3: Control panel (buttons) ==========
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=3, column=0, columnspan=4, pady=10)
        tk.Button(control_frame, text="Load EBSD Data", command=self.load_original_image).grid(row=0, column=0, padx=5)
        tk.Button(control_frame, text="Load LRS Data", command=self.load_transformed_image).grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Register with Affine", command=self.register_with_affine).grid(row=0, column=2, padx=5)
        tk.Button(control_frame, text="Register with RANSAC", command=self.register_with_ransac).grid(row=0, column=3, padx=5)

        # ========== Row 4: Point editing frame ==========
        edit_frame = tk.Frame(self.root)
        edit_frame.grid(row=4, column=0, columnspan=4, pady=10)
        tk.Label(edit_frame, text="EBSD Image Points:").grid(row=0, column=0)
        tk.Label(edit_frame, text="LRS Image Points:").grid(row=0, column=1)

        self.original_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.original_points_listbox.grid(row=1, column=0, padx=5)
        self.transformed_points_listbox = tk.Listbox(edit_frame, width=30, height=10)
        self.transformed_points_listbox.grid(row=1, column=1, padx=5)

        tk.Button(edit_frame, text="Delete Selected Point (Original)", command=self.delete_original_point).grid(row=2, column=0, pady=5)
        tk.Button(edit_frame, text="Delete Selected Point (Transformed)", command=self.delete_transformed_point).grid(row=2, column=1, pady=5)

        # ========== Row 5: Logger window ==========
        logger_frame = tk.Frame(self.root)
        logger_frame.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")
        self.logger = tk.Text(logger_frame, width=80, height=10, state=tk.DISABLED)
        self.logger.pack(fill=tk.BOTH, expand=True)

    # ----------------------------------------------------------------------
    # Logging Utility
    # ----------------------------------------------------------------------
    def log(self, message):
        """Logs a message to the logger window."""
        self.logger.config(state=tk.NORMAL)
        self.logger.insert(tk.END, message + "\n")
        self.logger.config(state=tk.DISABLED)
        self.logger.see(tk.END)

    # ----------------------------------------------------------------------
    # Mouse Events (Click + Zoom)
    # ----------------------------------------------------------------------
    def on_click(self, event):
        # If clicked inside the EBSD axes:
        if event.inaxes == self.axs[0]:
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                self.fixed_points.append((x, y))
                self.original_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
                self.axs[0].scatter(x, y, color='red', zorder=5)
                self.canvas.draw()

        # If clicked inside the LRS axes:
        elif event.inaxes == self.axs[1]:
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
                    # Zoom in
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min += x_range
                    x_max -= x_range
                    y_min += y_range
                    y_max -= y_range
                elif event.button == 'down':
                    # Zoom out
                    x_range = (x_max - x_min) * zoom_factor
                    y_range = (y_max - y_min) * zoom_factor
                    x_min -= x_range
                    x_max += x_range
                    y_min -= y_range
                    y_max += y_range

                # Keep zoom within image bounds
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                if self.original_image is not None:
                    x_max = min(x_max, self.original_image.shape[1])
                    y_max = min(y_max, self.original_image.shape[0])

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_max, y_min)
                self.canvas.draw()

    # ----------------------------------------------------------------------
    # Contrast Sliders
    # ----------------------------------------------------------------------
    @staticmethod
    def normalize_image(image, orig_dtype,contrast_factor):
        """
        Normalizes an adjusted image for display.

        For integer images (e.g. uint8, int16, etc.):
          - Computes the current min and max of the image.
          - Linearly scales the image data so that the minimum becomes 0 and the maximum becomes 255.
          - Clips any values outside [0, 255] and converts to uint8.

        For float images:
          - Computes the current min and max of the image.
          - Linearly scales the image data so that the minimum becomes 0 and the maximum becomes 1.
          - Clips any values outside [0, 1] and returns a float32 array.

        Parameters:
            image (np.ndarray): The contrast-adjusted image (should be float32 for arithmetic).
            orig_dtype (dtype): The original data type of the image.

        Returns:
            np.ndarray: The normalized image ready for display.
        """

        print(f"before adusting : min: {image.min()}, max: {image.max()}, mean: {image.mean()}")
        if image is None:
            return None

        im_min = image.min()
        im_max = image.max()

        # Avoid division by zero if the image is constant.
        if im_max == im_min:
            scaled = np.zeros_like(image)
        else:
            scaled = (image - im_min) / (im_max - im_min)

        if np.issubdtype(orig_dtype, np.integer):
            # Scale to full 0-255 range and convert to uint8.
            scaled = scaled * 255.0
            scaled = scaled.astype(np.float32) * contrast_factor
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
            #return scaled.astype(np.uint8)
        elif np.issubdtype(orig_dtype, np.floating):
            # Scale to full 0-1 range.
            scaled = scaled.astype(np.float32) * contrast_factor
            scaled = np.clip(scaled, 0, 1)
            #return scaled.astype(np.float32)
        adjusted = scaled
        print(f"after adusting : min: {adjusted.min()}, max: {adjusted.max()}, mean: {adjusted.mean()}")
        return adjusted


    # Inside your ImageRegistrationTool class:

    def update_contrast_ebsd(self, value):
        """
        Adjusts the contrast of the EBSD image (axs[0]) based on slider.
        """
        if self.original_image is not None:
            contrast_factor = float(value)
            # Multiply the original image by the contrast factor.
            adjusted_image = self.original_image.astype(np.float32) * contrast_factor
            # Normalize the adjusted image to the proper display range.
            adjusted_image = self.normalize_image(self.original_image, self.original_image.dtype,contrast_factor)
            print(f"EBSD contrast updated: {contrast_factor}")
            self.axs[0].imshow(adjusted_image, cmap='gray')
            self.canvas.draw()

    def update_contrast_lrs(self, value):
        """
        Adjusts the contrast of the LRS image (axs[1]) based on slider.
        Uses the unmodified self.lrs_image_original as a source.
        """
        if self.lrs_image_original is not None:
            contrast_factor = float(value)
            # Multiply the original LRS image by the contrast factor.
            adjusted_image = self.lrs_image_original.astype(np.float32) * contrast_factor
            # Normalize the adjusted image.
            adjusted_image = self.normalize_image(self.lrs_image_original, self.lrs_image_original.dtype,contrast_factor)
            print(f"LRS contrast updated: {contrast_factor}")
            self.axs[1].imshow(adjusted_image, cmap='gray')
            self.canvas.draw()

    # ----------------------------------------------------------------------
    # Registration Methods
    # ----------------------------------------------------------------------
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

            self.log("Affine Registration Completed.")
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

            model, _ = ransac(
                (moving_points_coords, fixed_points_coords),
                AffineTransform,
                min_samples=3,
                residual_threshold=2
            )
            self.apply_transformation(model)

            # Extract rotation and scaling from the transformation matrix
            rotation = np.degrees(np.arctan2(model.params[1, 0], model.params[0, 0]))
            scale = np.sqrt(model.params[0, 0]**2 + model.params[1, 0]**2)

            self.log("RANSAC Registration Completed.")
            self.log(f"Rotation: {rotation:.2f} degrees")
            self.log(f"Scaling: {scale:.2f}")
        except Exception as e:
            self.log(f"RANSAC registration error: {e}")

    # ----------------------------------------------------------------------
    # Loading Images
    # ----------------------------------------------------------------------
    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image_path = file_path
            if file_path.endswith('.ang'):
                # EBSD file
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
                # Standard image file
                self.original_image = np.array(imread(file_path, as_gray=True))
                self.axs[0].imshow(self.original_image, cmap='gray')
                self.canvas.draw()
                self.log("Loaded Original Image.")

    def load_transformed_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Store an original copy for repeated contrast adjustments
            self.lrs_image_original = self.load_lrs_csv(file_path)
            # Copy it for displaying
            self.transformed_image = self.lrs_image_original.copy()

            # Show LRS in subplot[1]
            self.axs[1].imshow(self.transformed_image, cmap='gray')
            self.canvas.draw()
            self.log("Loaded Transformed Image.")

    def load_lrs_csv(self, csv_file):
        """
        Reads a CSV file and returns a 2D array of MaxIntensity values.
        Expects columns: X, Y, WaveNumber, MaxIntensity, shift.
        """
        df = pd.read_csv(csv_file, delimiter=",")
        intensity_matrix = df.pivot(index='Y', columns='X', values='shift').to_numpy()
        waveNumber_matrix = df.pivot(index='Y', columns='X', values='WaveNumber').to_numpy()
        shift_matrix = df.pivot(index='Y', columns='X', values='shift').to_numpy()

        self.lrs_max = np.max(intensity_matrix)
        self.raw_lrs_intensity_matrix = intensity_matrix
        self.raw_lrs_waveNumber_matrix = waveNumber_matrix
        self.raw_lrs_shift_matrix = shift_matrix

        return intensity_matrix

    # ----------------------------------------------------------------------
    # Applying Transformation
    # ----------------------------------------------------------------------
    def apply_transformation(self, transform):
        if self.original_image is None or self.transformed_image is None:
            self.log("Error: Load both original and transformed images before registration.")
            return

        try:
            # Register the LRS image to EBSD shape
            self.registered_image = warp(
                self.transformed_image,
                transform.inverse,
                output_shape=self.original_image.shape
            )
            self.registed_lrs_intensity_matrix = warp(
                self.raw_lrs_intensity_matrix,
                transform.inverse,
                output_shape=self.original_image.shape
            )
            self.registed_lrs_waveNumber_matrix = warp(
                self.raw_lrs_waveNumber_matrix,
                transform.inverse,
                output_shape=self.original_image.shape
            )
            self.registed_lrs_shift_matrix = warp(
                self.raw_lrs_shift_matrix,
                transform.inverse,
                output_shape=self.original_image.shape
            )

            # Show registered image in axs[2], superimposed in axs[3]
            self.axs[2].imshow(self.registered_image, cmap='gray')
            self.axs[3].imshow(0.5 * self.original_image + 0.5 * self.registered_image, cmap='gray')
            self.canvas.draw()

            # Optionally export the registered image
            self.export_registered_image()
        except Exception as e:
            self.log(f"Error during transformation: {e}")

    # ----------------------------------------------------------------------
    # Exporting
    # ----------------------------------------------------------------------
    def export_registered_image(self):
        try:
            output_folder = os.path.dirname(self.original_image_path)
            output_path = os.path.join(output_folder, "registeredLRSImage.png")
            output_path_lrs_intensity = os.path.join(output_folder, "registeredLrsIntensity.csv")
            output_path_lrs_waveNumber = os.path.join(output_folder, "RegistredLrs_waveNumber.csv")
            output_path_lrs_shift = os.path.join(output_folder, "Registred_registeredLrsShift.csv")

            lrs_out_intensity = self.registered_image
            lrs_out_waveNumber = self.registed_lrs_waveNumber_matrix
            lrs_out_shift = self.registed_lrs_shift_matrix

            # Save CSV files
            np.savetxt(
                output_path_lrs_intensity,
                lrs_out_intensity,
                delimiter=",",
                header=f"{lrs_out_intensity.shape[0]},{lrs_out_intensity.shape[1]}",
                comments="",
                fmt="%.2f"
            )
            np.savetxt(
                output_path_lrs_waveNumber,
                lrs_out_waveNumber,
                delimiter=",",
                header=f"{lrs_out_intensity.shape[0]},{lrs_out_intensity.shape[1]}",
                comments="",
                fmt="%.2f"
            )
            np.savetxt(
                output_path_lrs_shift,
                lrs_out_shift,
                delimiter=",",
                header=f"{lrs_out_intensity.shape[0]},{lrs_out_intensity.shape[1]}",
                comments="",
                fmt="%.2f"
            )

            # Save PNG image
            cv2.imwrite(output_path, (self.registered_image * 255).astype(np.uint8))
            self.log(f"Registered image saved as: {output_path}")
        except Exception as e:
            self.log(f"Error exporting registered image: {e}")

    # ----------------------------------------------------------------------
    # Deleting Points
    # ----------------------------------------------------------------------
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
