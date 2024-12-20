import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import AffineTransform, warp
import numpy as np
from datetime import datetime


class ImageRegistrationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Registration Tool")

        # Initialize variables
        self.original_image = None
        self.transformed_image = None
        self.registered_image = None
        self.fixed_points = []
        self.moving_points = []

        # State for zooming and panning
        self.is_panning = False
        self.pan_start = None
        self.current_ax = None

        # Layout
        self.setup_ui()

    def setup_ui(self):
        # Frames for images
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        # Canvas for displaying images
        self.fig, self.axs = plt.subplots(1, 4, figsize=(16, 5))
        self.axs[0].set_title("EBSD (.ang) Image")
        self.axs[1].set_title("LRS Image")
        self.axs[2].set_title("Registered Image")
        self.axs[3].set_title("Superimposed Image")
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.get_tk_widget().pack()

        # Register click, zoom, and pan events
        self.cid_original = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_zoom = self.fig.canvas.mpl_connect('scroll_event', self.on_zoom)
        self.cid_pan_press = self.fig.canvas.mpl_connect('button_press_event', self.on_pan_press)
        self.cid_pan_release = self.fig.canvas.mpl_connect('button_release_event', self.on_pan_release)
        self.cid_pan_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_pan_motion)

        # Control panel
        self.control_frame = tk.Frame(self.root)
        self.control_frame.grid(row=1, column=0, columnspan=4, pady=10)

        tk.Button(self.control_frame, text="Load Original Image", command=self.load_original_image).grid(row=0, column=0)
        tk.Button(self.control_frame, text="Load Transformed Image", command=self.load_transformed_image).grid(row=0, column=1)
        tk.Button(self.control_frame, text="Register Images", command=self.register_images).grid(row=0, column=2)

        # Point editing
        self.edit_frame = tk.Frame(self.root)
        self.edit_frame.grid(row=2, column=0, columnspan=4, pady=10)

        tk.Label(self.edit_frame, text="Original Image Points:").grid(row=0, column=0, sticky="w")
        tk.Label(self.edit_frame, text="Transformed Image Points:").grid(row=0, column=1, sticky="w")

        self.original_points_listbox = tk.Listbox(self.edit_frame, width=30, height=10)
        self.original_points_listbox.grid(row=1, column=0, padx=5)

        self.transformed_points_listbox = tk.Listbox(self.edit_frame, width=30, height=10)
        self.transformed_points_listbox.grid(row=1, column=1, padx=5)

        tk.Button(self.edit_frame, text="Delete Selected Point (Original)", command=self.delete_original_point).grid(row=2, column=0)
        tk.Button(self.edit_frame, text="Delete Selected Point (Transformed)", command=self.delete_transformed_point).grid(row=2, column=1)

        # Log frame
        self.log_frame = tk.Frame(self.root)
        self.log_frame.grid(row=3, column=0, columnspan=4, pady=10)

        self.log_text = tk.Text(self.log_frame, width=100, height=5)
        self.log_text.grid(row=0, column=0, padx=10)

    def load_original_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
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

    def register_images(self):
        """
        Registers the transformed image to the original image using selected points.
        """
        if len(self.fixed_points) < 3 or len(self.moving_points) < 3:
            self.log("At least 3 points are required for registration.")
            return

        if len(self.fixed_points) != len(self.moving_points):
            self.log("The number of points in both images must be the same.")
            return

        fixed_points = np.array(self.fixed_points)
        moving_points = np.array(self.moving_points)

        # Estimate the affine transformation
        transform = AffineTransform()
        if not transform.estimate(moving_points, fixed_points):
            self.log("Affine transformation estimation failed.")
            return

        # Apply the affine transformation to register the transformed image
        self.registered_image = warp(
            self.transformed_image,
            transform.inverse,
            output_shape=self.original_image.shape,
            order=0,  # Use nearest-neighbor interpolation for sharpness
            preserve_range=True  # Preserve original pixel values
        ).astype(self.original_image.dtype)

        # Display the registered image in the third panel
        self.axs[2].imshow(self.registered_image, cmap='gray')
        self.axs[2].set_title("Registered Image")

        # Blend the original image and the registered image with 50% weightage for each
        blended_image = 0.5 * self.original_image + 0.5 * self.registered_image
        blended_image = np.clip(blended_image, 0, 1)  # Ensure pixel values are in the valid range [0, 1]
        self.axs[3].imshow(blended_image, cmap='gray')
        self.axs[3].set_title("Superimposed Image")
        self.canvas.draw()

        # Log the transformation parameters
        self.log_transformation(transform)

    def log_transformation(self, transform):
        scale_x = np.sqrt(transform.params[0, 0] ** 2 + transform.params[0, 1] ** 2)
        scale_y = np.sqrt(transform.params[1, 0] ** 2 + transform.params[1, 1] ** 2)
        rotation = np.arctan2(transform.params[1, 0], transform.params[0, 0]) * (180 / np.pi)
        translation = (transform.params[0, 2], transform.params[1, 2])

        self.log(f"Scaling (x, y): ({scale_x:.2f}, {scale_y:.2f})")
        self.log(f"Rotation (degrees): {rotation:.2f}")
        self.log(f"Translation (x, y): ({translation[0]:.2f}, {translation[1]:.2f})")

    def on_click(self, event):
        if event.inaxes == self.axs[0] and self.original_image is not None:
            x, y = event.xdata, event.ydata
            self.fixed_points.append((x, y))
            self.original_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
            self.axs[0].scatter(x, y, color='red', s=30)
            self.axs[0].text(x + 5, y, f"{len(self.fixed_points)}", color='red', fontsize=10)
            self.log(f"Point marked on Original Image: ({x:.1f}, {y:.1f})")
        elif event.inaxes == self.axs[1] and self.transformed_image is not None:
            x, y = event.xdata, event.ydata
            self.moving_points.append((x, y))
            self.transformed_points_listbox.insert(tk.END, f"({x:.1f}, {y:.1f})")
            self.axs[1].scatter(x, y, color='blue', s=30)
            self.axs[1].text(x + 5, y, f"{len(self.moving_points)}", color='blue', fontsize=10)
            self.log(f"Point marked on Transformed Image: ({x:.1f}, {y:.1f})")
        self.canvas.draw()

    def on_zoom(self, event):
        """
        Zoom in or out based on mouse pointer location.
        """
        ax = event.inaxes
        if ax in [self.axs[0], self.axs[1]]:  # Allow zooming only on specific axes
            zoom_factor = 0.9 if event.button == 'up' else 1.1
            x_mouse, y_mouse = event.xdata, event.ydata
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            x_range = (x_max - x_min) * zoom_factor
            y_range = (y_max - y_min) * zoom_factor

            ax.set_xlim(
                x_mouse - (x_mouse - x_min) * zoom_factor,
                x_mouse + (x_max - x_mouse) * zoom_factor
            )
            ax.set_ylim(
                y_mouse - (y_mouse - y_min) * zoom_factor,
                y_mouse + (y_max - y_mouse) * zoom_factor
            )
            self.canvas.draw()

    def on_pan_press(self, event):
        """
        Start panning when the mouse is pressed.
        """
        if event.inaxes in [self.axs[0], self.axs[1]]:
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)
            self.current_ax = event.inaxes

    def on_pan_release(self, event):
        """
        Stop panning when the mouse is released.
        """
        self.is_panning = False
        self.pan_start = None
        self.current_ax = None

    def on_pan_motion(self, event):
        """
        Perform panning when the mouse is dragged.
        """
        if self.is_panning and self.current_ax and event.xdata and event.ydata:
            x_start, y_start = self.pan_start
            dx = x_start - event.xdata
            dy = y_start - event.ydata

            x_min, x_max = self.current_ax.get_xlim()
            y_min, y_max = self.current_ax.get_ylim()

            self.current_ax.set_xlim(x_min + dx, x_max + dx)
            self.current_ax.set_ylim(y_min + dy, y_max + dy)
            self.pan_start = (event.xdata, event.ydata)

            self.canvas.draw()

    def delete_original_point(self):
        selected_index = self.original_points_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            del self.fixed_points[index]
            self.original_points_listbox.delete(index)
            self.redraw_points()

    def delete_transformed_point(self):
        selected_index = self.transformed_points_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            del self.moving_points[index]
            self.transformed_points_listbox.delete(index)
            self.redraw_points()

    def redraw_points(self):
        self.axs[0].clear()
        self.axs[0].imshow(self.original_image, cmap='gray')
        self.axs[0].set_title("Original Image")

        self.axs[1].clear()
        self.axs[1].imshow(self.transformed_image, cmap='gray')
        self.axs[1].set_title("Transformed Image")

        for i, (x, y) in enumerate(self.fixed_points):
            self.axs[0].scatter(x, y, color='red', s=30)
            self.axs[0].text(x + 5, y, f"{i + 1}", color='red', fontsize=10)

        for i, (x, y) in enumerate(self.moving_points):
            self.axs[1].scatter(x, y, color='blue', s=30)
            self.axs[1].text(x + 5, y, f"{i + 1}", color='blue', fontsize=10)

        self.canvas.draw()

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} INFO: {message}\n")
        self.log_text.see(tk.END)


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageRegistrationTool(root)
    root.mainloop()
