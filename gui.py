import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from test_image_generator import TestImageGenerator
import numpy as np
from datetime import datetime


class ImageRegistrationApp:
    def __init__(self, root, n_points=5):
        self.root = root
        self.root.title("Image Registration Tool")
        self.n_points = n_points

        # Frames
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas for images
        self.canvas1 = tk.Canvas(self.canvas_frame, width=500, height=500, bg="white")
        self.canvas1.pack(side=tk.LEFT, padx=5, pady=5)

        self.canvas2 = tk.Canvas(self.canvas_frame, width=500, height=500, bg="white")
        self.canvas2.pack(side=tk.LEFT, padx=5, pady=5)

        self.result_canvas = tk.Canvas(self.canvas_frame, width=500, height=500, bg="white")
        self.result_canvas.pack(side=tk.LEFT, padx=5, pady=5)

        # Coordinate input
        self.coord_frame1 = tk.Frame(self.control_frame)
        self.coord_frame1.pack(side=tk.LEFT, padx=10)
        self.coord_labels1 = []
        self.coord_entries1 = []

        self.coord_frame2 = tk.Frame(self.control_frame)
        self.coord_frame2.pack(side=tk.RIGHT, padx=10)
        self.coord_labels2 = []
        self.coord_entries2 = []

        for i in range(self.n_points):
            label1 = tk.Label(self.coord_frame1, text=f"Point {i + 1}:")
            label1.grid(row=i, column=0, padx=5)

            x_entry1 = tk.Entry(self.coord_frame1, width=5)
            x_entry1.grid(row=i, column=1, padx=5)

            y_entry1 = tk.Entry(self.coord_frame1, width=5)
            y_entry1.grid(row=i, column=2, padx=5)

            self.coord_labels1.append(label1)
            self.coord_entries1.append((x_entry1, y_entry1))

            label2 = tk.Label(self.coord_frame2, text=f"Point {i + 1}:")
            label2.grid(row=i, column=0, padx=5)

            x_entry2 = tk.Entry(self.coord_frame2, width=5)
            x_entry2.grid(row=i, column=1, padx=5)

            y_entry2 = tk.Entry(self.coord_frame2, width=5)
            y_entry2.grid(row=i, column=2, padx=5)

            self.coord_labels2.append(label2)
            self.coord_entries2.append((x_entry2, y_entry2))

        # Register button (disabled initially)
        self.register_button = tk.Button(self.control_frame, text="Register Images", command=self.register_images,
                                         state=tk.DISABLED)
        self.register_button.pack(pady=10)

        # Log box
        self.log_box = tk.Text(self.control_frame, height=5, wrap=tk.WORD)
        self.log_box.pack(fill=tk.X, padx=10, pady=10)

        # Attributes for storing images and points
        self.image1 = None
        self.image2 = None
        self.points1 = []
        self.points2 = []
        self.zoom_factor1 = 1.0  # Zoom factor for canvas1
        self.zoom_factor2 = 1.0  # Zoom factor for canvas2
        self.offset_x1 = 0
        self.offset_y1 = 0
        self.offset_x2 = 0
        self.offset_y2 = 0

        self.load_test_images()
        self.display_images()

        self.canvas1.bind("<Button-1>", self.mark_point_image1)
        self.canvas2.bind("<Button-1>", self.mark_point_image2)
        self.canvas1.bind("<MouseWheel>", self.zoom_canvas1)
        self.canvas2.bind("<MouseWheel>", self.zoom_canvas2)

        self.setup_entry_bindings()

    def load_test_images(self):
        """Generate test images."""
        generator = TestImageGenerator()
        self.image1 = generator.create_base_image()
        self.image2 = generator.apply_transformation(self.image1, scale=0.9, rotation=15, shift=(50, 30))

    def display_images(self):
        """Display images on the canvases."""
        img1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)))
        img2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)))

        self.canvas1.image = img1
        self.canvas1.create_image(0, 0, anchor=tk.NW, image=img1)

        self.canvas2.image = img2
        self.canvas2.create_image(0, 0, anchor=tk.NW, image=img2)

    def zoom_canvas1(self, event):
        """Handle zooming on Canvas 1."""
        self.zoom_factor1, self.offset_x1, self.offset_y1 = self.zoom_image(event, self.canvas1, self.image1,
                                                                            self.points1, "red", self.zoom_factor1,
                                                                            self.offset_x1, self.offset_y1)

    def zoom_canvas2(self, event):
        """Handle zooming on Canvas 2."""
        self.zoom_factor2, self.offset_x2, self.offset_y2 = self.zoom_image(event, self.canvas2, self.image2,
                                                                            self.points2, "blue", self.zoom_factor2,
                                                                            self.offset_x2, self.offset_y2)

    def zoom_image(self, event, canvas, image, points, color, zoom_factor, offset_x, offset_y):
        """Zoom in or out on the given canvas, centering on the mouse position."""
        scale_factor = 1.1 if event.delta > 0 else 0.9  # Zoom in or out
        new_zoom_factor = zoom_factor * scale_factor

        # Calculate mouse position relative to the current canvas view
        canvas_mouse_x = canvas.canvasx(event.x)
        canvas_mouse_y = canvas.canvasy(event.y)

        # Adjust offsets to keep zoom centered around the mouse
        offset_x = (offset_x + canvas_mouse_x) * scale_factor - canvas_mouse_x
        offset_y = (offset_y + canvas_mouse_y) * scale_factor - canvas_mouse_y

        # Clear canvas and redraw zoomed image
        canvas.delete("all")
        zoomed_image = cv2.resize(image, None, fx=new_zoom_factor, fy=new_zoom_factor, interpolation=cv2.INTER_LINEAR)
        img_display = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2RGB)))
        canvas.image = img_display
        canvas.create_image(-offset_x, -offset_y, anchor=tk.NW, image=img_display)

        # Recalculate and redraw markers
        for i, (x, y) in enumerate(points):
            scaled_x = int(x * new_zoom_factor - offset_x)
            scaled_y = int(y * new_zoom_factor - offset_y)
            canvas.create_oval(scaled_x - 3, scaled_y - 3, scaled_x + 3, scaled_y + 3, fill=color)
            canvas.create_text(scaled_x + 10, scaled_y, text=str(i + 1), fill=color)

        return new_zoom_factor, offset_x, offset_y

    def mark_point_image1(self, event):
        """Mark a point on image1 and log it."""
        x, y = event.x, event.y
        x = (x + self.offset_x1) / self.zoom_factor1
        y = (y + self.offset_y1) / self.zoom_factor1
        if len(self.points1) < self.n_points:
            self.points1.append((x, y))
            self.canvas1.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="red")
            self.canvas1.create_text(event.x + 10, event.y, text=str(len(self.points1)), fill="red")
            self.coord_entries1[len(self.points1) - 1][0].delete(0, tk.END)
            self.coord_entries1[len(self.points1) - 1][0].insert(0, str(int(x)))
            self.coord_entries1[len(self.points1) - 1][1].delete(0, tk.END)
            self.coord_entries1[len(self.points1) - 1][1].insert(0, str(int(y)))
            self.log_message(f"Point {len(self.points1)} marked on Image 1: ({x}, {y})")
            self.check_register_button()

    def mark_point_image2(self, event):
        """Mark a point on image2 and log it."""
        x, y = event.x, event.y
        x = (x + self.offset_x2) / self.zoom_factor2
        y = (y + self.offset_y2) / self.zoom_factor2
        if len(self.points2) < self.n_points:
            self.points2.append((x, y))
            self.canvas2.create_oval(event.x - 3, event.y - 3, event.x + 3, event.y + 3, fill="blue")
            self.canvas2.create_text(event.x + 10, event.y, text=str(len(self.points2)), fill="blue")
            self.coord_entries2[len(self.points2) - 1][0].delete(0, tk.END)
            self.coord_entries2[len(self.points2) - 1][0].insert(0, str(int(x)))
            self.coord_entries2[len(self.points2) - 1][1].delete(0, tk.END)
            self.coord_entries2[len(self.points2) - 1][1].insert(0, str(int(y)))
            self.log_message(f"Point {len(self.points2)} marked on Image 2: ({x}, {y})")
            self.check_register_button()

    def check_register_button(self):
        """Enable the register button if enough points are marked."""
        if len(self.points1) >= 4 and len(self.points2) >= 4:
            self.register_button.config(state=tk.NORMAL)

    def sync_points_from_entry(self, entry_list, points, canvas, color):
        """Synchronize marker positions on canvas with text box values."""
        canvas.delete("all")
        self.display_images()
        for i, (x_entry, y_entry) in enumerate(entry_list):
            try:
                x = int(x_entry.get())
                y = int(y_entry.get())
                if 0 <= x < 500 and 0 <= y < 500:
                    points[i] = (x, y)
                    canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color)
                    canvas.create_text(x + 10, y, text=str(i + 1), fill=color)
                else:
                    self.log_message(f"Warning: Point {i + 1} out of bounds ({x}, {y})", level="warning")
            except ValueError:
                self.log_message(f"Warning: Invalid value at Point {i + 1}. Ignoring.", level="warning")

    def setup_entry_bindings(self):
        """Bind text box events to synchronize markers."""
        for i, (x_entry1, y_entry1) in enumerate(self.coord_entries1):
            x_entry1.bind("<FocusOut>",
                          lambda e, idx=i: self.sync_points_from_entry(self.coord_entries1, self.points1, self.canvas1,
                                                                       "red"))
            y_entry1.bind("<FocusOut>",
                          lambda e, idx=i: self.sync_points_from_entry(self.coord_entries1, self.points1, self.canvas1,
                                                                       "red"))

        for i, (x_entry2, y_entry2) in enumerate(self.coord_entries2):
            x_entry2.bind("<FocusOut>",
                          lambda e, idx=i: self.sync_points_from_entry(self.coord_entries2, self.points2, self.canvas2,
                                                                       "blue"))
            y_entry2.bind("<FocusOut>",
                          lambda e, idx=i: self.sync_points_from_entry(self.coord_entries2, self.points2, self.canvas2,
                                                                       "blue"))

    def register_images(self):
        """Calculate and log the transformation matrix."""
        if len(self.points1) < 4 or len(self.points2) < 4:
            self.log_message("Error: At least 4 points are required on both images.", level="warning")
            return

        points1 = np.array(self.points1[:4], dtype=np.float32)
        points2 = np.array(self.points2[:4], dtype=np.float32)

        matrix, _ = cv2.estimateAffinePartial2D(points2, points1)

        # Calculate transformation components
        shift = matrix[:, 2]
        rotation_scale_matrix = matrix[:2, :2]
        scale = np.linalg.norm(rotation_scale_matrix[0])
        rotation = np.arctan2(rotation_scale_matrix[1, 0], rotation_scale_matrix[0, 0]) * (180 / np.pi)

        self.log_message(f"Calculated Shift: {shift}")
        self.log_message(f"Calculated Scaling: {scale}")
        self.log_message(f"Calculated Rotation: {rotation} degrees")
        self.log_message(f"Transformation Matrix:\n{matrix}")

        self.apply_transformation(matrix)

    def apply_transformation(self, matrix):
        """Apply the calculated transformation to Image 2 and display results."""
        rows, cols, _ = self.image1.shape
        transformed = cv2.warpAffine(self.image2, matrix, (cols, rows))
        blended = cv2.addWeighted(self.image1, 0.5, transformed, 0.5, 0)

        result_img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)))
        self.result_canvas.image = result_img
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=result_img)
        self.log_message("Images successfully registered and displayed.")

    def log_message(self, message, level="info"):
        """Log a message to the text box with a timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if level == "warning":
            self.log_box.insert(tk.END, f"{timestamp} WARNING: {message}\n", ("warning",))
            self.log_box.tag_config("warning", foreground="red")
        else:
            self.log_box.insert(tk.END, f"{timestamp} INFO: {message}\n")
        self.log_box.see(tk.END)
