# Image Registration Tool

## Overview
This project is an interactive **Image Registration Tool** built using Python and Tkinter. It allows users to:

1. Load two images with transformations (e.g., shift, scale, rotation).
2. Mark corresponding points interactively on both images.
3. View and adjust the coordinates of marked points through text boxes.
4. Calculate and apply the transformation matrix to align the second image to the first.
5. Display the combined result for visual verification of alignment.

## Features

- **Interactive Point Selection**: Users can click on images to mark corresponding points.
- **Adjustable Coordinates**: Marked points can be fine-tuned via text boxes.
- **Zoom and Reset**: Zoom in/out functionality and a reset-to-home view for precision.
- **Transformation Calculation**: Computes shift, rotation, scaling, and the transformation matrix using selected points.
- **Result Display**: Shows the blended alignment of the two images for visual feedback.
- **Logging**: Real-time logging of user actions and computed transformations.

## Requirements

- Python 3.7+
- Required libraries:
  - `opencv-python`
  - `Pillow`
  - `numpy`

Install the required libraries using:
```bash
pip install opencv-python Pillow numpy
```

## File Structure

```
image_registration_tool/
├── main.py                 # Entry point for the application
├── gui.py                  # Contains the ImageRegistrationApp class
├── test_image_generator.py # Utility to generate test images
└── README.md               # Documentation
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/image_registration_tool.git
   cd image_registration_tool
   ```
2. Run the application:
   ```bash
   python main.py
   ```

## Usage

1. **Load Images**:
   - The application starts with two test images.
   - The second image is generated by applying a known transformation (shift, scale, rotation) to the first.

2. **Mark Points**:
   - Click on the canvases to mark corresponding points on both images.
   - A minimum of 4 points is required for registration.

3. **Adjust Coordinates**:
   - Use the text boxes to fine-tune the coordinates if necessary.

4. **Zoom and Reset**:
   - Use the "Zoom" button to activate zoom mode.
   - Use the "Home" button to reset the view to the original scale.

5. **Register Images**:
   - Once at least 4 points are marked on both images, the "Register Images" button becomes active.
   - Click it to calculate the transformation matrix and display the aligned result.

## Example Output

- **Input Images**: Two images with transformations applied.
- **Registered Result**: Blended visualization of the alignment accuracy.

## Logging

- Logs important messages, warnings, and computed transformations.
- Examples:
  - `INFO: Point 1 marked on Image 1: (120, 150)`
  - `WARNING: Point 2 out of bounds (550, 600)`
  - `INFO: Calculated Shift: [50, 30]`
  - `INFO: Calculated Rotation: 15 degrees`

## Future Enhancements

- Add support for loading custom images.
- Implement advanced transformation models (e.g., perspective transformations).
- Add a zoom glass for precision marking.
- Save the transformation matrix for later use.

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

