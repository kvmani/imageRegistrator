import os
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image


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
        #self.save_image()

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

        expected_size = nrows * ncols_odd
        current_size = len(self.data[5])

        if current_size < expected_size:
            nrows=nrows-1
            self.header["NROWS"]=nrows
            print(f"warning!!! I am reducing the rows from {nrows+1} to {nrows}")
            assert nrows*ncols_odd ==  current_size, "not matching even when rows are reduced by 1 !!!"

        elif current_size != expected_size:
            warnings.warn(
                f"Data size mismatch: IQ values do not match specified grid dimensions. "
                f"expected: {expected_size} got: {current_size}"
            )
        else:
            print (" Data frame size matches with the ang data rows.!!! ALL oK.")
        # if len(self.data[5]) != nrows * ncols_odd:
        #     #raise ValueError(f"Data size mismatch: IQ values do not match specified grid dimensions. expected : {nrows * ncols_odd} got : {self.data[5]}")
        #     warnings.warn(f"Data size mismatch: IQ values do not match specified grid dimensions. expected : {nrows * ncols_odd} got : {self.data[5]}")

        self.image =Image.fromarray(self.data[5].values.reshape(nrows, ncols_odd))
        print("EBSD IQ image generated successfully.")


    def save_image(self):
        if self.image is not None:
            output_path = os.path.join(self.output_folder, "ImageAngNi.png")
            norm_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(output_path, norm_image)
            print(f"Image saved to: {output_path}")
        else:
            print("No image generated to save.")
