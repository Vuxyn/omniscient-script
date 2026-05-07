# Omniscient Script 🖼️✨

A lightweight Python library for Digital Image Processing focused on implementing image processing algorithms manually from scratch.

This project is designed for learning, experimentation, and understanding how image transformations actually work at the pixel level instead of relying heavily on high-level built-in processing functions.

---

## ✨ Features

### 🖼️ Image Operations

Implemented using direct matrix and pixel manipulation techniques:

* Grayscale conversion
* Image flipping
* Zoom in & zoom out
* Translation
* Rotation
* Basic pixel transformations

### 📊 Histogram Operations

* Grayscale histogram
* RGB histogram
* Histogram equalization
* Intensity distribution analysis

### ⚙️ Learning-Oriented Architecture

* Manual algorithm implementations
* Beginner-friendly code structure
* Easy to modify and extend
* Suitable for academic practice and experimentation

---

## 📂 Project Structure

```bash id="24xg8g"
omniscient-script/
│
├── general.py
├── image_operations.py
├── histogram_operations.py
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash id="f0d90c"
git clone https://github.com/Vuxyn/omniscient-script.git
```

Move into the project directory:

```bash id="v53bpd"
cd omniscient-script
```

Install dependencies:

```bash id="u2b2yl"
pip install numpy matplotlib
```

---

## ▶️ Usage Example

```python id="7k1i3o"
from image_operations import *

image = read_image('image.jpg')

result = grayscale(image)

show_image(result)
```

---

## 🧠 Project Goals

This project aims to:

* Understand image processing algorithms internally
* Learn pixel-based image manipulation
* Build image operations from scratch
* Explore matrix-based transformations
* Support Digital Image Processing education and research

---

## 🛠️ Technologies Used

* Python
* NumPy
* Matplotlib

---

## 📚 Concepts Covered

* Digital image representation
* Pixel manipulation
* Geometric transformations
* Histogram processing
* Histogram equalization
* Spatial image operations

---

## 🌱 Future Improvements

Planned features include:

* Thresholding
* Edge detection
* Morphological operations
* Image filtering
* Noise reduction
* GUI support
* Video processing

---

## 👤 Author

Built with ☕, matrices, and several catastrophic debugging arcs by Ara.

GitHub: [Vuxyn](https://github.com/Vuxyn?utm_source=chatgpt.com)
