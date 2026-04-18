# Document Scanner — Computer Vision Project

## Introduction to Computer Vision — Spring 2026

**International University of Rabat (UIR) — ESIN**
**Professor:** Ilias TOUGUI

---

## Project Summary

A complete document scanning pipeline that:

1. **Detects** a document (book page, receipt, whiteboard, etc.) from an image
2. **Corrects its perspective** to obtain a flat top-down view
3. **Enhances the result** for improved readability
4. Provides an **interactive GUI** application

### Pipeline

```
Input Image → Grayscale → Gaussian Blur → Canny Edge Detection
→ Contour Detection → Corner Ordering → Perspective Warp → Enhancement → Output
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd document-scanner
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook

```bash
cd code
jupyter notebook document_scanner.ipynb
```

Then run all cells from top to bottom.

### 4. Run the GUI Application

```bash
cd code
python gui_scanner.py
```

- Click **"Load Image"** to select a document photo
- Click **"Scan Document"** to process it
- Adjust parameters (blur, Canny thresholds, enhancement) as needed
- Click **"Save Result"** to export the scanned output

---

## Project Structure

```
document-scanner/
├── data/
│   ├── README.txt              # Dataset description
│   ├── sample_document_1.png   # A4 document on desk
│   ├── sample_receipt.png      # Receipt on dark surface
│   └── sample_bookpage.png     # Book page
├── code/
│   ├── document_scanner.ipynb  # Main notebook (full pipeline + visualizations)
│   └── gui_scanner.py          # Standalone GUI application (Tkinter)
├── report.pdf                  # Project report (2–4 pages)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

> **Note:** The `output/` folder is created automatically when you run the pipeline. Add your own document photos to `data/` for testing.

---

## Requirements

- **Python 3.8+**
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Pillow (for GUI)
- Jupyter (for notebook)

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## Course Concepts Applied

| Technique              | Course Lecture                    | Usage                                 |
| ---------------------- | --------------------------------- | ------------------------------------- |
| Gaussian Blur          | Linear Image Filtering            | Noise reduction before edge detection |
| Canny Edge Detector    | Edge Detection (Szeliski 7.2.1)   | Finding document edges                |
| Contour Analysis       | Boundary Detection (Szeliski 7.3) | Detecting the document quadrilateral  |
| Histogram Equalization | Non-Linear Image Filtering        | Enhancing scanned output              |

### New Concepts (Self-Study)

- Perspective Transform / Homography
- Contour Approximation (Ramer-Douglas-Peucker)
- CLAHE (Contrast-Limited Adaptive Histogram Equalization)
- Adaptive Thresholding
- GUI Development with Tkinter

---

## GUI Features

- Dark theme interface
- File dialog to load any image
- Automatic document detection and scanning
- Adjustable parameters: Blur kernel, Canny thresholds, enhancement mode, gamma
- Side-by-side original vs. scanned comparison
- Save scanned output to disk

---

## Group Members

- Benmouma salma
- Gassi oumaima

---

## License

This project is part of the Introduction to Computer Vision course at UIR.
