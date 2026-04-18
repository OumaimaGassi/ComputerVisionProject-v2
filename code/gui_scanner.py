"""
Document Scanner — Interactive GUI Application
================================================
Course: Introduction to Computer Vision — Spring 2026
University: International University of Rabat (UIR) — ESIN
Professor: Ilias TOUGUI

This standalone Tkinter application wraps the document scanning pipeline
and provides an interactive interface for:
- Loading document images
- Automatic document detection and perspective correction
- Adjustable parameters (blur, Canny thresholds, enhancement, gamma)
- Side-by-side original vs scanned comparison
- Saving scanned output

Usage:
    python gui_scanner.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os


# ============================================================
# Document Scanner Pipeline Functions
# ============================================================

def preprocess_image(image, blur_kernel_size=5):
    """
    Convert image to grayscale and apply Gaussian blur.
    
    Course Concept: Linear Image Filtering (Gaussian blur)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    return gray, blurred


def detect_edges(blurred_image, low_threshold=50, high_threshold=150):
    """
    Apply Canny edge detection.
    
    Course Concept: Edge Detection (Szeliski 7.2.1)
    """
    return cv2.Canny(blurred_image, low_threshold, high_threshold)


def find_document_contour(edges, image_shape, min_area_ratio=0.05):
    """
    Find the largest quadrilateral contour.
    
    Course Concept: Boundary Detection (Szeliski 7.3)
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    image_area = image_shape[0] * image_shape[1]
    min_area = image_area * min_area_ratio
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx
    return None


def order_points(pts):
    """Order points as [TL, TR, BR, BL]."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def perspective_warp(image, ordered_pts):
    """
    Apply perspective transform to get top-down view.
    
    New Concept: Perspective Transform / Homography
    """
    tl, tr, br, bl = ordered_pts
    max_width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    max_height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    
    dst_pts = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    return cv2.warpPerspective(image, M, (max_width, max_height))


def enhance_image(image, method='adaptive', gamma=0.7):
    """Apply image enhancement."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if method == 'adaptive':
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    elif method == 'equalize':
        return cv2.equalizeHist(gray)
    elif method == 'gamma':
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(image, table)
    elif method == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    else:
        return gray


def scan_document(image, blur_kernel=5, canny_low=50, canny_high=150,
                  enhancement='adaptive', gamma=0.7):
    """Complete document scanning pipeline."""
    gray, blurred = preprocess_image(image, blur_kernel)
    edges = detect_edges(blurred, canny_low, canny_high)
    contour = find_document_contour(edges, image.shape[:2])
    
    if contour is None:
        return None, None, None
    
    ordered_pts = order_points(contour)
    warped = perspective_warp(image, ordered_pts)
    enhanced = enhance_image(warped, enhancement, gamma)
    
    return enhanced, warped, ordered_pts


# ============================================================
# GUI Application
# ============================================================

class DocumentScannerApp:
    """Interactive Document Scanner GUI using Tkinter."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("📄 Document Scanner — UIR Computer Vision Project")
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Colors
        self.bg_color = "#1e1e2e"
        self.accent = "#89b4fa"
        self.text_color = "#cdd6f4"
        self.surface = "#313244"
        self.green = "#a6e3a1"
        self.red = "#f38ba8"
        
        self.root.configure(bg=self.bg_color)
        
        # State
        self.original_image = None
        self.scanned_image = None
        self.warped_image = None
        self.file_path = None
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the application UI."""
        # ==========================================
        # Top Bar
        # ==========================================
        top_frame = tk.Frame(self.root, bg=self.surface, height=60)
        top_frame.pack(fill='x', padx=0, pady=0)
        top_frame.pack_propagate(False)
        
        title_label = tk.Label(
            top_frame, text="📄 Document Scanner",
            font=("Segoe UI", 18, "bold"),
            fg=self.accent, bg=self.surface
        )
        title_label.pack(side='left', padx=20, pady=10)
        
        subtitle = tk.Label(
            top_frame, text="Introduction to Computer Vision — UIR 2026",
            font=("Segoe UI", 10),
            fg=self.text_color, bg=self.surface
        )
        subtitle.pack(side='left', padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            top_frame, text="Ready — Load an image to start",
            font=("Segoe UI", 10),
            fg=self.green, bg=self.surface
        )
        self.status_label.pack(side='right', padx=20, pady=10)
        
        # ==========================================
        # Main Content Area
        # ==========================================
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left: Controls Panel
        controls_frame = tk.Frame(main_frame, bg=self.surface, width=280)
        controls_frame.pack(side='left', fill='y', padx=(0, 10))
        controls_frame.pack_propagate(False)
        
        self._build_controls(controls_frame)
        
        # Right: Image Display Area
        display_frame = tk.Frame(main_frame, bg=self.bg_color)
        display_frame.pack(side='right', fill='both', expand=True)
        
        self._build_display(display_frame)
    
    def _build_controls(self, parent):
        """Build the controls panel."""
        pad = {'padx': 15, 'pady': 5}
        
        # Section: File Operations
        tk.Label(parent, text="📂 File Operations", font=("Segoe UI", 12, "bold"),
                fg=self.accent, bg=self.surface).pack(**pad, anchor='w', pady=(15, 5))
        
        btn_load = tk.Button(
            parent, text="📂  Load Image", command=self.load_image,
            font=("Segoe UI", 11), bg="#45475a", fg=self.text_color,
            activebackground=self.accent, activeforeground="#000",
            relief='flat', cursor='hand2', width=22
        )
        btn_load.pack(**pad)
        
        btn_scan = tk.Button(
            parent, text="🔍  Scan Document", command=self.scan,
            font=("Segoe UI", 11, "bold"), bg=self.accent, fg="#000",
            activebackground="#74c7ec", relief='flat', cursor='hand2', width=22
        )
        btn_scan.pack(**pad)
        
        btn_save = tk.Button(
            parent, text="💾  Save Result", command=self.save_result,
            font=("Segoe UI", 11), bg="#45475a", fg=self.text_color,
            activebackground=self.green, activeforeground="#000",
            relief='flat', cursor='hand2', width=22
        )
        btn_save.pack(**pad)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=15, pady=10)
        
        # Section: Parameters
        tk.Label(parent, text="🎛️ Parameters", font=("Segoe UI", 12, "bold"),
                fg=self.accent, bg=self.surface).pack(**pad, anchor='w')
        
        # Blur kernel
        tk.Label(parent, text="Gaussian Blur Kernel:", font=("Segoe UI", 9),
                fg=self.text_color, bg=self.surface).pack(**pad, anchor='w')
        self.blur_var = tk.IntVar(value=5)
        blur_scale = tk.Scale(parent, from_=1, to=15, orient='horizontal',
                             variable=self.blur_var, resolution=2,
                             bg=self.surface, fg=self.text_color,
                             troughcolor="#45475a", highlightthickness=0)
        blur_scale.pack(fill='x', **pad)
        
        # Canny Low
        tk.Label(parent, text="Canny Low Threshold:", font=("Segoe UI", 9),
                fg=self.text_color, bg=self.surface).pack(**pad, anchor='w')
        self.canny_low_var = tk.IntVar(value=50)
        canny_low_scale = tk.Scale(parent, from_=10, to=200, orient='horizontal',
                                   variable=self.canny_low_var,
                                   bg=self.surface, fg=self.text_color,
                                   troughcolor="#45475a", highlightthickness=0)
        canny_low_scale.pack(fill='x', **pad)
        
        # Canny High
        tk.Label(parent, text="Canny High Threshold:", font=("Segoe UI", 9),
                fg=self.text_color, bg=self.surface).pack(**pad, anchor='w')
        self.canny_high_var = tk.IntVar(value=150)
        canny_high_scale = tk.Scale(parent, from_=50, to=300, orient='horizontal',
                                    variable=self.canny_high_var,
                                    bg=self.surface, fg=self.text_color,
                                    troughcolor="#45475a", highlightthickness=0)
        canny_high_scale.pack(fill='x', **pad)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').pack(fill='x', padx=15, pady=10)
        
        # Enhancement method
        tk.Label(parent, text="🎨 Enhancement", font=("Segoe UI", 12, "bold"),
                fg=self.accent, bg=self.surface).pack(**pad, anchor='w')
        
        self.enhance_var = tk.StringVar(value='adaptive')
        enhancements = [
            ('Adaptive Threshold', 'adaptive'),
            ('CLAHE', 'clahe'),
            ('Histogram Equalization', 'equalize'),
            ('Gamma Correction', 'gamma'),
            ('Sharpen', 'sharpen'),
            ('None (Grayscale)', 'none')
        ]
        for text, value in enhancements:
            rb = tk.Radiobutton(
                parent, text=text, variable=self.enhance_var, value=value,
                font=("Segoe UI", 9), fg=self.text_color, bg=self.surface,
                selectcolor="#45475a", activebackground=self.surface,
                activeforeground=self.accent
            )
            rb.pack(anchor='w', padx=20)
        
        # Gamma slider
        tk.Label(parent, text="Gamma Value:", font=("Segoe UI", 9),
                fg=self.text_color, bg=self.surface).pack(**pad, anchor='w')
        self.gamma_var = tk.DoubleVar(value=0.7)
        gamma_scale = tk.Scale(parent, from_=0.1, to=3.0, orient='horizontal',
                              variable=self.gamma_var, resolution=0.1,
                              bg=self.surface, fg=self.text_color,
                              troughcolor="#45475a", highlightthickness=0)
        gamma_scale.pack(fill='x', **pad)
    
    def _build_display(self, parent):
        """Build the image display area."""
        # Labels frame
        labels_frame = tk.Frame(parent, bg=self.bg_color)
        labels_frame.pack(fill='x')
        
        tk.Label(labels_frame, text="📷 Original Image",
                font=("Segoe UI", 13, "bold"),
                fg=self.text_color, bg=self.bg_color).pack(side='left', expand=True)
        tk.Label(labels_frame, text="📄 Scanned Document",
                font=("Segoe UI", 13, "bold"),
                fg=self.green, bg=self.bg_color).pack(side='right', expand=True)
        
        # Image canvases
        canvas_frame = tk.Frame(parent, bg=self.bg_color)
        canvas_frame.pack(fill='both', expand=True, pady=5)
        
        self.canvas_original = tk.Canvas(canvas_frame, bg="#181825",
                                          highlightthickness=1,
                                          highlightbackground="#45475a")
        self.canvas_original.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.canvas_scanned = tk.Canvas(canvas_frame, bg="#181825",
                                         highlightthickness=1,
                                         highlightbackground="#45475a")
        self.canvas_scanned.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Info label at bottom
        self.info_label = tk.Label(
            parent, text="Load an image and click 'Scan Document' to begin",
            font=("Segoe UI", 10, "italic"),
            fg="#6c7086", bg=self.bg_color
        )
        self.info_label.pack(pady=5)
    
    def load_image(self):
        """Open file dialog and load an image."""
        file_path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        self.file_path = file_path
        self.original_image = cv2.imread(file_path)
        
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not read image: {file_path}")
            return
        
        self._display_image(self.original_image, self.canvas_original)
        self.canvas_scanned.delete("all")
        self.scanned_image = None
        
        h, w = self.original_image.shape[:2]
        name = os.path.basename(file_path)
        self.status_label.config(text=f"✅ Loaded: {name}", fg=self.green)
        self.info_label.config(text=f"Image: {name} ({w}×{h}) — Click 'Scan Document' to process")
    
    def scan(self):
        """Run the document scanning pipeline."""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        self.status_label.config(text="🔍 Scanning...", fg="#f9e2af")
        self.root.update()
        
        blur_k = self.blur_var.get()
        if blur_k % 2 == 0:
            blur_k += 1
        
        enhanced, warped, corners = scan_document(
            self.original_image,
            blur_kernel=blur_k,
            canny_low=self.canny_low_var.get(),
            canny_high=self.canny_high_var.get(),
            enhancement=self.enhance_var.get(),
            gamma=self.gamma_var.get()
        )
        
        if enhanced is None:
            self.status_label.config(text="❌ No document found!", fg=self.red)
            self.info_label.config(text="Could not detect a document. Try adjusting parameters.")
            messagebox.showinfo("Result", 
                "No document was found in the image.\n\n"
                "Tips:\n"
                "- Lower the Canny thresholds\n"
                "- Increase the blur kernel\n"
                "- Make sure the document corners are visible")
            return
        
        self.scanned_image = enhanced
        self.warped_image = warped
        
        # Draw contour on original for display
        display_original = self.original_image.copy()
        contour_pts = corners.astype(int)
        for i in range(4):
            p1 = tuple(contour_pts[i])
            p2 = tuple(contour_pts[(i + 1) % 4])
            cv2.line(display_original, p1, p2, (0, 255, 0), 3)
            cv2.circle(display_original, p1, 8, (0, 0, 255), -1)
        
        self._display_image(display_original, self.canvas_original)
        self._display_image(enhanced, self.canvas_scanned)
        
        self.status_label.config(text="✅ Scan complete!", fg=self.green)
        
        if enhanced is not None:
            h, w = enhanced.shape[:2] if len(enhanced.shape) == 2 else enhanced.shape[:2]
            self.info_label.config(
                text=f"Scan successful! Output: {w}×{h} — Enhancement: {self.enhance_var.get()}"
            )
    
    def save_result(self):
        """Save the scanned document."""
        if self.scanned_image is None:
            messagebox.showwarning("Warning", "No scanned document to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Scanned Document",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.scanned_image)
            self.status_label.config(text=f"💾 Saved: {os.path.basename(file_path)}", fg=self.green)
            messagebox.showinfo("Success", f"Scanned document saved to:\n{file_path}")
    
    def _display_image(self, image, canvas):
        """Display an OpenCV image on a Tkinter canvas."""
        canvas.delete("all")
        canvas.update()
        
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w, canvas_h = 500, 600
        
        # Convert to RGB
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        h, w = img_rgb.shape[:2]
        scale = min(canvas_w / w, canvas_h / h) * 0.95
        new_w, new_h = int(w * scale), int(h * scale)
        
        if new_w > 0 and new_h > 0:
            img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img_rgb
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display centered
        x = canvas_w // 2
        y = canvas_h // 2
        canvas.create_image(x, y, image=photo, anchor='center')
        canvas._photo = photo  # Keep reference


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentScannerApp(root)
    root.mainloop()
