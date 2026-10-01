"""
YOLO Re-labeling Tool
---------------------
Hotkeys:
  A / D         — prev / next image
  S             — save current labels
  N             — new bbox (draw mode)
  Del / Backspace — delete selected bbox
  Escape        — deselect / cancel draw
  R             — run YOLO predictions again
  H             — toggle prediction confidence labels
  F             — fit image (reset zoom)
  Ctrl+Z        — undo last action
  0-9           — change class of selected bbox

Mouse:
  Scroll wheel  — zoom in / out (centered on cursor)
  Middle drag   — pan image
  Left click    — select bbox
  Left drag on bbox — move bbox
  Left drag on handle — resize bbox
  Left drag on empty — draw new bbox
  Right click   — delete bbox under cursor

Color coding:
  Cyan dashed   — YOLO predictions (not yet confirmed)
  Green solid   — confirmed / manually drawn boxes
"""

import os
import sys
import glob
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ─── Config ────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HANDLE_SIZE = 8          # px, resize handle half-size
SNAP_DIST   = 12         # px, click-to-edge snapping distance
MIN_BOX     = 10         # px minimum box size

COLORS = {
    "predicted": "#00CFFF",   # cyan  — YOLO prediction
    "confirmed": "#00FF7F",   # green — confirmed / manual
    "selected":  "#FF4040",   # red   — selected state
    "handle":    "#FFFF00",   # yellow handles
}

# ─── Data structures ───────────────────────────────────────────────────────────

class BBox:
    """Normalized YOLO bbox: cx, cy, w, h in [0,1]"""
    _id_counter = 0

    def __init__(self, cx, cy, w, h, cls=0, confirmed=True):
        BBox._id_counter += 1
        self.id        = BBox._id_counter
        self.cx        = cx
        self.cy        = cy
        self.w         = w
        self.h         = h
        self.cls       = cls
        self.confirmed = confirmed   # False = still shown as YOLO prediction

    def to_pixel(self, iw, ih):
        """Return (x1, y1, x2, y2) in pixel space"""
        x1 = (self.cx - self.w / 2) * iw
        y1 = (self.cy - self.h / 2) * ih
        x2 = (self.cx + self.w / 2) * iw
        y2 = (self.cy + self.h / 2) * ih
        return x1, y1, x2, y2

    def from_pixel(self, x1, y1, x2, y2, iw, ih):
        self.cx = ((x1 + x2) / 2) / iw
        self.cy = ((y1 + y2) / 2) / ih
        self.w  = abs(x2 - x1) / iw
        self.h  = abs(y2 - y1) / ih

    def to_yolo_line(self):
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"

    def clone(self):
        return BBox(self.cx, self.cy, self.w, self.h, self.cls, self.confirmed)

# ─── Main App ──────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, images_dir=None, model_path=None, class_names=None):
        super().__init__()

        self.title("YOLO Labeler")
        self.configure(bg="#1a1a2e")
        self.geometry("1400x900")
        self.minsize(900, 600)

        # ── state ──
        self.images_dir   = images_dir
        self.image_paths  = []
        self.current_idx  = 0
        self.bboxes       = []          # list[BBox] for current image
        self.is_dirty     = False        # unsaved changes
        self.undo_stack   = []            # list of bbox snapshots
        self.selected_id  = None
        self.draw_mode    = False
        self.draw_start   = None
        self.drag_op      = None        # {"type": "move"|"resize", ...}
        self.show_conf    = True
        self.conf_thresh  = 0.25
        self.class_names  = class_names or ["digit"]

        # YOLO model
        self.model        = None
        self.model_path   = model_path
        if model_path and YOLO_AVAILABLE:
            self._load_model(model_path)

        # image display
        self.img_pil      = None
        self.img_tk       = None
        self.canvas_offset = (0, 0)     # top-left of image on canvas
        self.img_scale    = 1.0

        # zoom
        self.zoom_level   = 1.0         # user zoom multiplier (1.0 = fit)
        self.zoom_offset  = [0, 0]      # pan offset in image pixels
        self.ZOOM_STEP    = 1.25
        self.ZOOM_MAX     = 32.0
        self.ZOOM_MIN     = 0.1
        self._pan_last    = None        # for middle-button pan

        self._build_ui()
        self._bind_keys()

        if images_dir:
            self._load_folder(images_dir)

    # ── UI build ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── top toolbar ──
        toolbar = tk.Frame(self, bg="#16213e", pady=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_style = {"bg": "#0f3460", "fg": "white", "relief": "flat",
                     "padx": 10, "pady": 4, "font": ("Segoe UI", 9),
                     "activebackground": "#533483", "activeforeground": "white",
                     "cursor": "hand2"}

        tk.Button(toolbar, text="📂 Open Folder", command=self._open_folder, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="🤖 Load Model",  command=self._open_model,  **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="▶ Run YOLO (R)", command=self._run_yolo,    **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="✚ New Box (N)",  command=self._toggle_draw, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="💾 Save (S)",    command=self._save_labels, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="⟳ Fit (F)",      command=self._zoom_reset,  **btn_style).pack(side=tk.LEFT, padx=4)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        self.zoom_label_var = tk.StringVar(value="100%")
        tk.Label(toolbar, textvariable=self.zoom_label_var, bg="#16213e", fg="#00CFFF",
                 font=("Consolas", 9, "bold"), width=6).pack(side=tk.LEFT)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        tk.Label(toolbar, text="Conf:", bg="#16213e", fg="#aaa",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=self.conf_thresh)
        tk.Scale(toolbar, variable=self.conf_var, from_=0.05, to=0.95, resolution=0.05,
                 orient=tk.HORIZONTAL, length=120, bg="#16213e", fg="white",
                 highlightthickness=0, troughcolor="#0f3460",
                 command=lambda _: setattr(self, "conf_thresh", self.conf_var.get())
                 ).pack(side=tk.LEFT, padx=4)

        # class selector
        tk.Label(toolbar, text="Class:", bg="#16213e", fg="#aaa",
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 2))
        self.cls_var = tk.IntVar(value=0)
        self._cls_display = [f"{i}: {n}" for i, n in enumerate(self.class_names)]
        self.cls_menu = ttk.Combobox(toolbar, values=self._cls_display,
                                     width=14, state="readonly")
        self.cls_menu.current(0)
        self.cls_menu.bind("<<ComboboxSelected>>", self._on_cls_menu_select)
        self.cls_menu.pack(side=tk.LEFT)

        # status label (right side)
        self.status_var = tk.StringVar(value="Open a folder to begin")
        tk.Label(toolbar, textvariable=self.status_var, bg="#16213e", fg="#aaaaaa",
                 font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=12)

        # ── main area ──
        main = tk.Frame(self, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True)

        # left panel — image list
        left = tk.Frame(main, bg="#16213e", width=200)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="Images", bg="#16213e", fg="#00CFFF",
                 font=("Segoe UI", 10, "bold"), pady=6).pack()

        self.img_list = tk.Listbox(left, bg="#0d1117", fg="#e6edf3",
                                    selectbackground="#0f3460",
                                    font=("Consolas", 8), borderwidth=0,
                                    highlightthickness=0, activestyle="none")
        self.img_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.img_list.bind("<<ListboxSelect>>", self._on_list_select)

        scrollbar = ttk.Scrollbar(left, command=self.img_list.yview)
        self.img_list.configure(yscrollcommand=scrollbar.set)

        # ── canvas ──
        canvas_frame = tk.Frame(main, bg="#1a1a2e")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#0d1117",
                                 cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)
        self.canvas.bind("<Button-3>",        self._on_right_click)
        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        # zoom
        self.canvas.bind("<MouseWheel>",      self._on_mousewheel)        # Windows
        self.canvas.bind("<Button-4>",        self._on_mousewheel)        # Linux scroll up
        self.canvas.bind("<Button-5>",        self._on_mousewheel)        # Linux scroll down
        # pan (middle mouse)
        self.canvas.bind("<ButtonPress-2>",   self._on_pan_start)
        self.canvas.bind("<B2-Motion>",       self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # ── right panel — bbox list ──
        right = tk.Frame(main, bg="#16213e", width=220)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        tk.Label(right, text="Bounding Boxes", bg="#16213e", fg="#00CFFF",
                 font=("Segoe UI", 10, "bold"), pady=6).pack()
        self.stats_var = tk.StringVar(value="")
        tk.Label(right, textvariable=self.stats_var, bg="#16213e", fg="#888",
                 font=("Consolas", 8), justify=tk.LEFT, wraplength=200).pack(padx=4)

        self.bbox_list = tk.Listbox(right, bg="#0d1117", fg="#e6edf3",
                                     selectbackground="#0f3460",
                                     font=("Consolas", 8), borderwidth=0,
                                     highlightthickness=0, activestyle="none")
        self.bbox_list.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.bbox_list.bind("<<ListboxSelect>>", self._on_bbox_list_select)

        # bottom nav
        nav = tk.Frame(self, bg="#16213e", pady=4)
        nav.pack(side=tk.BOTTOM, fill=tk.X)

        nav_btn = {**btn_style, "font": ("Segoe UI", 10, "bold"), "padx": 20}
        tk.Button(nav, text="◀  Prev  (A)", command=self._prev_image, **nav_btn).pack(side=tk.LEFT, padx=8)
        self.nav_label = tk.Label(nav, text="— / —", bg="#16213e", fg="white",
                                   font=("Segoe UI", 10))
        self.nav_label.pack(side=tk.LEFT, expand=True)
        self.progress_var = tk.StringVar(value="")
        tk.Label(nav, textvariable=self.progress_var, bg="#16213e", fg="#00FF7F",
                 font=("Consolas", 9)).pack(side=tk.LEFT, expand=True)
        tk.Button(nav, text="Next  (D)  ▶", command=self._next_image, **nav_btn).pack(side=tk.RIGHT, padx=8)

    # ── key bindings ───────────────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind("<KeyPress-a>",      lambda _: self._prev_image())
        self.bind("<KeyPress-d>",      lambda _: self._next_image())
        self.bind("<KeyPress-s>",      lambda _: self._save_labels())
        self.bind("<KeyPress-n>",      lambda _: self._toggle_draw())
        self.bind("<KeyPress-r>",      lambda _: self._run_yolo())
        self.bind("<KeyPress-h>",      lambda _: self._toggle_conf_labels())
        self.bind("<KeyPress-f>",      lambda _: self._zoom_reset())
        self.bind("<Delete>",          lambda _: self._delete_selected())
        self.bind("<BackSpace>",       lambda _: self._delete_selected())
        self.bind("<Escape>",          lambda _: self._cancel())
        self.bind("<Control-z>",        lambda _: self._undo())
        self.bind("<Control-Z>",        lambda _: self._undo())
        for i in range(10):
            self.bind(str(i), lambda e, c=i: self._change_class(c))

    # ── model ──────────────────────────────────────────────────────────────────

    def _load_model(self, path):
        if not YOLO_AVAILABLE:
            messagebox.showerror("Error", "ultralytics not installed.\npip install ultralytics")
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self._set_status(f"Model: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Model load error", str(e))

    def _open_model(self):
        path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[("YOLO weights", "*.pt *.onnx"), ("All files", "*.*")]
        )
        if path:
            self._load_model(path)

    # ── folder / images ────────────────────────────────────────────────────────

    def _open_folder(self):
        d = filedialog.askdirectory(title="Select images folder")
        if d:
            self._load_folder(d)

    def _load_folder(self, d):
        self.images_dir = d
        self.image_paths = sorted([
            p for p in glob.glob(os.path.join(d, "*"))
            if Path(p).suffix.lower() in IMAGE_EXTS
        ])
        self.img_list.delete(0, tk.END)
        for p in self.image_paths:
            lp = self._label_path(p)
            suffix = " ✓" if os.path.exists(lp) else ""
            self.img_list.insert(tk.END, Path(p).name + suffix)
        self.current_idx = 0
        self._update_progress()
        if self.image_paths:
            self._load_image(0)

    def _load_image(self, idx):
        if not self.image_paths:
            return
        self.current_idx = idx
        path = self.image_paths[idx]

        # highlight list
        self.img_list.selection_clear(0, tk.END)
        self.img_list.selection_set(idx)
        self.img_list.see(idx)

        self.img_pil = Image.open(path)
        self.selected_id = None
        self.draw_mode = False
        self.drag_op = None
        # reset zoom for new image
        self.zoom_level = 1.0
        iw, ih = self.img_pil.size
        self.zoom_offset = [iw / 2, ih / 2]
        self.zoom_label_var.set("100%")
        self.is_dirty = False
        self.undo_stack = []

        # load existing labels or run YOLO
        label_path = self._label_path(path)
        has_labels = os.path.exists(label_path)
        if has_labels:
            self._load_labels(label_path)
        else:
            self.bboxes = []
            if self.model:
                self._run_yolo(silent=True)

        # sync checkmark in list
        self.img_list.see(idx)

        self._redraw()
        self._update_nav()
        self._update_bbox_list()

    def _label_path(self, img_path):
        img_path = Path(img_path)
        labels_dir = img_path.parent / "labels"
        labels_dir.mkdir(exist_ok=True)
        return str(labels_dir / img_path.with_suffix(".txt").name)

    def _load_labels(self, label_path):
        self.bboxes = []
        try:
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                        self.bboxes.append(BBox(cx, cy, w, h, cls, confirmed=True))
        except Exception as e:
            print(f"Error loading labels: {e}")

    def _save_labels(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        label_path = self._label_path(path)
        # confirm all boxes on save
        for b in self.bboxes:
            b.confirmed = True
        with open(label_path, "w") as f:
            for b in self.bboxes:
                f.write(b.to_yolo_line() + "\n")
        self.is_dirty = False
        self.selected_id = None
        self._set_status(f"Saved {len(self.bboxes)} boxes \u2192 {Path(label_path).name}")
        self._redraw()
        self._update_bbox_list()
        self._update_list_item()
        self._update_progress()

    # ── YOLO ───────────────────────────────────────────────────────────────────

    def _run_yolo(self, silent=False):
        if not self.model:
            if not silent:
                messagebox.showinfo("No model", "Load a YOLO model first (toolbar button).")
            return
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        try:
            results = self.model.predict(path, conf=self.conf_thresh, verbose=False)
            # remove previous unconfirmed predictions
            self.bboxes = [b for b in self.bboxes if b.confirmed]
            iw, ih = self.img_pil.size
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls = int(box.cls[0])
                    b = BBox(0, 0, 0, 0, cls, confirmed=False)
                    b.from_pixel(x1, y1, x2, y2, iw, ih)
                    b._conf = float(box.conf[0])
                    self.bboxes.append(b)
            self._redraw()
            self._update_bbox_list()
            if not silent:
                n = sum(1 for b in self.bboxes if not b.confirmed)
                self._set_status(f"YOLO: {n} predictions (conf ≥ {self.conf_thresh:.2f})")
        except Exception as e:
            if not silent:
                messagebox.showerror("YOLO error", str(e))

    # ── canvas drawing ─────────────────────────────────────────────────────────

    def _image_to_canvas(self, x, y):
        ox, oy = self.canvas_offset
        return x * self.img_scale + ox, y * self.img_scale + oy

    def _canvas_to_image(self, cx, cy):
        ox, oy = self.canvas_offset
        return (cx - ox) / self.img_scale, (cy - oy) / self.img_scale

    def _base_fit_scale(self):
        """Scale that fits the whole image in the canvas (no zoom)."""
        if not self.img_pil:
            return 1.0
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 1 or ch < 1:
            return 1.0
        iw, ih = self.img_pil.size
        return min(cw / iw, ch / ih)

    def _apply_zoom(self):
        """Recompute img_scale and canvas_offset from zoom_level and zoom_offset."""
        if not self.img_pil:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        iw, ih = self.img_pil.size

        base = self._base_fit_scale()
        self.img_scale = base * self.zoom_level

        # canvas_offset = screen position of image pixel (0,0)
        # zoom_offset holds the image-pixel that should be at canvas center
        cx_center = cw / 2
        cy_center = ch / 2
        ox = cx_center - self.zoom_offset[0] * self.img_scale
        oy = cy_center - self.zoom_offset[1] * self.img_scale
        self.canvas_offset = (ox, oy)

    def _zoom_reset(self):
        """Reset to fit view."""
        self.zoom_level = 1.0
        if self.img_pil:
            iw, ih = self.img_pil.size
            self.zoom_offset = [iw / 2, ih / 2]
        self._redraw()

    def _on_mousewheel(self, event):
        if not self.img_pil:
            return
        # determine scroll direction
        if event.num == 4:          # Linux up
            delta = 1
        elif event.num == 5:        # Linux down
            delta = -1
        else:                       # Windows: event.delta ±120
            delta = 1 if event.delta > 0 else -1

        # image coords under cursor before zoom
        ix, iy = self._canvas_to_image(event.x, event.y)
        iw, ih = self.img_pil.size
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))

        old_zoom = self.zoom_level
        if delta > 0:
            self.zoom_level = min(self.ZOOM_MAX, self.zoom_level * self.ZOOM_STEP)
        else:
            self.zoom_level = max(self.ZOOM_MIN, self.zoom_level / self.ZOOM_STEP)

        if self.zoom_level == old_zoom:
            return

        # keep the pixel under cursor stationary:
        # after zoom, canvas_offset will shift so that (ix,iy) stays at (event.x, event.y)
        base = self._base_fit_scale()
        new_scale = base * self.zoom_level
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # new offset such that ix*new_scale + ox = event.x
        ox = event.x - ix * new_scale
        oy = event.y - iy * new_scale
        self.canvas_offset = (ox, oy)
        self.img_scale = new_scale
        # sync zoom_offset (image pixel at canvas center)
        cx_center = cw / 2
        cy_center = ch / 2
        self.zoom_offset[0] = (cx_center - ox) / new_scale
        self.zoom_offset[1] = (cy_center - oy) / new_scale

        pct = int(self.zoom_level * self._base_fit_scale() /
                  self._base_fit_scale() * 100)
        self.zoom_label_var.set(f"{int(self.zoom_level*100)}%")
        self._redraw_no_recalc()

    def _on_pan_start(self, event):
        self._pan_last = (event.x, event.y)

    def _on_pan_drag(self, event):
        if not self._pan_last or not self.img_pil:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)
        # shift canvas offset
        ox, oy = self.canvas_offset
        ox += dx; oy += dy
        self.canvas_offset = (ox, oy)
        # sync zoom_offset
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.zoom_offset[0] = (cw / 2 - ox) / self.img_scale
        self.zoom_offset[1] = (ch / 2 - oy) / self.img_scale
        self._redraw_no_recalc()

    def _on_pan_end(self, event):
        self._pan_last = None

    def _redraw_no_recalc(self):
        """Redraw without recalculating zoom (offset already set)."""
        self.canvas.delete("all")
        if not self.img_pil:
            return
        iw, ih = self.img_pil.size
        nw = max(1, int(iw * self.img_scale))
        nh = max(1, int(ih * self.img_scale))
        resized = self.img_pil.resize((nw, nh), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized)
        ox, oy = self.canvas_offset
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.img_tk)
        self._draw_bboxes()

    def _redraw(self):
        self.canvas.delete("all")
        if not self.img_pil:
            return

        self._apply_zoom()
        iw, ih = self.img_pil.size
        nw = max(1, int(iw * self.img_scale))
        nh = max(1, int(ih * self.img_scale))

        resized = self.img_pil.resize((nw, nh), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized)
        ox, oy = self.canvas_offset
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.img_tk)
        self._draw_bboxes()

    def _draw_bboxes(self):
        if not self.img_pil:
            return
        iw, ih = self.img_pil.size

        for b in self.bboxes:
            x1, y1, x2, y2 = b.to_pixel(iw, ih)
            cx1, cy1 = self._image_to_canvas(x1, y1)
            cx2, cy2 = self._image_to_canvas(x2, y2)

            selected = (b.id == self.selected_id)

            if selected:
                color = COLORS["selected"]
                width = 2
                dash = ()
            elif b.confirmed:
                color = COLORS["confirmed"]
                width = 2
                dash = ()
            else:
                color = COLORS["predicted"]
                width = 1
                dash = (6, 3)

            self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                          outline=color, width=width,
                                          dash=dash, tags=f"bbox_{b.id}")

            # class + conf label
            conf_str = ""
            if hasattr(b, "_conf") and self.show_conf:
                conf_str = f" {b._conf:.2f}"
            cls_name = self.class_names[b.cls] if b.cls < len(self.class_names) else str(b.cls)
            label = f"{cls_name}{conf_str}"
            self.canvas.create_text(cx1 + 4, cy1 - 10,
                                     text=label, fill=color,
                                     font=("Consolas", 8, "bold"),
                                     anchor=tk.W, tags=f"label_{b.id}")

            # draw resize handles if selected
            if selected:
                for hx, hy in [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2),
                                ((cx1+cx2)/2, cy1), ((cx1+cx2)/2, cy2),
                                (cx1, (cy1+cy2)/2), (cx2, (cy1+cy2)/2)]:
                    s = HANDLE_SIZE // 2
                    self.canvas.create_rectangle(hx-s, hy-s, hx+s, hy+s,
                                                  fill=COLORS["handle"],
                                                  outline="black",
                                                  tags=f"handle_{b.id}")

    # ── mouse interaction ──────────────────────────────────────────────────────

    def _hit_test(self, cx, cy):
        """Return (bbox, region) where region is 'move', 'tl','tr','bl','br','t','b','l','r' or None"""
        if not self.img_pil:
            return None, None
        iw, ih = self.img_pil.size
        best = None
        best_area = float("inf")

        for b in reversed(self.bboxes):
            x1, y1, x2, y2 = b.to_pixel(iw, ih)
            px1, py1 = self._image_to_canvas(x1, y1)
            px2, py2 = self._image_to_canvas(x2, y2)

            # handle hit test (selected box only)
            if b.id == self.selected_id:
                handles = {
                    "tl": (px1, py1), "tr": (px2, py1),
                    "bl": (px1, py2), "br": (px2, py2),
                    "t":  ((px1+px2)/2, py1), "b": ((px1+px2)/2, py2),
                    "l":  (px1, (py1+py2)/2), "r": (px2, (py1+py2)/2),
                }
                for region, (hx, hy) in handles.items():
                    if abs(cx - hx) <= HANDLE_SIZE and abs(cy - hy) <= HANDLE_SIZE:
                        return b, region

            # inside bbox
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                area = (px2-px1) * (py2-py1)
                if area < best_area:
                    best_area = area
                    best = b

        if best:
            return best, "move"
        return None, None

    def _on_mouse_press(self, event):
        cx, cy = event.x, event.y
        self.focus_set()

        if self.draw_mode:
            self.draw_start = self._canvas_to_image(cx, cy)
            return

        b, region = self._hit_test(cx, cy)
        if b:
            self.selected_id = b.id
            ix, iy = self._canvas_to_image(cx, cy)
            iw, ih = self.img_pil.size
            x1, y1, x2, y2 = b.to_pixel(iw, ih)
            self.drag_op = {
                "type":   "move" if region == "move" else "resize",
                "region": region,
                "bbox":   b,
                "start_mouse": (ix, iy),
                "start_box":   (x1, y1, x2, y2),
            }
            self._redraw()
            self._update_bbox_list()
        else:
            # click on empty — start drawing
            self.selected_id = None
            self.draw_mode = True
            self.draw_start = self._canvas_to_image(cx, cy)
            self._redraw()

    def _on_mouse_drag(self, event):
        cx, cy = event.x, event.y
        if not self.img_pil:
            return
        iw, ih = self.img_pil.size
        ix, iy = self._canvas_to_image(cx, cy)
        # clamp to image
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))

        if self.draw_mode and self.draw_start:
            # live preview of new box
            self._redraw()
            sx, sy = self.draw_start
            px1, py1 = self._image_to_canvas(sx, sy)
            px2, py2 = self._image_to_canvas(ix, iy)
            self.canvas.create_rectangle(min(px1,px2), min(py1,py2),
                                          max(px1,px2), max(py1,py2),
                                          outline=COLORS["confirmed"], width=2,
                                          dash=(4,2))
            return

        if not self.drag_op:
            return

        b   = self.drag_op["bbox"]
        op  = self.drag_op["type"]
        reg = self.drag_op["region"]
        sx, sy       = self.drag_op["start_mouse"]
        bx1,by1,bx2,by2 = self.drag_op["start_box"]
        dx, dy = ix - sx, iy - sy

        if op == "move":
            nx1 = max(0, min(iw - (bx2-bx1), bx1 + dx))
            ny1 = max(0, min(ih - (by2-by1), by1 + dy))
            nx2 = nx1 + (bx2 - bx1)
            ny2 = ny1 + (by2 - by1)
            b.from_pixel(nx1, ny1, nx2, ny2, iw, ih)

        elif op == "resize":
            nx1, ny1, nx2, ny2 = bx1, by1, bx2, by2
            if "l" in reg: nx1 = min(bx2 - MIN_BOX, bx1 + dx)
            if "r" in reg: nx2 = max(bx1 + MIN_BOX, bx2 + dx)
            if "t" in reg: ny1 = min(by2 - MIN_BOX, by1 + dy)
            if "b" in reg: ny2 = max(by1 + MIN_BOX, by2 + dy)
            nx1 = max(0, nx1); ny1 = max(0, ny1)
            nx2 = min(iw, nx2); ny2 = min(ih, ny2)
            b.from_pixel(nx1, ny1, nx2, ny2, iw, ih)

        if not getattr(self, '_drag_undo_pushed', False):
            self._push_undo()
            self._drag_undo_pushed = True
        b.confirmed = True
        self._mark_dirty()
        self._redraw()

    def _on_mouse_release(self, event):
        cx, cy = event.x, event.y
        if not self.img_pil:
            return
        iw, ih = self.img_pil.size
        ix, iy = self._canvas_to_image(cx, cy)
        ix = max(0, min(iw, ix))
        iy = max(0, min(ih, iy))

        if self.draw_mode and self.draw_start:
            sx, sy = self.draw_start
            if abs(ix - sx) > MIN_BOX / iw and abs(iy - sy) > MIN_BOX / ih:
                b = BBox(0, 0, 0, 0, self.cls_var.get(), confirmed=True)
                b.from_pixel(min(sx,ix), min(sy,iy), max(sx,ix), max(sy,iy), iw, ih)
                self._push_undo()
                self.bboxes.append(b)
                self.selected_id = b.id
                self._mark_dirty()
            self.draw_mode = False
            self.draw_start = None
            self._redraw()
            self._update_bbox_list()
            return

        self.drag_op = None
        self._drag_undo_pushed = False
        self._update_bbox_list()

    def _on_right_click(self, event):
        b, _ = self._hit_test(event.x, event.y)
        if b:
            self._push_undo()
            self.bboxes = [x for x in self.bboxes if x.id != b.id]
            if self.selected_id == b.id:
                self.selected_id = None
            self._mark_dirty()
            self._redraw()
            self._update_bbox_list()

    def _on_canvas_resize(self, event):
        self._redraw()

    # ── actions ────────────────────────────────────────────────────────────────

    def _toggle_draw(self):
        self.draw_mode = not self.draw_mode
        self.selected_id = None
        cursor = "tcross" if self.draw_mode else "crosshair"
        self.canvas.configure(cursor=cursor)
        self._set_status("Draw mode ON — drag to create box" if self.draw_mode else "Draw mode OFF")
        self._redraw()

    def _delete_selected(self):
        if self.selected_id is None:
            return
        self._push_undo()
        self.bboxes = [b for b in self.bboxes if b.id != self.selected_id]
        self.selected_id = None
        self._mark_dirty()
        self._redraw()
        self._update_bbox_list()

    def _cancel(self):
        self.draw_mode = False
        self.draw_start = None
        self.selected_id = None
        self.canvas.configure(cursor="crosshair")
        self._redraw()

    def _change_class(self, cls):
        self.cls_var.set(cls)
        if self.selected_id is not None:
            for b in self.bboxes:
                if b.id == self.selected_id:
                    self._push_undo()
                    b.cls = cls
                    self._mark_dirty()
                    self._redraw()
                    self._update_bbox_list()
                    break

    def _toggle_conf_labels(self):
        self.show_conf = not self.show_conf
        self._redraw()

    def _prev_image(self):
        if self.current_idx > 0:
            self._load_image(self.current_idx - 1)

    def _next_image(self):
        if self.current_idx < len(self.image_paths) - 1:
            self._load_image(self.current_idx + 1)

    def _on_list_select(self, event):
        sel = self.img_list.curselection()
        if sel:
            self._load_image(sel[0])

    def _on_bbox_list_select(self, event):
        sel = self.bbox_list.curselection()
        if sel:
            idx = sel[0]
            if idx < len(self.bboxes):
                self.selected_id = self.bboxes[idx].id
                self._redraw()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _mark_dirty(self):
        if not self.is_dirty:
            self.is_dirty = True
            self._update_list_item()

    def _update_list_item(self):
        if not self.image_paths:
            return
        path = self.image_paths[self.current_idx]
        raw_name = path.split("/")[-1].split("\\")[-1]
        from pathlib import Path as _P
        raw_name = _P(path).name
        label_path = self._label_path(path)
        import os as _os
        has_labels = _os.path.exists(label_path)
        if self.is_dirty:
            display = raw_name + " ✎"
        elif has_labels:
            display = raw_name + " ✓"
        else:
            display = raw_name
        self.img_list.delete(self.current_idx)
        self.img_list.insert(self.current_idx, display)
        self.img_list.selection_set(self.current_idx)

    def _update_nav(self):
        n = len(self.image_paths)
        i = self.current_idx + 1 if n else 0
        self.nav_label.configure(text=f"{i} / {n}")
        self._update_progress()

    def _update_bbox_list(self):
        self.bbox_list.delete(0, tk.END)
        # stats per class
        from collections import Counter
        counts = Counter(b.cls for b in self.bboxes)
        if counts:
            parts = [f"{self.class_names[c] if c < len(self.class_names) else c}: {n}"
                     for c, n in sorted(counts.items())]
            self.stats_var.set("  ".join(parts))
        else:
            self.stats_var.set("no boxes")
        for b in self.bboxes:
            cls_name = self.class_names[b.cls] if b.cls < len(self.class_names) else str(b.cls)
            source = "YOLO" if not b.confirmed else "manual"
            conf = f" {b._conf:.2f}" if hasattr(b, "_conf") and not b.confirmed else ""
            entry = f"[{cls_name}]{conf}  {source}  ({b.w*100:.0f}%×{b.h*100:.0f}%)"
            self.bbox_list.insert(tk.END, entry)
            # color predicted differently
            color = COLORS["predicted"] if not b.confirmed else COLORS["confirmed"]
            if b.id == self.selected_id:
                color = COLORS["selected"]
            self.bbox_list.itemconfig(tk.END, fg=color)

    def _push_undo(self):
        """Snapshot current bboxes onto undo stack."""
        snapshot = [b.clone() for b in self.bboxes]
        # copy _conf if present
        for orig, snap in zip(self.bboxes, snapshot):
            if hasattr(orig, '_conf'):
                snap._conf = orig._conf
        self.undo_stack.append((snapshot, self.selected_id))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self):
        if not self.undo_stack:
            self._set_status("Nothing to undo")
            return
        snapshot, sel_id = self.undo_stack.pop()
        self.bboxes = snapshot
        self.selected_id = sel_id
        self._mark_dirty()
        self._redraw()
        self._update_bbox_list()
        self._set_status(f"Undo — {len(self.undo_stack)} steps left")

    def _update_progress(self):
        if not self.image_paths:
            self.progress_var.set("")
            return
        total = len(self.image_paths)
        done = sum(1 for p in self.image_paths if os.path.exists(self._label_path(p)))
        pct = int(done / total * 100) if total else 0
        self.progress_var.set(f"{done}/{total} saved  ({pct}%)")

    def _on_cls_menu_select(self, event):
        idx = self.cls_menu.current()
        self.cls_var.set(idx)
        self._change_class(idx)

    def _set_status(self, msg):
        self.status_var.set(msg)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Re-labeling Tool")
    parser.add_argument("--images",  "-i", default=None, help="Path to images folder")
    parser.add_argument("--model",   "-m", default=None, help="Path to YOLO .pt model")
    parser.add_argument("--classes", "-c", default=None, help="Comma-separated class names")
    args = parser.parse_args()

    class_names = args.classes.split(",") if args.classes else None

    app = App(
        images_dir=args.images,
        model_path=args.model,
        class_names=class_names,
    )
    app.mainloop()


if __name__ == "__main__":
    main()