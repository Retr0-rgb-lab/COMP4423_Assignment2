# -*- coding: utf-8 -*-
"""选一张场景照片，显示原图，并给出 inside / outside。

在仓库根目录启动::

    python -m src.task5.app

没有桌面时，同一条预处理和同一条 Pipeline 可以只打印结果::

    python -m src.task5.app --image path/to/photo.jpg
"""
import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

# 既支持 python -m src.task5.app，也支持 python src/task5/app.py。
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from PIL import Image, ImageTk

from src.task3.config import ROOT
from src.task3.preprocess import load_rgb
from src.task5.predict import load_pipeline, predict_rgb

BG = "#f4f1ea"
CARD = "#fffcf7"
INK = "#1c1b19"
MUTED = "#5f5a52"
LINE = "#e3ddd2"
INSIDE = "#0f6b4c"
OUTSIDE = "#1a4f8b"
TRACK = "#ece7dc"

PREVIEW_W = 460
PREVIEW_H = 340
CROP_SIDE = 112

_CLF_EN = {
    "C1": "Linear SVM",
    "C2": "Logistic Regression",
    "C3": "k-NN",
    "C4": "Random Forest",
    "C5": "Histogram Gradient Boosting",
    "C6": "Multinomial Naive Bayes",
    "C1-RBF": "RBF SVM",
}

def _image_filetypes():
    # Windows 的 Tk 用分号分隔扩展名；Linux / macOS 用空格。
    if sys.platform == "win32":
        patterns = "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp"
        jpeg = "*.jpg;*.jpeg"
    else:
        patterns = "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"
        jpeg = "*.jpg *.jpeg"
    return (
        ("Image files", patterns),
        ("JPEG", jpeg),
        ("PNG", "*.png"),
        ("All files", "*.*"),
    )


def _enable_windows_dpi():
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            return


def _font_family(root):
    import tkinter.font as tkfont
    return tkfont.nametofont("TkDefaultFont").cget("family")


def _initial_dir():
    pictures = ROOT / "data" / "pictures"
    if pictures.is_dir():
        return pictures
    return Path.home()


def model_summary(pipeline):
    feature = getattr(pipeline, "feature_name_", "features")
    clf_id = getattr(pipeline, "clf_id_", "")
    clf = _CLF_EN.get(clf_id) or getattr(pipeline, "clf_name_", "classifier")
    return f"{feature} + {clf}"


class _ProbRow:
    def __init__(self, parent, title, color, font, bold_font):
        self.font = font
        self.bold_font = bold_font
        self.color = color
        block = tk.Frame(parent, bg=CARD)
        block.pack(fill="x", pady=(0, 8))
        head = tk.Frame(block, bg=CARD)
        head.pack(fill="x")
        self.name = tk.Label(head, text=title, bg=CARD, fg=INK, anchor="w", font=font)
        self.name.pack(side="left")
        self.value = tk.StringVar(value="—")
        self.value_label = tk.Label(
            head, textvariable=self.value, bg=CARD, fg=MUTED, anchor="e", font=font,
        )
        self.value_label.pack(side="right")
        self.track = tk.Frame(block, bg=TRACK, height=8)
        self.track.pack(fill="x", pady=(4, 0))
        self.track.pack_propagate(False)
        self.bar = tk.Frame(self.track, bg=color, width=0, height=8)

    def set(self, probability, chosen):
        font = self.bold_font if chosen else self.font
        self.name.configure(font=font, fg=self.color if chosen else INK)
        self.value_label.configure(font=font, fg=self.color if chosen else MUTED)
        if probability is None:
            self.value.set("—")
            self.bar.place_forget()
            return
        self.value.set(f"{probability * 100:.1f}%")
        self.bar.place(x=0, y=0, relheight=1.0, relwidth=max(0.0, min(1.0, probability)))


class ClassifierApp:
    def __init__(self, root, pipeline):
        self.root = root
        self.pipeline = pipeline
        self._photo = None
        self._crop_photo = None
        self._last_dir = str(_initial_dir())
        family = _font_family(root)
        self.font_title = (family, 18, "bold")
        self.font_body = (family, 11)
        self.font_small = (family, 10)
        self.font_result = (family, 20, "bold")
        self.font_bold = (family, 11, "bold")
        self._build()

    def _build(self):
        root = self.root
        root.title("PolyU Main Campus Classifier")
        root.configure(bg=BG)
        root.minsize(880, 520)
        root.bind("<Control-o>", lambda _event: self.choose_image())

        self.accent = tk.Frame(root, bg=INK, height=4)
        self.accent.pack(fill="x")

        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=20, pady=14)

        tk.Label(
            outer, text="PolyU Main Campus Classifier",
            bg=BG, fg=INK, font=self.font_title, anchor="w",
        ).pack(fill="x")
        tk.Label(
            outer,
            text="Select a scene photograph. The result is inside or outside the main campus.",
            bg=BG, fg=MUTED, font=self.font_body, anchor="w",
        ).pack(fill="x", pady=(2, 10))

        button_row = tk.Frame(outer, bg=BG)
        button_row.pack(fill="x", pady=(0, 10))
        self.select_button = tk.Button(
            button_row,
            text="Select image…",
            command=self.choose_image,
            bg=INK, fg="#ffffff",
            activebackground="#3a3834", activeforeground="#ffffff",
            relief="flat", bd=0, padx=16, pady=6,
            font=self.font_bold, cursor="hand2",
        )
        self.select_button.pack(side="left")
        self.file_var = tk.StringVar(value="No image selected")
        tk.Label(
            button_row, textvariable=self.file_var,
            bg=BG, fg=MUTED, font=self.font_small, anchor="w",
        ).pack(side="left", padx=(12, 0))

        columns = tk.Frame(outer, bg=BG)
        columns.pack(fill="both", expand=True)

        preview = tk.Frame(columns, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        preview.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            preview, width=PREVIEW_W, height=PREVIEW_H,
            bg=CARD, highlightthickness=0,
        )
        self.canvas.pack()
        self._draw_placeholder()
        self.size_var = tk.StringVar(value=" ")
        tk.Label(
            preview, textvariable=self.size_var,
            bg=CARD, fg=MUTED, font=self.font_small, anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        result = tk.Frame(columns, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        result.pack(side="left", fill="y", padx=(12, 0))
        body = tk.Frame(result, bg=CARD)
        body.pack(fill="both", expand=True, padx=14, pady=12)

        tk.Label(
            body, text="Prediction", bg=CARD, fg=MUTED,
            font=self.font_small, anchor="w",
        ).pack(fill="x")
        self.result_var = tk.StringVar(value="—")
        self.result_label = tk.Label(
            body, textvariable=self.result_var,
            bg=CARD, fg=INK, font=self.font_result, anchor="w", justify="left",
            wraplength=310,
        )
        self.result_label.pack(fill="x", pady=(2, 10))

        self.prob_note = tk.StringVar(value="")
        self.prob_rows = {
            1: _ProbRow(body, "Inside PolyU main campus", INSIDE, self.font_body, self.font_bold),
            0: _ProbRow(body, "Outside PolyU main campus", OUTSIDE, self.font_body, self.font_bold),
        }
        self.prob_note_label = tk.Label(
            body, textvariable=self.prob_note,
            bg=CARD, fg=MUTED, font=self.font_small, anchor="w", justify="left",
            wraplength=310,
        )

        crop_row = tk.Frame(body, bg=CARD)
        crop_row.pack(fill="x", pady=(8, 0))
        self.crop_canvas = tk.Canvas(
            crop_row, width=CROP_SIDE, height=CROP_SIDE,
            bg=TRACK, highlightthickness=0,
        )
        self.crop_canvas.pack(side="left")
        note = tk.Frame(crop_row, bg=CARD)
        note.pack(side="left", fill="both", expand=True, padx=(10, 0))
        tk.Label(
            note, text="Model input", bg=CARD, fg=INK, font=self.font_bold, anchor="w",
        ).pack(fill="x")
        tk.Label(
            note,
            text="256×256 center crop. Same resize_and_crop as evaluation.",
            bg=CARD, fg=MUTED, font=self.font_small, anchor="w", justify="left", wraplength=190,
        ).pack(fill="x", pady=(4, 0))

        summary = model_summary(self.pipeline)
        tk.Label(
            outer,
            text=f"models/task3_pipeline.joblib · {summary}",
            bg=BG, fg=MUTED, font=self.font_small, anchor="w",
        ).pack(fill="x", pady=(10, 0))
        tk.Label(
            outer,
            text="Pixels only. The file name, folder, EXIF, and GPS are not used.",
            bg=BG, fg=MUTED, font=self.font_small, anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            PREVIEW_W // 2, PREVIEW_H // 2,
            text="Selected photo appears here",
            fill=MUTED, font=self.font_body,
        )

    def choose_image(self):
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select a scene photograph",
            initialdir=self._last_dir,
            filetypes=_image_filetypes(),
        )
        if not path:
            return
        self._last_dir = str(Path(path).parent)
        self.load_and_predict(path)

    def load_and_predict(self, path):
        """``path`` 只用于读像素和显示文件名，不参与判别。"""
        try:
            rgb = load_rgb(path)
        except Exception as exc:
            messagebox.showerror(
                "Cannot read image",
                f"The file could not be read as an image.\n\n{exc}",
                parent=self.root,
            )
            return
        self.file_var.set(Path(path).name)
        self._show_rgb(rgb)
        self.result_var.set("Predicting…")
        self.result_label.configure(fg=INK)
        self.root.update_idletasks()
        try:
            prediction = predict_rgb(self.pipeline, rgb)
        except Exception as exc:
            self.result_var.set("Prediction failed")
            messagebox.showerror("Prediction failed", str(exc), parent=self.root)
            return
        self._show_prediction(prediction)

    def _show_rgb(self, rgb):
        height, width = rgb.shape[:2]
        self.size_var.set(f"Original {width}×{height} px")
        image = Image.fromarray(rgb)
        image.thumbnail((PREVIEW_W - 24, PREVIEW_H - 24), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(PREVIEW_W // 2, PREVIEW_H // 2, image=self._photo)

    def _show_prediction(self, prediction):
        self.result_var.set(prediction.text)
        color = INSIDE if prediction.label_id == 1 else OUTSIDE if prediction.label_id == 0 else INK
        self.result_label.configure(fg=color)
        self.accent.configure(bg=color)
        if prediction.probabilities:
            self.prob_note_label.pack_forget()
            for label_id, row in self.prob_rows.items():
                row.set(
                    prediction.probabilities.get(label_id),
                    chosen=label_id == prediction.label_id,
                )
        else:
            self.prob_note.set("This classifier does not provide class probabilities.")
            self.prob_note_label.pack(fill="x", pady=(0, 8))
            for row in self.prob_rows.values():
                row.set(None, chosen=False)
        crop = Image.fromarray(prediction.cropped)
        crop.thumbnail((CROP_SIDE, CROP_SIDE), Image.Resampling.LANCZOS)
        self._crop_photo = ImageTk.PhotoImage(crop)
        self.crop_canvas.delete("all")
        self.crop_canvas.create_image(CROP_SIDE // 2, CROP_SIDE // 2, image=self._crop_photo)


def _cli(path):
    try:
        pipeline = load_pipeline()
        rgb = load_rgb(path)
        prediction = predict_rgb(pipeline, rgb)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(prediction.text)
    names = {0: "outside", 1: "inside"}
    for label_id in sorted(prediction.probabilities):
        name = names.get(label_id, str(label_id))
        print(f"{name}: {prediction.probabilities[label_id]:.3f}")
    return 0


def _gui():
    _enable_windows_dpi()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print("Could not open a window.", file=sys.stderr)
        print(exc, file=sys.stderr)
        print("Run with --image FILE to predict one photo without a GUI.", file=sys.stderr)
        return 1
    try:
        pipeline = load_pipeline()
    except Exception as exc:
        messagebox.showerror("Model not available", str(exc), parent=root)
        root.destroy()
        return 1
    ClassifierApp(root, pipeline)
    root.mainloop()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Local inside/outside classifier for PolyU main campus photos.",
    )
    parser.add_argument(
        "--image",
        help="Predict one image and print the label. Does not open a window.",
    )
    args = parser.parse_args(argv)
    if args.image:
        return _cli(args.image)
    return _gui()


if __name__ == "__main__":
    sys.exit(main())
