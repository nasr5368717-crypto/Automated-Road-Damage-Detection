import datetime
import os
import sys
import threading
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score, confusion_matrix
from tkinter import END, BOTH, X, FLAT, LEFT, RIGHT, NORMAL, DISABLED, Button, Frame, Label, Canvas, Text, Scrollbar
from tkinter import filedialog

import road_damage_core as rd

GRADIENT_BANDS = rd.GRADIENT_BANDS
GRADIENT_DEBOUNCE_MS = rd.GRADIENT_DEBOUNCE_MS

BG = rd.BG
CARD = rd.CARD
ACCENT = rd.ACCENT
TEXT = rd.TEXT

main = tkinter.Tk()
main.title("Automated Road Damage detection using UAV images and DEEP Learning Techniques")
main.geometry("1440x900")
main.minsize(1280, 820)

bg_start, bg_end = rd.bg_start, rd.bg_end
card_bg = rd.card_bg
card_inner = rd.card_inner
text_dark = rd.text_dark
border_color = rd.border_color
secondary = rd.secondary
primary = accent_cyan = rd.ACCENT
muted_text = rd.muted_text
neon_green = rd.neon_green
PLOT_BG = rd.PLOT_BG

# --- Dashboard UI helpers (display only) ---
_metric_value_labels = {}
_model_in_use_label = None
_workflow_cells = []
_sidebar_buttons = []
_sidebar_active_btn = None
_time_header_label = None
_date_header_label = None
_gradient_redraw = None
_hdr_ready = None
_hdr_progress = None
_worker_busy = False
_pending_metric_sparks = []
_metrics_sparks_packed = False
_preview_photo_refs = [None, None]
_plot_canvas_refs = {}


def embed_matplotlib_figure(parent, fig):
    for w in parent.winfo_children():
        w.destroy()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=BOTH, expand=True, padx=6, pady=6)
    canvas.draw_idle()
    # Keep a strong reference so Tk canvas is not garbage-collected.
    _plot_canvas_refs[str(parent)] = canvas
    parent.update_idletasks()


def set_sidebar_busy(busy: bool):
    st = DISABLED if busy else NORMAL
    for btn in _sidebar_buttons:
        try:
            btn.config(state=st)
        except tkinter.TclError:
            pass


def create_card(parent, bg=None, glow=border_color, bd=1, **kw):
    bg = bg or card_bg
    return Frame(
        parent,
        bg=bg,
        highlightthickness=bd,
        highlightbackground=glow,
        highlightcolor=glow,
        **kw,
    )


def create_metric_card(parent, title, initial_value="—"):
    card = create_card(parent, glow=primary)
    inner = Frame(card, bg=card_inner)
    inner.pack(fill=BOTH, expand=True, padx=10, pady=10)
    title_lbl = Label(
        inner,
        text=title,
        bg=card_inner,
        fg=muted_text,
        font=("Segoe UI", 9),
    )
    title_lbl.pack(anchor="w")
    spark_row = Frame(inner, bg=card_inner)
    spark = Canvas(spark_row, height=26, width=168, bg=card_inner, highlightthickness=0, bd=0)
    spark.pack(anchor="w")
    spark.create_line(0, 18, 32, 10, 64, 16, 96, 6, 128, 14, 168, 8, fill=primary, width=2)
    val_lbl = Label(
        inner,
        text=initial_value,
        bg=card_inner,
        fg=muted_text if initial_value == "No Data" else text_dark,
        font=("Segoe UI", 22, "bold"),
    )
    if initial_value == "No Data":
        _pending_metric_sparks.append((spark_row, val_lbl))
        val_lbl.pack(anchor="w", pady=(6, 0))
    else:
        spark_row.pack(anchor="w", fill=X, pady=(4, 6))
        val_lbl.pack(anchor="w", pady=(6, 0))
    return card, val_lbl


def _sidebar_idle_style(btn):
    btn.config(
        bg=CARD,
        fg=text_dark,
        activebackground=primary,
        activeforeground=text_dark,
        bd=0,
        relief=FLAT,
        highlightthickness=2,
        highlightbackground=border_color,
        font=("Segoe UI", 10, "bold"),
        padx=14,
        pady=10,
        cursor="hand2",
        anchor="w",
    )


def _sidebar_hover(btn, entering):
    if _sidebar_active_btn is btn:
        return
    btn.config(
        bg="#2a3f66" if entering else CARD,
        highlightbackground=accent_cyan if entering else border_color,
    )


def set_sidebar_active(btn):
    global _sidebar_active_btn
    for b in _sidebar_buttons:
        _sidebar_idle_style(b)
        b.config(highlightbackground=border_color)
    _sidebar_active_btn = btn
    if btn is not None:
        btn.config(bg="#312E81", highlightbackground=primary)


def create_sidebar_button(parent, label, command):
    btn = Button(parent, text=label, command=lambda: None)

    def run():
        set_sidebar_active(btn)
        command()

    btn.config(command=run)
    _sidebar_idle_style(btn)
    btn.bind("<Enter>", lambda e: _sidebar_hover(btn, True))
    btn.bind("<Leave>", lambda e: _sidebar_hover(btn, False))
    _sidebar_buttons.append(btn)
    return btn


def update_metric_display(key, value_str):
    lbl = _metric_value_labels.get(key)
    if lbl is not None:
        lbl.config(text=value_str)


def set_model_in_use(name):
    global _model_in_use_label
    if _model_in_use_label is not None:
        _model_in_use_label.config(text="Model in Use: " + name)


def _workflow_paint():
    colors = {
        "done": neon_green,
        "active": primary,
        "pending": secondary,
    }
    name_fg = {
        "done": text_dark,
        "active": accent_cyan,
        "pending": "#64748B",
    }
    for cell in _workflow_cells:
        st = cell["state"]
        cell["dot"].config(fg=colors.get(st, secondary))
        cell["name"].config(fg=name_fg.get(st, muted_text))


def set_workflow_state(state_map):
    """state_map: step int 1..5 -> 'done' | 'active' | 'pending'."""
    for cell in _workflow_cells:
        sn = cell["step"]
        cell["state"] = state_map.get(sn, "pending")
    _workflow_paint()


def dashboard_on_upload():
    set_workflow_state(
        {1: "done", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},
    )


def dashboard_on_preprocess():
    set_workflow_state(
        {1: "done", 2: "done", 3: "pending", 4: "pending", 5: "pending"},
    )


def dashboard_on_split():
    set_workflow_state(
        {1: "done", 2: "done", 3: "done", 4: "pending", 5: "pending"},
    )


def dashboard_on_train():
    set_workflow_state(
        {1: "done", 2: "done", 3: "done", 4: "done", 5: "pending"},
    )


def dashboard_on_detect_start():
    set_workflow_state(
        {1: "done", 2: "done", 3: "done", 4: "done", 5: "active"},
    )


def dashboard_on_detect_end():
    set_workflow_state(
        {1: "done", 2: "done", 3: "done", 4: "done", 5: "done"},
    )


def refresh_metrics_from_scores(p_pct, r_pct, f_pct, a_pct):
    """Sync right-rail metrics from calculateMetrics percentages."""
    global _metrics_sparks_packed
    update_metric_display("map", f"{a_pct / 100.0:.2f}")
    update_metric_display("precision", f"{p_pct / 100.0:.2f}")
    update_metric_display("recall", f"{r_pct / 100.0:.2f}")
    update_metric_display("f1", f"{f_pct / 100.0:.2f}")
    for key in ("map", "precision", "recall", "f1"):
        lbl = _metric_value_labels.get(key)
        if lbl is not None:
            lbl.config(fg=text_dark)
    if not _metrics_sparks_packed and _pending_metric_sparks:
        for spark_row, val_lbl in _pending_metric_sparks:
            spark_row.pack(anchor="w", fill=X, pady=(4, 6), before=val_lbl)
        _pending_metric_sparks.clear()
        _metrics_sparks_packed = True


def draw_detection_label(frame_bgr, xmin, ymin, xmax, ymax, label_text):
    """White on red, filled label bg; scale ~0.7 with resolution; thickness 2–3; LINE_AA."""
    img_h, img_w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_short = float(min(img_h, img_w))
    font_scale = 0.7 * max(0.35, min(1.4, img_short / 720.0))
    thickness = 3 if img_short > 960 else 2
    pad_x = max(6, int(9 * font_scale))
    pad_y = max(5, int(7 * font_scale))
    display = label_text
    while font_scale >= 0.3:
        (tw, th), baseline = cv2.getTextSize(display, font, font_scale, thickness)
        box_w = tw + 2 * pad_x
        box_h = th + baseline + 2 * pad_y
        if box_w <= img_w - 6 and box_h <= img_h - 6:
            break
        font_scale = round(font_scale - 0.05, 2)
    (tw, th), baseline = cv2.getTextSize(display, font, font_scale, thickness)
    box_w, box_h = tw + 2 * pad_x, th + baseline + 2 * pad_y
    cv2.rectangle(
        frame_bgr,
        (int(xmin), int(ymin)),
        (int(xmax), int(ymax)),
        (0, 0, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    label_y_top = int(ymin) - box_h - 2
    if label_y_top < 0:
        label_y_top = int(ymax) + 2
    label_y_top = max(0, min(label_y_top, img_h - box_h))
    box_xmin = max(0, min(int(xmin), img_w - box_w))
    x_text = box_xmin + pad_x + max(0, (box_w - 2 * pad_x - tw) // 2)
    y_base = label_y_top + pad_y + th
    cv2.rectangle(
        frame_bgr,
        (box_xmin, label_y_top),
        (box_xmin + box_w, label_y_top + box_h),
        (0, 0, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        display,
        (x_text, y_base),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_damage_label_on_frame(frame_bgr, xmin, ymin, xmax, ymax, label_text):
    draw_detection_label(frame_bgr, xmin, ymin, xmax, ymax, label_text)


def draw_no_damage_badge(frame_bgr, label_text="No Damage"):
    img_h, img_w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_short = float(min(img_h, img_w))
    font_scale = 0.7 * max(0.35, min(1.2, img_short / 720.0))
    thickness = 3 if img_short > 960 else 2
    pad_x = max(8, int(10 * font_scale))
    pad_y = max(6, int(8 * font_scale))
    display = label_text
    while font_scale >= 0.3:
        (tw, th), baseline = cv2.getTextSize(display, font, font_scale, thickness)
        box_w, box_h = tw + 2 * pad_x, th + baseline + 2 * pad_y
        if box_w <= img_w - 6 and box_h <= img_h - 6:
            break
        font_scale = round(font_scale - 0.05, 2)
    (tw, th), baseline = cv2.getTextSize(display, font, font_scale, thickness)
    box_w, box_h = tw + 2 * pad_x, th + baseline + 2 * pad_y
    x, label_y_top = max(6, int(img_w * 0.02)), max(6, int(img_h * 0.04))
    cv2.rectangle(
        frame_bgr,
        (x, label_y_top),
        (x + box_w, label_y_top + box_h),
        (0, 0, 255),
        -1,
        lineType=cv2.LINE_AA,
    )
    yb = label_y_top + pad_y + th
    cv2.putText(
        frame_bgr,
        display,
        (x + pad_x, yb),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def _to_tk_image(bgr_img, max_size):
    """BGR (OpenCV) → RGB PIL → ImageTk.PhotoImage; keeps aspect ratio within max_size."""
    arr = np.ascontiguousarray(bgr_img)
    if arr.ndim == 2:
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 4:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if rgb.dtype != np.uint8:
        mx = float(np.nanmax(rgb)) if rgb.size else 0.0
        if rgb.dtype.kind == "f" and mx <= 1.0:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb)
    pil_img = Image.fromarray(rgb, mode="RGB")
    mw, mh = max_size
    mw = max(int(mw), 48)
    mh = max(int(mh), 48)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    pil_img.thumbnail((mw, mh), resample)
    return ImageTk.PhotoImage(pil_img)


PREVIEW_THUMB = (500, 400)


def _current_preview_thumb():
    """Use live preview panel size so images stay visible and proportional."""
    try:
        w = max(preview_top.winfo_width(), 1)
        h = max(preview_top.winfo_height(), 1)
        if w > 1 and h > 1:
            # Two cards side-by-side with small gutters.
            return max(220, (w // 2) - 28), max(180, h - 44)
    except Exception:
        pass
    return PREVIEW_THUMB


def show_images(original, detected):
    """Show original vs detected (run from main thread, e.g. inside main.after callback)."""

    def _sh(a):
        return a.shape if isinstance(a, np.ndarray) and a is not None and getattr(a, "size", 0) else None

    print(type(original), _sh(original))
    print(type(detected), _sh(detected))

    o = np.ascontiguousarray(original) if isinstance(original, np.ndarray) and original.size else None
    d = np.ascontiguousarray(detected) if isinstance(detected, np.ndarray) and detected.size else None

    image_placeholder.grid_remove()
    original_card.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(0, 6))
    detected_card.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=(0, 6))

    def _photo(arr):
        if arr is None or arr.size == 0 or arr.ndim < 2:
            return None
        try:
            return _to_tk_image(arr, _current_preview_thumb())
        except Exception as ex:
            print("Image conversion error:", ex)
            return None

    img1 = _photo(o)
    img2 = _photo(d)

    if img1 is None:
        original_label.config(image="", text="Image not available")
        original_label.image = None
    else:
        original_label.config(image=img1, text="")
        original_label.image = img1

    if img2 is None:
        detected_label.config(image="", text="Image not available")
        detected_label.image = None
    else:
        detected_label.config(image=img2, text="")
        detected_label.image = img2

    _preview_photo_refs[0] = img1
    _preview_photo_refs[1] = img2
    main._img_refs = (img1, img2)

def create_gradient_background(root):
    canvas = Canvas(root, highlightthickness=0, bd=0)
    debounce_id = [None]
    last_wh = [0, 0]

    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb_color):
        return "#{:02x}{:02x}{:02x}".format(*rgb_color)

    def draw_gradient():
        c1 = hex_to_rgb(bg_start)
        c2 = hex_to_rgb(bg_end)
        canvas.delete("grad")
        w = max(root.winfo_width(), 1)
        h = max(root.winfo_height(), 1)
        bands = min(GRADIENT_BANDS, max(24, h // 6))
        step = max(1, (h + bands - 1) // bands)
        for y0 in range(0, h, step):
            y1 = min(y0 + step, h)
            t = ((y0 + y1) * 0.5) / max(h - 1, 1)
            r = int(c1[0] * (1 - t) + c2[0] * t)
            g = int(c1[1] * (1 - t) + c2[1] * t)
            b = int(c1[2] * (1 - t) + c2[2] * t)
            canvas.create_rectangle(
                0, y0, w, y1,
                fill=rgb_to_hex((r, g, b)),
                outline="",
                tags=("grad",),
            )

    def on_configure(_event=None):
        if debounce_id[0] is not None:
            root.after_cancel(debounce_id[0])
        debounce_id[0] = root.after(GRADIENT_DEBOUNCE_MS, _apply_resize)

    def _apply_resize():
        debounce_id[0] = None
        w = max(root.winfo_width(), 1)
        h = max(root.winfo_height(), 1)
        if w == last_wh[0] and h == last_wh[1]:
            return
        last_wh[0], last_wh[1] = w, h
        draw_gradient()

    def force_redraw():
        last_wh[0] = last_wh[1] = 0
        draw_gradient()
        last_wh[0] = max(root.winfo_width(), 1)
        last_wh[1] = max(root.winfo_height(), 1)

    root.bind("<Configure>", on_configure)
    force_redraw()
    global _gradient_redraw
    _gradient_redraw = force_redraw
    return canvas

filename = None


def _set_processing_header(busy: bool):
    global _hdr_ready, _hdr_progress
    if _hdr_ready is not None:
        _hdr_ready.config(text="Processing... Please wait" if busy else "System Ready")
    if _hdr_progress is not None:
        _hdr_progress.config(text="Working..." if busy else "")


def run_in_background(op_name, work_fn, ui_fn=None):
    global _worker_busy
    if _worker_busy:
        text.insert(
            END,
            f"[{op_name}] Skipped — another operation is running. Please wait.\n",
        )
        return
    _worker_busy = True
    _set_processing_header(True)
    set_sidebar_busy(True)

    def worker():
        err = None
        result = None
        try:
            result = work_fn()
        except Exception as ex:
            err = ex

        def finish():
            global _worker_busy
            _worker_busy = False
            _set_processing_header(False)
            set_sidebar_busy(False)
            if err is not None:
                text.insert(END, f"[{op_name}] Error: {err}\n")
                if op_name == "Damage detection":
                    dashboard_on_detect_end()
            elif ui_fn is not None:

                def _run_ui():
                    ui_fn(result)

                main.after(0, _run_ui)

        main.after(0, finish)

    threading.Thread(target=worker, daemon=True).start()


run_in_thread = run_in_background



def calculateMetrics(algorithm, predict, testY):
    text.delete('1.0', END)
    _select_tab('log')
    try:
        p, r, f = rd.macro_scores(testY, predict)
        a = accuracy_score(testY, predict) * 100
    except ValueError as e:
        text.insert(END, 'Metrics error: ' + str(e) + '\n')
        return
    text.insert(END, algorithm + ' Accuracy  : ' + str(a) + '\n')
    text.insert(END, algorithm + ' Precision   : ' + str(p) + '\n')
    text.insert(END, algorithm + ' Recall      : ' + str(r) + '\n')
    text.insert(END, algorithm + ' FMeasure    : ' + str(f) + '\n')
    rd.METRICS_BY_MODEL[algorithm] = (p, r, f, a)
    conf_matrix = confusion_matrix(testY, predict)
    classes = rd.confusion_class_labels(conf_matrix.shape[0])
    refresh_metrics_from_scores(p, r, f, a)
    text.insert(END, 'Confusion matrix updated.\n')
    fig_cm = Figure(figsize=(10, 7), dpi=100, facecolor=PLOT_BG)
    ax_cm = fig_cm.add_subplot(111)
    rd.plot_confusion_matrix(fig_cm, ax_cm, algorithm, conf_matrix, classes)
    embed_matplotlib_figure(f_conf, fig_cm)
    _select_tab('conf')


def UploadDataset():
    global filename
    picked = filedialog.askdirectory(initialdir='.') or ''
    text.delete('1.0', END)
    if not picked:
        text.insert(END, 'Upload cancelled.\n')
        return
    filename = picked

    def ui(res):
        text.delete('1.0', END)
        if not res.get('ok'):
            text.insert(END, res.get('msg', 'Upload failed.\n'))
            return
        rd.data = res['data']
        rd.labels = res['labels']
        rd.bboxes = res['bboxes']
        text.insert(END, 'Dataset images loaded\n')
        text.insert(END, 'Total images found in dataset : ' + str(rd.data.shape[0]) + '\n')
        dashboard_on_upload()

    run_in_background('Upload dataset', rd._upload_dataset_work, ui)


def PreprocessDataset():
    def ui(res):
        text.delete('1.0', END)
        if not res.get('ok'):
            text.insert(END, res.get('msg', ''))
            return
        text.insert(END, 'Dataset images are preprocessed.\n')
        dashboard_on_preprocess()
        fig_pp = Figure(figsize=(5, 5), dpi=100, facecolor=PLOT_BG)
        ax_pp = fig_pp.add_subplot(111)
        ax_pp.imshow(cv2.cvtColor(res['sample'], cv2.COLOR_BGR2RGB))
        ax_pp.axis('off')
        ax_pp.set_title('Preprocessed sample', color=TEXT, fontsize=11)
        fig_pp.tight_layout(pad=2)
        embed_matplotlib_figure(f_graphs, fig_pp)
        _select_tab('graphs')

    run_in_background('Preprocess', rd._preprocess_work, ui)


def split():
    text.delete('1.0', END)

    def ui(res):
        if not res.get('ok'):
            text.insert(END, res.get('msg', ''))
            return
        rd.data = res['data']
        rd.labels = res['labels']
        rd.bboxes = res['bboxes']
        rd.trainImages = res['trainImages']
        rd.testImages = res['testImages']
        rd.trainLabels = res['trainLabels']
        rd.testLabels = res['testLabels']
        rd.trainBBoxes = res['trainBBoxes']
        rd.testBBoxes = res['testBBoxes']
        rd.METRICS_BY_MODEL.clear()
        rd.models['yolov5'] = rd.models['yolov7'] = None
        rd.models['yolov8'] = None
        rd.yolov8_model = None
        text.insert(END, 'Dataset split complete.\n')
        text.insert(END, 'Training samples (80%): ' + str(rd.trainImages.shape[0]) + '\n')
        text.insert(END, 'Test samples (20%): ' + str(rd.testImages.shape[0]))
        dashboard_on_split()

    run_in_background('Split dataset', rd._split_work, ui)


def RunYOLOV5():
    def ui(res):
        predict, test = res
        calculateMetrics('YoloV5', predict, test)
        dashboard_on_train()
        set_model_in_use('YOLOv5')

    run_in_background('YOLOv5', rd._yolov5_worker, ui)


def RunYOLOV7():
    def ui(res):
        predict, test = res
        calculateMetrics('YoloV7', predict, test)
        dashboard_on_train()
        set_model_in_use('YOLOv7')

    run_in_background('YOLOv7', rd._yolov7_worker, ui)


def RunYOLOV8():
    def ui(res):
        predict, test = res
        calculateMetrics('YoloV8', predict, test)
        dashboard_on_train()
        set_model_in_use('YOLOv8')

    run_in_background('YOLOv8', rd._yolov8_worker, ui)


def comparisonGraph():
    need = ('YoloV5', 'YoloV7', 'YoloV8')
    missing = [k for k in need if k not in rd.METRICS_BY_MODEL]
    if len(missing) == len(need):
        text.delete('1.0', END)
        text.insert(END, 'No model metrics available yet. Run at least one YOLO model first.\n')
        return
    _select_tab('graphs')
    main.update_idletasks()
    text.delete('1.0', END)
    if missing:
        text.insert(END, 'Missing models for full comparison: ' + ', '.join(missing) + '\n')
        text.insert(END, 'Showing available model(s) in the Graphs tab.\n')
    else:
        text.insert(END, 'Comparison chart updated in the Graphs tab.\n')
    try:
        fig_cmp = rd.build_comparison_figure()
        embed_matplotlib_figure(f_graphs, fig_cmp)
    except Exception as ex:
        text.insert(END, f'Unable to build comparison graph: {ex}\n')
        return
    _select_tab('graphs')


def damageDetection():
    path = filedialog.askopenfilename(initialdir='testImages')
    if not path:
        text.delete('1.0', END)
        text.insert(END, 'No image selected\n')
        return
    frame = cv2.imread(path)
    if frame is None:
        text.delete('1.0', END)
        text.insert(END, 'Invalid Image File\n')
        return
    original_frame = frame.copy()
    text.delete('1.0', END)
    dashboard_on_detect_start()

    def work():
        det = rd._ensure_yolov8_detector()
        proc = frame.copy()
        results = det(proc)[0]
        flag = False
        boxes = results.boxes
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                if confidence >= 0.3:
                    xmin, ymin, xmax, ymax = map(int, boxes.xyxy[i])
                    pct = float(confidence) * 100.0
                    draw_detection_label(
                        proc, xmin, ymin, xmax, ymax, f'Damage {int(round(pct))}%'
                    )
                    flag = True
        if not flag:
            draw_no_damage_badge(proc, 'No Damage')
        return original_frame, proc, flag

    def ui(res):
        def update_ui():
            of, proc, flag = res
            if not flag:
                text.insert(END, 'Road Detected but No Damage\n')
            else:
                text.insert(END, 'Road Damage Detected\n')
            dashboard_on_detect_end()
            show_images(of, proc)

        main.after(0, update_ui)

    run_in_background('Damage detection', work, ui)


# --- Dashboard layout: strict root grid 20% / 55% / 25% + full-width header row ---
HEADER_ROW_MINSIZE = 84

background_canvas = create_gradient_background(main)

main.grid_rowconfigure(0, weight=0, minsize=HEADER_ROW_MINSIZE)
main.grid_rowconfigure(1, weight=1)
main.grid_columnconfigure(0, weight=20)
main.grid_columnconfigure(1, weight=55)
main.grid_columnconfigure(2, weight=25)

background_canvas.grid(row=0, column=0, rowspan=2, columnspan=3, sticky="nsew")
background_canvas.lower("all")   

header_shell = Frame(main, bg=bg_start, highlightbackground=accent_cyan, highlightthickness=1)
header_shell.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=(8, 4))
header_inner = Frame(header_shell, bg=bg_start, highlightthickness=0)
header_inner.pack(fill=BOTH, expand=True, padx=3, pady=3)
header_inner.grid_columnconfigure(0, weight=1)
header_inner.grid_columnconfigure(1, weight=0)

_hdr_bg = bg_start

hdr_left = Frame(header_inner, bg=_hdr_bg)
hdr_left.grid(row=0, column=0, sticky="w", padx=(16, 12), pady=10)

_hdr_title = Label(
    hdr_left,
    text="Automated Road Damage Detection",
    bg=_hdr_bg,
    fg="#FFFFFF",
    font=("Segoe UI", 22, "bold"),
)
_hdr_title.pack(anchor="w")

_hdr_sub = Label(
    hdr_left,
    text="AI-Powered analysis of UAV images using deep learning",
    bg=_hdr_bg,
    fg=muted_text,
    font=("Segoe UI", 11),
)
_hdr_sub.pack(anchor="w", pady=(6, 0))

hdr_right = Frame(header_inner, bg=_hdr_bg)
hdr_right.grid(row=0, column=1, sticky="e", padx=(12, 16), pady=10)

status_row = Frame(hdr_right, bg=_hdr_bg)
status_row.pack(anchor="e")

_hdr_dot = Label(
    status_row,
    text="●",
    fg=neon_green,
    bg=_hdr_bg,
    font=("Segoe UI", 11),
)
_hdr_dot.pack(side=LEFT)

_hdr_ready = Label(
    status_row,
    text="System Ready",
    fg=text_dark,
    bg=_hdr_bg,
    font=("Segoe UI", 10, "bold"),
)
_hdr_ready.pack(side=LEFT, padx=(6, 0))

_hdr_progress = Label(
    status_row,
    text="",
    fg=accent_cyan,
    bg=_hdr_bg,
    font=("Segoe UI", 9, "italic"),
)
_hdr_progress.pack(side=LEFT, padx=(10, 0))

time_row = Frame(hdr_right, bg=_hdr_bg)
time_row.pack(anchor="e", pady=(6, 0))

_hdr_tcap = Label(
    time_row,
    text="Live Time",
    bg=_hdr_bg,
    fg=muted_text,
    font=("Segoe UI", 8),
)
_hdr_tcap.pack(side=LEFT, padx=(0, 10))

_time_header_label = Label(
    time_row,
    text="",
    fg=text_dark,
    bg=_hdr_bg,
    font=("Segoe UI", 11, "bold"),
)
_time_header_label.pack(side=LEFT)

date_row = Frame(hdr_right, bg=_hdr_bg)
date_row.pack(anchor="e", pady=(4, 0))

_hdr_dcap = Label(
    date_row,
    text="Date",
    bg=_hdr_bg,
    fg=muted_text,
    font=("Segoe UI", 8),
)
_hdr_dcap.pack(side=LEFT, padx=(0, 10))

_date_header_label = Label(
    date_row,
    text="",
    fg=text_dark,
    bg=_hdr_bg,
    font=("Segoe UI", 10),
)
_date_header_label.pack(side=LEFT)


def tick_clock():
    now = datetime.datetime.now()
    if _time_header_label is not None:
        _time_header_label.config(text=now.strftime("%H:%M:%S"))
    if _date_header_label is not None:
        _date_header_label.config(text=now.strftime("%d %b %Y"))
    main.after(1000, tick_clock)


now0 = datetime.datetime.now()
if _time_header_label is not None:
    _time_header_label.config(text=now0.strftime("%H:%M:%S"))
if _date_header_label is not None:
    _date_header_label.config(text=now0.strftime("%d %b %Y"))
tick_clock()

sidebar_card = create_card(main, glow=border_color)
sidebar_card.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))

sidebar_body = Frame(sidebar_card, bg=card_bg)
sidebar_body.pack(fill=BOTH, expand=True, padx=10, pady=12)


def _sidebar_section(title, first=False):
    Label(
        sidebar_body,
        text=title,
        bg=card_bg,
        fg=muted_text,
        font=("Segoe UI", 8, "bold"),
    ).pack(anchor="w", pady=(4 if first else 14, 6))


_sidebar_section("WORKFLOW", first=True)
uploadButton = create_sidebar_button(sidebar_body, "Upload Dataset", UploadDataset)
uploadButton.pack(fill=X, pady=2)
preprocessButton = create_sidebar_button(sidebar_body, "Preprocess Data", PreprocessDataset)
preprocessButton.pack(fill=X, pady=2)
SplitButton = create_sidebar_button(sidebar_body, "Split Dataset", split)
SplitButton.pack(fill=X, pady=2)

_sidebar_section("MODELS")
RunYOLOV5Button = create_sidebar_button(sidebar_body, "Run YOLOv5", RunYOLOV5)
RunYOLOV5Button.pack(fill=X, pady=2)
RunYOLOV7Button = create_sidebar_button(sidebar_body, "Run YOLOv7", RunYOLOV7)
RunYOLOV7Button.pack(fill=X, pady=2)
RunYOLOV8Button = create_sidebar_button(sidebar_body, "Run YOLOv8", RunYOLOV8)
RunYOLOV8Button.pack(fill=X, pady=2)

_sidebar_section("ANALYSIS")
comparisonGraphButton = create_sidebar_button(sidebar_body, "Compare Models", comparisonGraph)
comparisonGraphButton.pack(fill=X, pady=2)

_sidebar_section("TESTING")
damageDetectionButton = create_sidebar_button(sidebar_body, "Upload Test Images", damageDetection)
damageDetectionButton.pack(fill=X, pady=2)

center_frame = Frame(main, bg=bg_start, highlightthickness=0)
center_frame.grid(row=1, column=1, sticky="nsew", padx=4, pady=(0, 8))
center_frame.grid_rowconfigure(0, weight=7)
center_frame.grid_rowconfigure(1, weight=0)
center_frame.grid_rowconfigure(2, weight=2)
center_frame.grid_columnconfigure(0, weight=1)

preview_top = Frame(center_frame, bg=bg_start, highlightthickness=0)
preview_top.grid(row=0, column=0, sticky="nsew", padx=4, pady=(0, 6))
preview_top.grid_columnconfigure(0, weight=1)
preview_top.grid_columnconfigure(1, weight=1)
preview_top.grid_rowconfigure(0, weight=1)

image_placeholder = Label(
    preview_top,
    text="No Image Loaded",
    bg=bg_start,
    fg=muted_text,
    font=("Segoe UI", 15, "bold"),
)
image_placeholder.grid(row=0, column=0, columnspan=2, sticky="nsew")

original_card = create_card(preview_top, glow=accent_cyan)
card_title_orig = Label(
    original_card,
    text="Original Image",
    bg=card_bg,
    fg=text_dark,
    font=("Segoe UI", 10, "bold"),
    anchor="w",
    justify="left",
)
card_title_orig.pack(anchor="w", fill=X, padx=12, pady=(10, 4))
original_inner = Frame(original_card, bg=card_bg)
original_inner.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))
original_label = Label(original_inner, bg=card_bg, anchor="center")
original_label.pack(expand=True, fill=BOTH)

detected_card = create_card(preview_top, glow=primary)
card_title_det = Label(
    detected_card,
    text="Detected Damage",
    bg=card_bg,
    fg=text_dark,
    font=("Segoe UI", 10, "bold"),
    anchor="w",
    justify="left",
)
card_title_det.pack(anchor="w", fill=X, padx=12, pady=(10, 4))
detected_inner = Frame(detected_card, bg=card_bg)
detected_inner.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))
detected_label = Label(detected_inner, bg=card_bg, anchor="center")
detected_label.pack(expand=True, fill=BOTH)

original_card.grid_remove()
detected_card.grid_remove()

stepper_card = create_card(center_frame, glow=border_color)
stepper_card.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
step_inner = Frame(stepper_card, bg=card_bg)
step_inner.pack(fill=X, padx=10, pady=10)

_workflow_cells.clear()
_step_names = ["Upload", "Preprocess", "Split", "Train", "Detect"]
for si, nm in enumerate(_step_names):
    cell = Frame(step_inner, bg=card_bg)
    cell.pack(side=LEFT, expand=True, fill=X, padx=2)
    dot = Label(cell, text="●", bg=card_bg, fg=secondary, font=("Segoe UI", 10))
    dot.pack(side=LEFT, padx=(0, 4))
    name_lbl = Label(
        cell,
        text=f"{si + 1} {nm}",
        bg=card_bg,
        fg=muted_text,
        font=("Segoe UI", 8, "bold"),
    )
    name_lbl.pack(side=LEFT)
    _workflow_cells.append(
        {"step": si + 1, "dot": dot, "name": name_lbl, "state": "pending"},
    )

bottom_row = Frame(center_frame, bg=bg_start, highlightthickness=0)
bottom_row.grid(row=2, column=0, sticky="nsew", padx=2, pady=(2, 0))
bottom_row.grid_columnconfigure(0, weight=1)
bottom_row.grid_rowconfigure(0, weight=1)
bottom_row.grid_propagate(False)


def _rebalance_center_layout(_event=None):
    """Cap tabs height so preview area remains dominant and visible."""
    total_h = max(center_frame.winfo_height(), 1)
    max_tabs_h = max(200, min(300, int(total_h * 0.34)))
    bottom_row.configure(height=max_tabs_h)


center_frame.bind("<Configure>", _rebalance_center_layout)
main.after(0, _rebalance_center_layout)

tabs_shell = create_card(bottom_row, glow=accent_cyan)
tabs_shell.grid(row=0, column=0, sticky="nsew")
tabs_shell.grid_rowconfigure(1, weight=1)
tabs_shell.grid_columnconfigure(0, weight=1)

tab_bar = Frame(tabs_shell, bg=card_bg)
tab_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))

tab_body = Frame(tabs_shell, bg=PLOT_BG, highlightthickness=0)
tab_body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
tab_body.grid_rowconfigure(0, weight=1)
tab_body.grid_columnconfigure(0, weight=1)

f_log = Frame(tab_body, bg=PLOT_BG)
f_graphs = Frame(tab_body, bg=PLOT_BG)
f_conf = Frame(tab_body, bg=PLOT_BG)
for f in (f_log, f_graphs, f_conf):
    f.grid(row=0, column=0, sticky="nsew")

def _style_tab_btn(btn, active):
    if active:
        btn.config(
            bg=card_inner,
            fg=accent_cyan,
            highlightbackground=accent_cyan,
            highlightthickness=1,
        )
    else:
        btn.config(
            bg=card_bg,
            fg=muted_text,
            highlightbackground=border_color,
            highlightthickness=1,
        )


def _select_tab(which):
    if which == "log":
        f_log.tkraise()
    elif which == "graphs":
        f_graphs.tkraise()
    else:
        f_conf.tkraise()
    for child in tab_bar.winfo_children():
        if isinstance(child, Button) and hasattr(child, "tab_id"):
            _style_tab_btn(child, child.tab_id == which)


def _make_tab(text, tid):
    b = Button(
        tab_bar,
        text=text,
        font=("Segoe UI", 9, "bold"),
        bd=0,
        padx=10,
        pady=7,
        cursor="hand2",
        highlightthickness=1,
        command=lambda t=tid: _select_tab(t),
    )
    b.tab_id = tid
    _style_tab_btn(b, False)
    b.pack(side=LEFT, padx=(0, 6))
    return b


b_log = _make_tab("Execution Log", "log")
_make_tab("Graphs", "graphs")
_make_tab("Confusion Matrix", "conf")
_select_tab("log")

log_inner = Frame(f_log, bg=PLOT_BG)
log_inner.pack(fill=BOTH, expand=True, padx=4, pady=4)
text = Text(
    log_inner,
    bg=PLOT_BG,
    fg=TEXT,
    font=("Consolas", 11),
    bd=0,
    padx=8,
    pady=6,
    insertbackground=TEXT,
    relief="flat",
    highlightthickness=0,
)
scroll = Scrollbar(
    log_inner,
    command=text.yview,
    relief="flat",
    bd=0,
    highlightthickness=0,
    bg=PLOT_BG,
    troughcolor=card_bg,
    activebackground=secondary,
    width=12,
)
text.configure(yscrollcommand=scroll.set)
text.pack(side=LEFT, fill=BOTH, expand=True)
scroll.pack(side=RIGHT, fill="y", padx=(4, 0))

# Right rail: metrics (top) + model info (bottom) — full dashboard column
right_rail = Frame(main, bg=bg_start, highlightthickness=0)
right_rail.grid(row=1, column=2, sticky="nsew", padx=(4, 8), pady=(0, 8))
right_rail.grid_rowconfigure(0, weight=1)
right_rail.grid_rowconfigure(1, weight=1)
right_rail.grid_columnconfigure(0, weight=1)

metrics_card = create_card(right_rail, glow=primary)
metrics_card.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

metrics_inner = Frame(metrics_card, bg=card_bg)
metrics_inner.pack(fill=BOTH, expand=True, padx=10, pady=12)

model_card = create_card(right_rail, glow=border_color)
model_card.grid(row=1, column=0, sticky="nsew")
model_inner = Frame(model_card, bg=card_bg)
model_inner.pack(fill=BOTH, expand=True, padx=10, pady=10)

_model_in_use_label = Label(
    model_inner,
    text="Model in Use: YOLOv8",
    bg=card_bg,
    fg=text_dark,
    font=("Segoe UI", 11, "bold"),
)
_model_in_use_label.pack(anchor="w", pady=(0, 8))

Label(
    model_inner,
    text="Damage Types",
    bg=card_bg,
    fg=muted_text,
    font=("Segoe UI", 9, "bold"),
).pack(anchor="w", pady=(0, 4))

for line in ("Crack", "Pothole", "Patch", "Alligator Crack"):
    Label(
        model_inner,
        text="  •  " + line,
        bg=card_bg,
        fg=text_dark,
        font=("Segoe UI", 10),
    ).pack(anchor="w", pady=1)

Label(
    metrics_inner,
    text="Metrics",
    bg=card_bg,
    fg=text_dark,
    font=("Segoe UI", 12, "bold"),
).pack(anchor="w", pady=(0, 8))

_c1, _m_map = create_metric_card(metrics_inner, "mAP@0.5", "No Data")
_c1.pack(fill=X, pady=6)
_metric_value_labels["map"] = _m_map

_c2, _m_p = create_metric_card(metrics_inner, "Precision", "No Data")
_c2.pack(fill=X, pady=6)
_metric_value_labels["precision"] = _m_p

_c3, _m_r = create_metric_card(metrics_inner, "Recall", "No Data")
_c3.pack(fill=X, pady=6)
_metric_value_labels["recall"] = _m_r

_c4, _m_f = create_metric_card(metrics_inner, "F1 Score", "No Data")
_c4.pack(fill=X, pady=6)
_metric_value_labels["f1"] = _m_f

set_workflow_state(
    {1: "pending", 2: "pending", 3: "pending", 4: "pending", 5: "pending"},
)

main.mainloop()


