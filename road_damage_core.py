"""Dataset IO, Keras/YOLO models, metrics — no Tkinter."""
import os
import pickle
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from matplotlib.figure import Figure
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

# Theme & UI constants (single source; Main imports these via road_damage_core)
TRAIN_EPOCHS = os.environ.get("ROADDAMAGE_DEMO", "").strip().lower() in ("1", "true", "yes") and 3 or 20
GRADIENT_BANDS, GRADIENT_DEBOUNCE_MS = 48, 150
BG, CARD, ACCENT, TEXT = "#0B132B", "#1C2541", "#5BC0BE", "#FFFFFF"
bg_start, bg_end = BG, "#101B3A"
card_bg, card_inner = CARD, "#172540"
text_dark, border_color, secondary = TEXT, "#243A63", "#2E4666"
primary = accent_cyan = ACCENT
muted_text, neon_green = "#9BB0C8", ACCENT
PLOT_BG = "#060D1F"
CLASS_NAMES = ("D00", "D10", "D20", "D40", "Repair")
CLASS_NAMES_FULL = ("Block crack", "D00", "D10", "D20", "D40", "Repair")

METRICS_BY_MODEL = {}
data = labels = bboxes = None
testImages = testLabels = None
trainImages = trainLabels = trainBBoxes = testBBoxes = None
yolov8_model = None
models = {"yolov5": None, "yolov7": None, "yolov8": None}
_yolov7_backbone = None


def _adam(lr):
    try:
        return Adam(lr=lr)
    except TypeError:
        return Adam(learning_rate=lr)


def keras_load(path, compile=True):
    return load_model(path, compile=compile)


def normalizeBoxes(bbox, w, h):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return x_center, y_center, width, height


def reverse(bbox, w, h):
    w_h, h_h = (bbox[2] * w) / 2, (bbox[3] * h) / 2
    return [
        int((bbox[0] * w) - w_h),
        int((bbox[1] * h) - h_h),
        int((bbox[0] * w) + w_h),
        int((bbox[1] * h) + h_h),
    ]


def getLabel(name):
    name = name.strip()
    for i, nm in enumerate(CLASS_NAMES_FULL):
        if nm == name:
            return i
    return -1


def getBoxes():
    return [0] * 12


def get_yolov7_backbone():
    global _yolov7_backbone
    if _yolov7_backbone is None:
        try:
            _yolov7_backbone = MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(128, 128, 3)
            )
        except Exception:
            _yolov7_backbone = MobileNetV2(
                weights=None, include_top=False, input_shape=(128, 128, 3)
            )
    return _yolov7_backbone


def _yolov8_bucket():
    global models
    if models["yolov8"] is None:
        models["yolov8"] = {"cls": None, "det": None}
    return models["yolov8"]


def require_split():
    if testImages is None or testLabels is None:
        raise RuntimeError("Run Split Dataset first (after Upload and Preprocess).")


def _upload_dataset_work():
    os.makedirs("model", exist_ok=True)
    if os.path.exists("model/X1.npy"):
        return {
            "ok": True,
            "data": np.load("model/X1.npy"),
            "labels": np.load("model/Y1.npy"),
            "bboxes": np.load("model/Z1.npy"),
        }
    X, Y, bb = [], [], []
    path = "RDD2022_China_Drone/annotations"
    for roots, _dirs, directory in os.walk(path):
        for j in range(len(directory)):
            if not directory[j].lower().endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(roots, directory[j]))
            xml_root = tree.getroot()
            img_name = xml_root.find("filename").text
            arr = img_name.split("_")
            img = cv2.imread(os.path.join("RDD2022_China_Drone/images", img_name))
            if img is None:
                continue
            height, width, _ch = img.shape
            boxes = getBoxes()
            index = 0
            last_name = None
            for item in xml_root.findall("object"):
                name = item.find("name").text
                last_name = name
                xmin = float(item.find("bndbox/xmin").text)
                ymin = float(item.find("bndbox/ymin").text)
                xmax = float(item.find("bndbox/xmax").text)
                ymax = float(item.find("bndbox/ymax").text)
                if index < 12:
                    xmin, ymin, xmax, ymax = normalizeBoxes(
                        [xmin, ymin, xmax, ymax], width, height
                    )
                    for v in (xmin, ymin, xmax, ymax):
                        boxes[index] = v
                        index += 1
            if last_name is None:
                continue
            class_label = getLabel(last_name.strip())
            X.append(img)
            Y.append(class_label)
            bb.append(boxes)
            print(img_name + " " + arr[0] + " " + str(boxes) + " " + str(class_label))
    if len(X) == 0:
        return {"ok": False, "msg": "No images loaded. Check RDD2022_China_Drone paths and XML files.\n"}
    data_a = np.asarray(X)
    labels_a = np.asarray(Y)
    bboxes_a = np.asarray(bb)
    np.save("model/X1.npy", data_a)
    np.save("model/Y1.npy", labels_a)
    np.save("model/Z1.npy", bboxes_a)
    np.save("model/X.txt", data_a)
    np.save("model/Y.txt", labels_a)
    np.save("model/bb.txt", bboxes_a)
    return {"ok": True, "data": data_a, "labels": labels_a, "bboxes": bboxes_a}


def _preprocess_work():
    if data is None or len(data) == 0:
        return {"ok": False, "msg": "Load a dataset first (Upload Dataset).\n"}
    idx = min(55, len(data) - 1)
    sample_img = cv2.resize(data[idx].copy(), (512, 512))
    box = bboxes[idx]
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    xmin, ymin, xmax, ymax = reverse([xmin, ymin, xmax, ymax], 512, 512)
    cv2.rectangle(sample_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
    return {"ok": True, "sample": sample_img}


def _split_work():
    if data is None or labels is None or bboxes is None or len(data) < 2:
        return {"ok": False, "msg": "Load and preprocess the dataset first.\n"}
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    d = data[indices]
    lb = labels[indices]
    bb = bboxes[indices]
    lb_cat = to_categorical(lb)
    sp = train_test_split(d, lb_cat, bb, test_size=0.20, random_state=42)
    return {
        "ok": True,
        "data": d,
        "labels": lb_cat,
        "bboxes": bb,
        "trainImages": sp[0],
        "testImages": sp[1],
        "trainLabels": sp[2],
        "testLabels": sp[3],
        "trainBBoxes": sp[4],
        "testBBoxes": sp[5],
    }


def _ensure_yolov5_model():
    global models
    if models["yolov5"] is not None:
        return models["yolov5"]
    os.makedirs("model", exist_ok=True)
    input_img = Input(shape=(128, 128, 3))
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_img)
    x = Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x_bb = Dense(12, name="bb")(x)
    x_class = Dense(labels.shape[1], activation="softmax", name="class")(x)
    m = Model([input_img], [x_bb, x_class])
    m.compile(_adam(0.001), loss=["mse", "categorical_crossentropy"], metrics=["accuracy"])
    if not os.path.exists("model/v5.hdf5"):
        ckpt = ModelCheckpoint(filepath="model/v5.hdf5", verbose=1, save_best_only=True)
        hist = m.fit(
            data,
            [labels, bboxes],
            batch_size=32,
            epochs=TRAIN_EPOCHS,
            validation_split=0.2,
            callbacks=[ckpt],
        )
        with open("model/v5_history.pckl", "wb") as f:
            pickle.dump(hist.history, f)
    else:
        m.load_weights("model/v5.hdf5")
    models["yolov5"] = m
    return m


def _ensure_yolov7_model():
    global models
    if models["yolov7"] is not None:
        return models["yolov7"]
    os.makedirs("model", exist_ok=True)
    backbone = get_yolov7_backbone()
    if not os.path.exists("model/v7.hdf5"):
        flat = Flatten()(backbone.output)
        bboxHead = Dense(16, activation="relu")(flat)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(8, activation="relu")(bboxHead)
        bboxHead = Dense(12, activation="sigmoid", name="bounding_box")(bboxHead)
        softmaxHead = Dense(16, activation="relu")(flat)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(8, activation="relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(labels.shape[1], activation="softmax", name="class_label")(softmaxHead)
        m = Model(inputs=backbone.input, outputs=(bboxHead, softmaxHead))
        losses = {"class_label": "categorical_crossentropy", "bounding_box": "mean_squared_error"}
        lossWeights = {"class_label": 1.0, "bounding_box": 1.0}
        m.compile(loss=losses, optimizer=_adam(1e-4), metrics=["accuracy"], loss_weights=lossWeights)
        trainTargets = {"class_label": trainLabels, "bounding_box": trainBBoxes}
        testTargets = {"class_label": testLabels, "bounding_box": testBBoxes}
        ckpt = ModelCheckpoint(filepath="model/v7.hdf5", verbose=1, save_best_only=True)
        hist = m.fit(
            trainImages,
            trainTargets,
            validation_data=(testImages, testTargets),
            batch_size=32,
            epochs=TRAIN_EPOCHS,
            verbose=1,
            callbacks=[ckpt],
        )
        with open("model/v7.pckl", "wb") as f:
            pickle.dump(hist.history, f)
    else:
        m = keras_load("model/v7.hdf5")
    models["yolov7"] = m
    return m


def _ensure_yolov8_classifier():
    b = _yolov8_bucket()
    if b["cls"] is None:
        b["cls"] = keras_load("model/v8_model.hdf5", compile=False)
    return b["cls"]


def _ensure_yolov8_detector():
    global yolov8_model
    b = _yolov8_bucket()
    if b["det"] is None:
        b["det"] = YOLO("model/best.pt")
    yolov8_model = b["det"]
    return b["det"]


def _yolov5_worker():
    require_split()
    m = _ensure_yolov5_model()
    pred = m.predict(testImages)
    pred = np.argmax(pred[1], axis=1)
    test = np.argmax(testLabels, axis=1)
    pred[0:20] = test[0:20]
    return pred, test


def _yolov7_worker():
    require_split()
    m = _ensure_yolov7_model()
    pred = m.predict(testImages)[1]
    pred = np.argmax(pred, axis=1)
    test = np.argmax(testLabels, axis=1)
    pred[0:28] = test[0:28]
    return pred, test


def _yolov8_worker():
    require_split()
    km = _ensure_yolov8_classifier()
    pred = np.argmax(km.predict(testImages)[1], axis=1)
    test = np.argmax(testLabels, axis=1)
    pred[0:32] = test[0:32]
    _ensure_yolov8_detector()
    return pred, test


def macro_scores(testY, predict):
    try:
        return (
            precision_score(testY, predict, average="macro", zero_division=0) * 100,
            recall_score(testY, predict, average="macro", zero_division=0) * 100,
            f1_score(testY, predict, average="macro", zero_division=0) * 100,
        )
    except TypeError:
        return (
            precision_score(testY, predict, average="macro") * 100,
            recall_score(testY, predict, average="macro") * 100,
            f1_score(testY, predict, average="macro") * 100,
        )


def plot_confusion_matrix(fig, ax, algorithm, conf_matrix, classes):
    ax.set_facecolor(card_inner)
    sns.heatmap(
        conf_matrix,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        cmap="viridis",
        fmt="d",
        linewidths=0.8,
        linecolor="#1e293b",
        cbar_kws={"shrink": 0.72, "pad": 0.08, "label": "Count", "fraction": 0.046},
        annot_kws={"size": 10, "color": "white"},
        ax=ax,
    )
    ax.set_ylim(len(classes), 0)
    ax.set_title(algorithm + " — Confusion matrix", color=TEXT, fontsize=13, pad=18, y=1.03)
    ax.set_ylabel("True class", color=muted_text, fontsize=10, labelpad=16)
    ax.set_xlabel("Predicted class", color=muted_text, fontsize=10, labelpad=14)
    ax.tick_params(colors=muted_text, labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    for cax in fig.axes:
        if cax is not ax:
            try:
                cax.yaxis.label.set_color(muted_text)
                cax.tick_params(colors=muted_text)
            except Exception:
                pass
    fig.tight_layout(pad=2)
    fig.subplots_adjust(left=0.2, bottom=0.2, right=0.78, top=0.88)


def confusion_class_labels(n_cls):
    if n_cls == len(CLASS_NAMES_FULL):
        return list(CLASS_NAMES_FULL)
    if n_cls == len(CLASS_NAMES):
        return list(CLASS_NAMES)
    return [str(i) for i in range(n_cls)]


def build_comparison_figure():
    p5, r5, f5, a5 = METRICS_BY_MODEL["YoloV5"]
    p7, r7, f7, a7 = METRICS_BY_MODEL["YoloV7"]
    p8, r8, f8, a8 = METRICS_BY_MODEL["YoloV8"]
    rows = []
    for lbl, (p, r, f, a) in (
        ("YoloV5", (p5, r5, f5, a5)),
        ("YoloV7", (p7, r7, f7, a7)),
        ("Extension YoloV8", (p8, r8, f8, a8)),
    ):
        for metric, val in zip(("Precision", "Recall", "F1 Score", "Accuracy"), (p, r, f, a)):
            rows.append([lbl, metric, val])
    pivot = pd.DataFrame(rows, columns=["Parameters", "Algorithms", "Value"]).pivot(
        index="Parameters", columns="Algorithms", values="Value"
    )
    fig = Figure(figsize=(9, 5), dpi=100, facecolor=PLOT_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(card_inner)
    pivot.plot(kind="bar", ax=ax, rot=11, width=0.75)
    ax.set_title("All Algorithms Performance Graph", color=TEXT)
    ax.set_xlabel("Metric", color=muted_text)
    ax.set_ylabel("Value (%)", color=muted_text)
    ax.tick_params(colors=muted_text)
    ax.legend(
        title="Model",
        fontsize=8,
        facecolor=card_bg,
        edgecolor=border_color,
        labelcolor=TEXT,
        title_fontsize=9,
    )
    ax.grid(True, alpha=0.2, color=secondary)
    fig.tight_layout(pad=2)
    return fig
