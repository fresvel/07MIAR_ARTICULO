# ============================================================
# UrbanSound8K — Script COMPLETO
# - Descarga/valida con soundata
# - Cachea audios a .npy (1 vez)
# - Entrena 10-fold CV (TRAIN = frames)
#   * Reporta métricas frame-level (evaluate sobre generador)
#   * Reporta métricas clip-level (agregación por clip: mean softmax)
# - Entrena un modelo FINAL, lo guarda (.keras)
# - Guarda histories (JSON/CSV) y genera gráficas profesionales
# - Guarda reportes y matrices de confusión (clip-level) para final
# ============================================================

import os
import gc
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Flatten, Dense, Dropout, Activation
)
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import librosa
import soundata

# ----------------------------
# 0) Reproducibilidad
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# 1) Paths y configuración
# ----------------------------
SR = 16000
FRAME_SIZE = 16000     # 1 segundo
HOP_SIZE = 8000        # 0.5 segundos -> 50% solapamiento
NUM_CLASSES = 10

EPOCHS = 100
BATCH_SIZE = 100

OUT_DIR = "results_run"
NPY_DIR = "audio_npy"

CV_DIR = os.path.join(OUT_DIR, "cv")
CV_PLOTS_DIR = os.path.join(CV_DIR, "plots")
CV_HIST_DIR = os.path.join(CV_DIR, "histories")
CV_REPORTS_DIR = os.path.join(CV_DIR, "reports")

FINAL_DIR = os.path.join(OUT_DIR, "final_model")
FINAL_PLOTS_DIR = os.path.join(FINAL_DIR, "plots")
FINAL_REPORTS_DIR = os.path.join(FINAL_DIR, "reports")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

for d in [OUT_DIR, NPY_DIR, CV_DIR, CV_PLOTS_DIR, CV_HIST_DIR, CV_REPORTS_DIR,
          FINAL_DIR, FINAL_PLOTS_DIR, FINAL_REPORTS_DIR]:
    ensure_dir(d)

# ============================================================
# 2) Dataset UrbanSound8K (soundata)
# ============================================================
dataset = soundata.initialize("urbansound8k")
dataset.download()
dataset.validate()

# ============================================================
# 3) Cache a .npy (solo 1 vez)
# ============================================================
print("\n📦 Cacheando audios a .npy (solo la primera vez)...")
for cid in dataset.clip_ids:
    out_path = os.path.join(NPY_DIR, f"{cid}.npy")
    if not os.path.exists(out_path):
        clip = dataset.clip(cid)
        y, _ = librosa.load(clip.audio_path, sr=SR, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        np.save(out_path, y.astype(np.float32))
print("✅ Cache .npy completado.\n")

# ============================================================
# 4) Modelo — 1D CNN (Abdoli et al., 2019)
# ============================================================
def build_1dcnn_16k(input_length=FRAME_SIZE, num_classes=NUM_CLASSES):
    inp = Input(shape=(input_length, 1), name="input_wave")

    x = Conv1D(16, 64, strides=2, padding="valid", name="conv1")(inp)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name="pool1")(x)

    x = Conv1D(32, 32, strides=2, padding="valid", name="conv2")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name="pool2")(x)

    x = Conv1D(64, 16, strides=2, padding="valid", name="conv3")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 8, strides=2, padding="valid", name="conv4")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu", name="fc1")(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation="relu", name="fc2")(x)
    x = Dropout(0.25)(x)

    out = Dense(num_classes, activation="softmax", name="softmax")(x)
    return Model(inp, out, name="1D_CNN_16k")

# ============================================================
# 5) Generador (TRAIN/EVAL frame-level)
# ============================================================
class UrbanSoundGenerator(Sequence):
    def __init__(
        self,
        clip_ids,
        dataset,
        label_encoder,
        batch_size=BATCH_SIZE,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        shuffle=True,
        npy_dir=NPY_DIR,
    ):
        self.clip_ids = list(clip_ids)
        self.dataset = dataset
        self.le = label_encoder
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shuffle = shuffle
        self.npy_dir = npy_dir
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.clip_ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            rng = np.random.default_rng(SEED)
            rng.shuffle(self.clip_ids)

    def __getitem__(self, idx):
        batch_clips = self.clip_ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        X_batch, y_batch = [], []

        for cid in batch_clips:
            y = np.load(os.path.join(self.npy_dir, f"{cid}.npy"), mmap_mode="r")
            label = int(self.le.transform([self.dataset.clip(cid).tags.labels[0]])[0])

            for start in range(0, len(y) - self.frame_size + 1, self.hop_size):
                frame = y[start : start + self.frame_size]
                X_batch.append(frame.reshape(self.frame_size, 1))
                y_batch.append(label)

        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = to_categorical(y_batch, num_classes=NUM_CLASSES)
        return X_batch, y_batch

# ============================================================
# 6) Plotting “pro” (matplotlib)
# ============================================================
def plot_history(history_dict, out_prefix, title_suffix=""):
    ensure_dir(os.path.dirname(out_prefix))

    # Loss
    plt.figure(figsize=(10, 5), dpi=140)
    plt.plot(history_dict.get("loss", []), label="Train loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_loss.png")
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 5), dpi=140)
    plt.plot(history_dict.get("accuracy", []), label="Train acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy{title_suffix}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "_acc.png")
    plt.close()

def plot_histories_overlay(histories, key, out_path, title):
    ensure_dir(os.path.dirname(out_path))
    max_len = max(len(h.get(key, [])) for h in histories)
    M = np.full((len(histories), max_len), np.nan, dtype=np.float32)
    for i, h in enumerate(histories):
        arr = np.array(h.get(key, []), dtype=np.float32)
        M[i, : len(arr)] = arr

    mean_curve = np.nanmean(M, axis=0)

    plt.figure(figsize=(10, 5), dpi=140)
    for i in range(M.shape[0]):
        plt.plot(M[i], alpha=0.25)
    plt.plot(mean_curve, linewidth=2.5, label="Mean (folds)")
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, out_path, title="Confusion Matrix"):
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(10, 8), dpi=140)
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ============================================================
# 7) Evaluación CLIP-level (agregación sobre frames)
# ============================================================
def clip_probs_from_npy(model, npy_path, frame_size=FRAME_SIZE, hop_size=HOP_SIZE,
                       batch_frames=256, pad_if_short=True):
    """
    Devuelve:
      - clip_probs: (num_classes,) promedio de probs por frame
      - n_frames: cantidad de frames usados
    pad_if_short:
      - True: si el audio < frame_size, hace zero-pad para producir 1 frame
      - False: se salta ese clip
    """
    y = np.load(npy_path, mmap_mode="r")

    if len(y) < frame_size:
        if not pad_if_short:
            return None, 0
        # zero-pad a 1 segundo
        y_pad = np.zeros((frame_size,), dtype=np.float32)
        y_pad[: len(y)] = y[:]
        X = y_pad.reshape(1, frame_size, 1)
        probs = model.predict(X, verbose=0)[0]
        return probs, 1

    starts = list(range(0, len(y) - frame_size + 1, hop_size))
    n_frames = len(starts)
    if n_frames == 0:
        return None, 0

    sum_probs = None
    processed = 0

    while processed < n_frames:
        chunk = starts[processed : processed + batch_frames]
        X = np.empty((len(chunk), frame_size, 1), dtype=np.float32)
        for i, st in enumerate(chunk):
            X[i, :, 0] = y[st : st + frame_size]
        probs = model.predict(X, verbose=0)  # (chunk, C)

        if sum_probs is None:
            sum_probs = probs.sum(axis=0)
        else:
            sum_probs += probs.sum(axis=0)

        processed += len(chunk)

    clip_probs = sum_probs / n_frames
    return clip_probs, n_frames

def evaluate_clip_level(model, clip_ids, dataset, le, npy_dir=NPY_DIR,
                        frame_size=FRAME_SIZE, hop_size=HOP_SIZE, method="mean",
                        pad_if_short=True):
    """
    method:
      - "mean": argmax(promedio de probs) ✅ recomendado
    """
    y_true, y_pred = [], []
    skipped = 0

    for cid in clip_ids:
        true_name = dataset.clip(cid).tags.labels[0]
        true_label = int(le.transform([true_name])[0])

        npy_path = os.path.join(npy_dir, f"{cid}.npy")
        clip_probs, n_frames = clip_probs_from_npy(
            model, npy_path, frame_size=frame_size, hop_size=hop_size,
            pad_if_short=pad_if_short
        )

        if clip_probs is None:
            skipped += 1
            continue

        if method != "mean":
            raise ValueError("En esta versión completa dejamos method='mean' (más estable).")

        pred_label = int(np.argmax(clip_probs))
        y_true.append(true_label)
        y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_))))
    report = classification_report(
        y_true, y_pred, target_names=list(le.classes_), digits=4, zero_division=0
    )
    return acc, cm, report, skipped

# ============================================================
# 8) Encoder + etiquetas
# ============================================================
labels = [dataset.clip(cid).tags.labels[0] for cid in dataset.clip_ids]
le = LabelEncoder().fit(labels)
y_encoded = le.transform(labels)
class_names = list(le.classes_)

# ============================================================
# 9) 10-Fold CV: frame-level + clip-level
# ============================================================
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

fold_frame_accs = []
fold_clip_accs = []
fold_histories = []

print("🚀 Iniciando 10-fold CV...\n")

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.clip_ids, y_encoded), start=1):
    print(f"===== Fold {fold}/10 =====")

    train_ids = [dataset.clip_ids[i] for i in train_idx]
    val_ids   = [dataset.clip_ids[i] for i in val_idx]

    train_gen = UrbanSoundGenerator(train_ids, dataset, le, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = UrbanSoundGenerator(val_ids, dataset, le, batch_size=BATCH_SIZE, shuffle=False)

    model = build_1dcnn_16k()
    model.compile(
        optimizer=Adadelta(learning_rate=1.0),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    csv_logger = CSVLogger(os.path.join(CV_HIST_DIR, f"fold_{fold:02d}.csv"))

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[csv_logger],
        workers=4,
        use_multiprocessing=True,
    )

    # --------- Frame-level (ventanas) ----------
    frame_acc = float(model.evaluate(val_gen, verbose=0)[1])
    fold_frame_accs.append(frame_acc)

    # --------- Clip-level (agregación por clip) ----------
    clip_acc, cm, rep, skipped = evaluate_clip_level(
        model=model,
        clip_ids=val_ids,
        dataset=dataset,
        le=le,
        method="mean",
        pad_if_short=True,
    )
    fold_clip_accs.append(float(clip_acc))

    # Guardar history JSON
    fold_histories.append(history.history)
    with open(os.path.join(CV_HIST_DIR, f"fold_{fold:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f)

    # Plots por fold
    plot_history(history.history, os.path.join(CV_PLOTS_DIR, f"fold_{fold:02d}"), title_suffix=f" — Fold {fold}")

    # Reporte clip-level por fold
    with open(os.path.join(CV_REPORTS_DIR, f"fold_{fold:02d}_clip_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
        f.write(f"\nSkipped (muy cortos sin frames, si pad_if_short=False): {skipped}\n")
        f.write(f"Clip-level acc: {clip_acc:.6f}\n")

    # (Opcional) Guardar CM por fold
    np.save(os.path.join(CV_REPORTS_DIR, f"fold_{fold:02d}_clip_cm.npy"), cm)

    print(f"Fold {fold} FRAME-level Acc: {frame_acc*100:.2f}%")
    print(f"Fold {fold} CLIP-level  Acc: {clip_acc*100:.2f}% (skipped={skipped})\n")

    del model, train_gen, val_gen
    gc.collect()

# Overlays CV
plot_histories_overlay(
    fold_histories, "val_accuracy",
    os.path.join(CV_PLOTS_DIR, "overlay_val_accuracy.png"),
    "CV Overlay — val_accuracy (folds + mean)"
)
plot_histories_overlay(
    fold_histories, "val_loss",
    os.path.join(CV_PLOTS_DIR, "overlay_val_loss.png"),
    "CV Overlay — val_loss (folds + mean)"
)

# Resumen CV
frame_mean, frame_std = np.mean(fold_frame_accs), np.std(fold_frame_accs)
clip_mean, clip_std = np.mean(fold_clip_accs), np.std(fold_clip_accs)

print("📌 Resumen 10-fold CV")
print(f" - FRAME-level: {frame_mean*100:.2f}% ± {frame_std*100:.2f}%")
print(f" - CLIP-level : {clip_mean*100:.2f}% ± {clip_std*100:.2f}%\n")

with open(os.path.join(CV_DIR, "cv_summary.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "frame_accs": fold_frame_accs,
            "clip_accs": fold_clip_accs,
            "frame_mean": float(frame_mean),
            "frame_std": float(frame_std),
            "clip_mean": float(clip_mean),
            "clip_std": float(clip_std),
        },
        f,
        indent=2
    )

# ============================================================
# 10) Modelo FINAL (fuera del CV) + guardado
# ============================================================
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(dataset.clip_ids, y_encoded))

final_train_ids = [dataset.clip_ids[i] for i in train_idx]
final_val_ids   = [dataset.clip_ids[i] for i in val_idx]

final_train_gen = UrbanSoundGenerator(final_train_ids, dataset, le, batch_size=BATCH_SIZE, shuffle=True)
final_val_gen   = UrbanSoundGenerator(final_val_ids, dataset, le, batch_size=BATCH_SIZE, shuffle=False)

final_model = build_1dcnn_16k()
final_model.compile(
    optimizer=Adadelta(learning_rate=1.0),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

best_path = os.path.join(FINAL_DIR, "urbansound_best.keras")
last_path = os.path.join(FINAL_DIR, "urbansound_last.keras")

ckpt = ModelCheckpoint(
    best_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
final_csv = CSVLogger(os.path.join(FINAL_DIR, "final_history.csv"))

print("🏁 Entrenando modelo FINAL...\n")
final_history = final_model.fit(
    final_train_gen,
    validation_data=final_val_gen,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[ckpt, final_csv],
    workers=4,
    use_multiprocessing=True,
)

# Guardar último estado
final_model.save(last_path)

# Guardar history final
with open(os.path.join(FINAL_DIR, "final_history.json"), "w", encoding="utf-8") as f:
    json.dump(final_history.history, f)

# Plots final
plot_history(final_history.history, os.path.join(FINAL_PLOTS_DIR, "final"), title_suffix=" — Final model")

# ============================================================
# 11) Evaluación FINAL: frame-level y clip-level (BEST + LAST)
# ============================================================
best_model = tf.keras.models.load_model(best_path)

# Frame-level
best_frame_acc = float(best_model.evaluate(final_val_gen, verbose=0)[1])
last_frame_acc = float(final_model.evaluate(final_val_gen, verbose=0)[1])

# Clip-level
best_clip_acc, best_cm, best_rep, best_skipped = evaluate_clip_level(
    model=best_model,
    clip_ids=final_val_ids,
    dataset=dataset,
    le=le,
    method="mean",
    pad_if_short=True,
)
last_clip_acc, last_cm, last_rep, last_skipped = evaluate_clip_level(
    model=final_model,
    clip_ids=final_val_ids,
    dataset=dataset,
    le=le,
    method="mean",
    pad_if_short=True,
)

print("✅ FINAL (Validación)")
print(f" - BEST  frame-level: {best_frame_acc*100:.2f}%")
print(f" - BEST  clip-level : {best_clip_acc*100:.2f}% (skipped={best_skipped})")
print(f" - LAST  frame-level: {last_frame_acc*100:.2f}%")
print(f" - LAST  clip-level : {last_clip_acc*100:.2f}% (skipped={last_skipped})\n")

# Guardar reportes y CM (BEST)
with open(os.path.join(FINAL_REPORTS_DIR, "best_clip_report.txt"), "w", encoding="utf-8") as f:
    f.write(best_rep)
    f.write(f"\nSkipped: {best_skipped}\n")
    f.write(f"Clip-level acc: {best_clip_acc:.6f}\n")

np.save(os.path.join(FINAL_REPORTS_DIR, "best_clip_cm.npy"), best_cm)
plot_confusion_matrix(
    best_cm, class_names,
    os.path.join(FINAL_REPORTS_DIR, "best_clip_confusion_matrix.png"),
    title="Final BEST — Clip-level Confusion Matrix"
)

# Guardar reportes y CM (LAST)
with open(os.path.join(FINAL_REPORTS_DIR, "last_clip_report.txt"), "w", encoding="utf-8") as f:
    f.write(last_rep)
    f.write(f"\nSkipped: {last_skipped}\n")
    f.write(f"Clip-level acc: {last_clip_acc:.6f}\n")

np.save(os.path.join(FINAL_REPORTS_DIR, "last_clip_cm.npy"), last_cm)
plot_confusion_matrix(
    last_cm, class_names,
    os.path.join(FINAL_REPORTS_DIR, "last_clip_confusion_matrix.png"),
    title="Final LAST — Clip-level Confusion Matrix"
)

print("📌 Guardados principales:")
print(" - BEST model:", best_path)
print(" - LAST model:", last_path)
print(" - CV plots:", CV_PLOTS_DIR)
print(" - Final plots:", FINAL_PLOTS_DIR)
print(" - Final reports:", FINAL_REPORTS_DIR)
