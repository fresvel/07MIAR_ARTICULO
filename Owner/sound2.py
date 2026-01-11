# ============================================================
# 1. Librerías
# ============================================================
import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     MaxPooling1D, Flatten, Dense, Dropout, Activation)
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adadelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import librosa
import soundata

# ============================================================
# 2. Descargar y validar UrbanSound8K
# ============================================================
dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

# ============================================================
# 3. Preprocesamiento a .npy (Solo se ejecuta 1 vez)
# ============================================================
SAVE_DIR = "audio_npy"
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n📦 Preprocesando audios en .npy (solo la primera vez)...")
for cid in dataset.clip_ids:
    out_path = f"{SAVE_DIR}/{cid}.npy"
    if not os.path.exists(out_path):
        clip = dataset.clip(cid)
        y, _ = librosa.load(clip.audio_path, sr=16000, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
        np.save(out_path, y.astype(np.float32))
print("✅ Preprocesamiento completado.\n")

# ============================================================
# 4. Arquitectura del modelo — 1D CNN
# ============================================================
def build_1dcnn_16k(input_length=16000, num_classes=10):
    inp = Input(shape=(input_length, 1), name="input_wave")

    x = Conv1D(16, 64, strides=2, padding='valid')(inp)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8)(x)

    x = Conv1D(32, 32, strides=2, padding='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8)(x)

    x = Conv1D(64, 16, strides=2, padding='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 8, strides=2, padding='valid')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)

    out = Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

# ============================================================
# 5. Nuevo Generador usando .npy en vez de librosa.load
# ============================================================
class UrbanSoundGenerator(Sequence):
    def __init__(self, clip_ids, dataset, label_encoder,
                 batch_size=100, frame_size=16000, hop_size=8000, shuffle=True):
        self.clip_ids = clip_ids
        self.dataset = dataset
        self.le = label_encoder
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.clip_ids) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.clip_ids)

    def __getitem__(self, idx):
        batch_clips = self.clip_ids[idx*self.batch_size:(idx+1)*self.batch_size]
        X_batch, y_batch = [], []
        for cid in batch_clips:
            y = np.load(f"audio_npy/{cid}.npy", mmap_mode='r')  # 🔥 Carga inmediata sin CPU pesado
            label = self.le.transform([self.dataset.clip(cid).tags.labels[0]])[0]

            for start in range(0, len(y) - self.frame_size + 1, self.hop_size):
                frame = y[start:start+self.frame_size]
                X_batch.append(frame.reshape(self.frame_size, 1))
                y_batch.append(label)

        return np.array(X_batch, dtype=np.float32), to_categorical(y_batch, num_classes=10)

# ============================================================
# 6. Cross-Validation
# ============================================================
labels = [dataset.clip(cid).tags.labels[0] for cid in dataset.clip_ids]
le = LabelEncoder().fit(labels)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.clip_ids, le.transform(labels))):
    print(f"\n===== Fold {fold+1}/10 =====")
    train_ids = [dataset.clip_ids[i] for i in train_idx]
    val_ids   = [dataset.clip_ids[i] for i in val_idx]

    train_gen = UrbanSoundGenerator(train_ids, dataset, le, batch_size=100)
    val_gen   = UrbanSoundGenerator(val_ids, dataset, le, batch_size=100, shuffle=False)

    model = build_1dcnn_16k()
    model.compile(optimizer=Adadelta(1.0),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        workers=4,                  # 🔥 Procesa batches en paralelo
        use_multiprocessing=True,   # 🔥 Asegura GPU siempre alimentada
        verbose=1
    )

    score = model.evaluate(val_gen, verbose=0)[1]
    fold_accuracies.append(score)
    print(f"Fold {fold+1} Accuracy: {score*100:.2f}%")

    del model, train_gen, val_gen
    gc.collect()

print(f"\n=== Mean Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}% ===")
