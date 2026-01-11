import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     MaxPooling1D, Flatten, Dense, Dropout, Activation)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import librosa, gc, soundata


# ============================================================
# Dataset UrbanSound8k
# ============================================================
dataset = soundata.initialize('urbansound8k')
dataset.download()
dataset.validate()

FRAME_SIZE = 16000
HOP = 8000


# ============================================================
# Arquitectura del modelo — 1D CNN (Abdoli et al., 2019)
# ============================================================
def build_1dcnn_16k(input_length=16000, num_classes=10):
    inp = Input(shape=(input_length, 1), name="input_wave")

    x = Conv1D(16, 64, strides=2, padding='valid', name='conv1')(inp)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name='pool1')(x)

    x = Conv1D(32, 32, strides=2, padding='valid', name='conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name='pool2')(x)

    x = Conv1D(64, 16, strides=2, padding='valid', name='conv3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 8, strides=2, padding='valid', name='conv4')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)

    out = Dense(num_classes, activation='softmax', name='softmax')(x)
    return Model(inputs=inp, outputs=out, name="1D_CNN_16k")


# ============================================================
# Etiquetas y Folds
# ============================================================
labels = [dataset.clip(cid).tags.labels[0] for cid in dataset.clip_ids]
le = LabelEncoder().fit(labels)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []


# ============================================================
# Entrenamiento 10-fold sin generador (cargando a RAM por fold)
# ============================================================
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.clip_ids, le.transform(labels))):
    print(f"\n===== Fold {fold+1}/10 =====")

    train_ids = [dataset.clip_ids[i] for i in train_idx]
    val_ids   = [dataset.clip_ids[i] for i in val_idx]

    # ---- Cargar solo los clips del fold en RAM ----
    X_train_list, y_train_list = [], []
    for cid in train_ids:
        clip = dataset.clip(cid)
        y_audio, _ = librosa.load(clip.audio_path, sr=16000, mono=True)
        y_audio = y_audio / (np.max(np.abs(y_audio)) + 1e-9) * 0.5
        label = le.transform([clip.tags.labels[0]])[0]
        for start in range(0, len(y_audio) - FRAME_SIZE + 1, HOP):
            X_train_list.append(y_audio[start:start+FRAME_SIZE])
            y_train_list.append(label)

    X_val_list, y_val_list = [], []
    for cid in val_ids:
        clip = dataset.clip(cid)
        y_audio, _ = librosa.load(clip.audio_path, sr=16000, mono=True)
        y_audio = y_audio / (np.max(np.abs(y_audio)) + 1e-9) * 0.5
        label = le.transform([clip.tags.labels[0]])[0]
        for start in range(0, len(y_audio) - FRAME_SIZE + 1, HOP):
            X_val_list.append(y_audio[start:start+FRAME_SIZE])
            y_val_list.append(label)

    # Convertir a tensores
    X_train = np.array(X_train_list, dtype=np.float32).reshape(-1, FRAME_SIZE, 1)
    X_val   = np.array(X_val_list, dtype=np.float32).reshape(-1, FRAME_SIZE, 1)
    y_train = to_categorical(y_train_list, num_classes=10)
    y_val   = to_categorical(y_val_list, num_classes=10)

    print(f"Train frames: {X_train.shape}, Val frames: {X_val.shape}")

    # Modelo y entrenamiento (igual al paper)
    model = build_1dcnn_16k()
    opt = Adadelta(learning_rate=1.0)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100, batch_size=64, verbose=1)

    scores = model.evaluate(X_val, y_val, verbose=0)
    fold_accuracies.append(scores[1])
    print(f"Fold {fold+1} Accuracy: {scores[1]*100:.2f}%")

    del model, X_train, X_val, y_train, y_val
    gc.collect()

print(f"\n=== Mean Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}% ===")
