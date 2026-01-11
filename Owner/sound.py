import soundata

dataset = soundata.initialize('urbansound8k')
dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data


# ============================================================
# 1. Librerías
# ============================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, BatchNormalization,
                                     MaxPooling1D, Flatten, Dense, Dropout, Activation)
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.optimizers import Adadelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import librosa, gc, soundata

# ============================================================
# 2. Dataset UrbanSound8k
# ============================================================
#dataset = soundata.initialize('urbansound8k')
#dataset.download()
#dataset.validate()

# ============================================================
# 3. Arquitectura del modelo — 1D CNN (Abdoli et al., 2019)
# ============================================================
def build_1dcnn_16k(input_length=16000, num_classes=10):
    inp = Input(shape=(input_length, 1), name="input_wave")

    # Conv 1
    x = Conv1D(16, 64, strides=2, padding='valid', name='conv1')(inp)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name='pool1')(x)

    # Conv 2
    x = Conv1D(32, 32, strides=2, padding='valid', name='conv2')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=8, strides=8, name='pool2')(x)

    # Conv 3
    x = Conv1D(64, 16, strides=2, padding='valid', name='conv3')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Conv 4
    x = Conv1D(128, 8, strides=2, padding='valid', name='conv4')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.25)(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    x = Dropout(0.25)(x)

    out = Dense(num_classes, activation='softmax', name='softmax')(x)
    return Model(inputs=inp, outputs=out, name="1D_CNN_16k")

# ============================================================
# 4. Generador de frames con 50 % de solapamiento
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
        for clip_id in batch_clips:
            clip = self.dataset.clip(clip_id)
            y, _ = librosa.load(clip.audio_path, sr=16000, mono=True)
            y = y / (np.max(np.abs(y)) + 1e-9) * 0.5
            for start in range(0, len(y) - self.frame_size + 1, self.hop_size):
                frame = y[start:start + self.frame_size]
                X_batch.append(frame.reshape(self.frame_size, 1))
                y_batch.append(self.le.transform([clip.tags.labels[0]])[0])
        X_batch = np.array(X_batch, dtype=np.float32)
        y_batch = to_categorical(y_batch, num_classes=10)
        return X_batch, y_batch

# ============================================================
# 5. Configuración de folds estratificados (10-fold CV)
# ============================================================
labels = [dataset.clip(cid).tags.labels[0] for cid in dataset.clip_ids]
le = LabelEncoder().fit(labels)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []

# ============================================================
# 6. Entrenamiento fiel al paper
# ============================================================
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset.clip_ids, le.transform(labels))):
    print(f"\n===== Fold {fold+1}/10 =====")
    train_ids = [dataset.clip_ids[i] for i in train_idx]
    val_ids   = [dataset.clip_ids[i] for i in val_idx]

    train_gen = UrbanSoundGenerator(train_ids, dataset, le, batch_size=100)
    val_gen   = UrbanSoundGenerator(val_ids, dataset, le, batch_size=100, shuffle=False)

    model = build_1dcnn_16k()
    opt = Adadelta(learning_rate=1.0)          # ⚠️ Igual que en el paper
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=100, verbose=1)  # ⚠️ 100 épocas

    scores = model.evaluate(val_gen, verbose=0)
    fold_accuracies.append(scores[1])
    print(f"Fold {fold+1} Accuracy: {scores[1]*100:.2f}%")

    del model, train_gen, val_gen
    gc.collect()

print(f"\n=== Mean Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}% ===")