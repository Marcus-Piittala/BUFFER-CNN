import os
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------
# 1) Indstillinger
# --------------------------
DATA_DIR = "bird_audio_dataset"                 # <- ret til jeres mappe
CLASS_NAMES = None                # None = auto (mapperne i DATA_DIR)
SR = 22050                        # sample rate
CLIP_SECONDS = 4.0                # fast kliplængde
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16
EPOCHS = 30
RANDOM_SEED = 42


# --------------------------
# 2) Hjælpefunktioner
# --------------------------
def list_files_and_labels(root_dir, class_names=None):
    if class_names is None:
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    paths, labels = [], []
    for c in class_names:
        cdir = os.path.join(root_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                paths.append(os.path.join(cdir, fn))
                labels.append(c)
    return paths, labels, class_names


def load_audio_fixed(path, sr, clip_seconds):
    """Loader lyd og gør den præcis clip_seconds lang (pad/truncate)."""
    n_samples = int(sr * clip_seconds)

    # librosa.load bruger soundfile/audioread og er ret robust til "besværlige" filer
    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    else:
        y = y[:n_samples]
    return y


def audio_to_mel_db(y, sr, n_mels, n_fft, hop_length):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # shape: (n_mels, time)


def normalize_per_sample(x, eps=1e-6):
    """Standardiserer hvert spektrogram (mean=0, std=1) for stabil træning."""
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (sd + eps)


def build_cnn(input_shape, num_classes):
    # En lidt mere “rigtig” CNN til spektrogrammer end jeres helt lille eksempel
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D(2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# --------------------------
# 3) Byg dataset (X, y)
# --------------------------
paths, labels_str, class_names = list_files_and_labels(DATA_DIR, CLASS_NAMES)
print("Klasser:", class_names)
print("Antal filer:", len(paths))

le = LabelEncoder()
y = le.fit_transform(labels_str)  # 0..K-1

# Lav mel-spektrogrammer i en liste
X_list = []
for p in paths:
    audio = load_audio_fixed(p, SR, CLIP_SECONDS)
    mel_db = audio_to_mel_db(audio, SR, N_MELS, N_FFT, HOP_LENGTH)
    mel_db = normalize_per_sample(mel_db)

    # CNN forventer (H, W, C) => (n_mels, time, 1)
    mel_db = mel_db.astype(np.float32)[..., np.newaxis]
    X_list.append(mel_db)

X = np.stack(X_list, axis=0)
print("X shape:", X.shape, "y shape:", y.shape)

# Train/val split (stratify holder klassefordeling)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------
# 4) Træn model
# --------------------------
num_classes = len(class_names)
input_shape = X.shape[1:]  # (n_mels, time, 1)

model = build_cnn(input_shape, num_classes)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Gem model
model.save("bird_cnn_melspec.keras")
print("Gemt som bird_cnn_melspec.keras")


import random
import numpy as np

def predict_random_file(model, data_dir, class_names, top_k=3):
    # Find alle lydfiler under data_dir (samme som træningen)
    audio_paths = []
    for c in class_names:
        cdir = os.path.join(data_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                audio_paths.append(os.path.join(cdir, fn))

    if not audio_paths:
        raise ValueError("Fandt ingen lydfiler i DATA_DIR. Tjek mappestruktur og filendelser.")

    p = random.choice(audio_paths)

    # Preprocess på samme måde som træning
    y = load_audio_fixed(p, SR, CLIP_SECONDS)
    mel_db = audio_to_mel_db(y, SR, N_MELS, N_FFT, HOP_LENGTH)
    mel_db = normalize_per_sample(mel_db).astype(np.float32)[..., np.newaxis]  # (n_mels, time, 1)

    # Model forventer batch-dimension: (1, H, W, C)
    x = np.expand_dims(mel_db, axis=0)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    # Forsøg at udlede "sand" label fra mappenavn (kun hvis stien følger data_dir/klasse/fil.wav)
    true_label = os.path.basename(os.path.dirname(p))

    # Top-k
    top = np.argsort(probs)[::-1][:top_k]

    print("Fil:", p)
    print("Sand label (fra mappe):", true_label)
    print("Model gæt:", pred_label, f"({probs[pred_idx]*100:.1f}%)")
    print("\nTop", top_k, "gæt:")
    for i in top:
        print(f"  {class_names[i]:15s}  {probs[i]*100:5.1f}%")

    return p, true_label, pred_label, probs


# Kør én tilfældig prediction
predict_random_file(model, DATA_DIR, class_names, top_k=3)
