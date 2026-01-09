# =========================
# Bird audio classifier (Transfer Learning)
# Audio -> Mel-spektrogram -> (Resize + 3-kanal) -> MobileNetV2 -> Fugleart
# =========================

import os
import random
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --------------------------
# 1) Indstillinger (samme idé som din kode)
# --------------------------

DATA_DIR = "bird_audio_dataset"
CLASS_NAMES = None

SR = 22050
CLIP_SECONDS = 4.0

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

BATCH_SIZE = 16
EPOCHS = 30
RANDOM_SEED = 42

# MobileNetV2 forventer typisk "billeder" ~224x224 og 3 kanaler
IMG_SIZE = 224

# Transfer learning knobs
FREEZE_BASE = True          # Start med at fryse MobileNetV2 og træn kun "head"
FINE_TUNE_AT = 100          # Når vi finetuner: lås de første N lag og træn resten (justér efter behov)
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FT = 1e-5     # lav LR til finetuning, ellers "ødelægger" man pretrain


# --------------------------
# 2) Hjælpefunktioner (fra din kode + små tilpasninger)
# --------------------------

def list_files_and_labels(root_dir, class_names=None):
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

    paths, labels = [], []
    for c in class_names:
        cdir = os.path.join(root_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                paths.append(os.path.join(cdir, fn))
                labels.append(c)
    return paths, labels, class_names


def load_audio_fixed(path, sr, clip_seconds):
    n_samples = int(sr * clip_seconds)
    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    else:
        y = y[:n_samples]
    return y


def audio_to_mel_db(y, sr, n_mels, n_fft, hop_length):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (n_mels, time_frames)


def normalize_per_sample(x, eps=1e-6):
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / (sd + eps)


# --------------------------
# 3) Byg dataset (X, y) som i din kode
# --------------------------

paths, labels_str, class_names = list_files_and_labels(DATA_DIR, CLASS_NAMES)
print("Klasser:", class_names)
print("Antal filer:", len(paths))

le = LabelEncoder()
y = le.fit_transform(labels_str)

X_list = []
for p in paths:
    audio = load_audio_fixed(p, SR, CLIP_SECONDS)
    mel_db = audio_to_mel_db(audio, SR, N_MELS, N_FFT, HOP_LENGTH)
    mel_db = normalize_per_sample(mel_db)

    # Her gemmer vi stadig mel som (H, W, 1) float32 ligesom før
    mel_db = mel_db.astype(np.float32)[..., np.newaxis]
    X_list.append(mel_db)

X = np.stack(X_list, axis=0)
print("X shape:", X.shape, "y shape:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(1024, seed=RANDOM_SEED, reshuffle_each_iteration=True)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))


# --------------------------
# 4) Preprocess til MobileNetV2
#    - Mobilenet forventer: (224,224,3) og bestemte input-skalaer
#    - Din X er: (n_mels, time_frames, 1) = (H, W, 1)
#    - Vi: resize -> 224x224, konverter 1 kanal -> 3 kanaler, og preprocess_input
# --------------------------

mobilenet_preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

def to_mobilenet_input(x, y):
    # x: (H, W, 1) float32
    x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE), method="bilinear")

    # 1 kanal -> 3 kanaler (kopiér samme data 3 gange)
    x = tf.image.grayscale_to_rgb(x)  # (224,224,3)

    # MobileNetV2 preprocess_input forventer typisk pixelrange ~[0..255] eller float,
    # men vigtigst er at den skalerer til [-1,1] (intern standard).
    #
    # Vores mel_db er normaliseret (ca. mean 0 std 1), ikke [0..255].
    # Tricket her er: vi "lader" som om det er et billede ved at rescale til et
    # rimeligt interval før preprocess_input.
    #
    # En enkel og robust løsning: klip værdier og map til [0..255]
    x = tf.clip_by_value(x, -3.0, 3.0)              # begræns ekstreme værdier
    x = (x + 3.0) / 6.0                             # -> [0..1]
    x = x * 255.0                                   # -> [0..255]

    x = mobilenet_preprocess(x)                     # -> [-1..1] i den form MobileNetV2 forventer
    return x, y

# Anvend map + batch + prefetch
train_ds = train_ds.map(to_mobilenet_input, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(to_mobilenet_input, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------
# 5) Byg MobileNetV2 transfer learning model
# --------------------------

def build_mobilenetv2_classifier(num_classes, freeze_base=True):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Pretrained backbone (ImageNet)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,          # drop den originale ImageNet classifier
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    if freeze_base:
        base.trainable = False      # frys hele backbone i første fase

    x = base(inputs, training=False)  # training=False så BN-lag kører i inference-mode når frozen

    # I stedet for Flatten (som kan blive kæmpe), brug GlobalAveragePooling2D
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, base


num_classes = len(class_names)
model, base_model = build_mobilenetv2_classifier(num_classes, freeze_base=FREEZE_BASE)
model.summary()


# --------------------------
# 6) Træning fase 1: Train head (frozen backbone)
# --------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        patience=6,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

history_1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)


# --------------------------
# 7) (Valgfrit men anbefalet) Fine-tuning fase 2
#    - Hvis I har nok data pr. klasse, kan I ofte få et løft ved at finetune de sidste lag.
# --------------------------

# Slå finetuning til ved at unfreeze backbone (helt eller delvist)
base_model.trainable = True

# Lås de tidlige lag (generelle features), og træn de senere (mere specialiserede)
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

# Re-compile med lav learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FT),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=max(5, EPOCHS // 2),  # typisk færre epochs i finetuning
    callbacks=callbacks
)

model.save("bird_mobilenetv2_melspec.keras")
print("Gemt som bird_mobilenetv2_melspec.keras")


# --------------------------
# 8) Predict på en tilfældig fil (samme idé som din)
#    (Vi genbruger din mel-pipeline, men går gennem to_mobilenet_input preprocessing)
# --------------------------

def predict_random_file_mobilenet(model, data_dir, class_names, top_k=3):
    audio_paths = []
    for c in class_names:
        cdir = os.path.join(data_dir, c)
        for fn in os.listdir(cdir):
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                audio_paths.append(os.path.join(cdir, fn))
    if not audio_paths:
        raise ValueError("Fandt ingen lydfiler i DATA_DIR.")

    p = random.choice(audio_paths)

    y_audio = load_audio_fixed(p, SR, CLIP_SECONDS)
    mel_db = audio_to_mel_db(y_audio, SR, N_MELS, N_FFT, HOP_LENGTH)
    mel_db = normalize_per_sample(mel_db).astype(np.float32)[..., np.newaxis]  # (H,W,1)

    # lav et "fake" dataset-element og kør samme preprocess
    x = tf.convert_to_tensor(mel_db)
    x = tf.expand_dims(x, axis=0)  # (1,H,W,1)
    # map-funktion forventer (x,y), så vi bruger dummy label
    x, _ = to_mobilenet_input(x[0], tf.constant(0))
    x = tf.expand_dims(x, axis=0)  # (1,224,224,3)

    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    true_label = os.path.basename(os.path.dirname(p))

    top = np.argsort(probs)[::-1][:top_k]

    print("Fil:", p)
    print("Sand label (fra mappe):", true_label)
    print("Model gæt:", pred_label, f"({probs[pred_idx]*100:.1f}%)")
    print("\nTop", top_k, "gæt:")
    for i in top:
        print(f"  {class_names[i]:15s}  {probs[i]*100:5.1f}%")

    return p, true_label, pred_label, probs


_ = predict_random_file_mobilenet(model, DATA_DIR, class_names, top_k=3)
