# =========================
# Bird audio CNN classifier
# (Audio -> Mel-spektrogram -> CNN -> Fugleart)
# =========================

import os  # bruges til at finde mapper/filer (stier, listdir, join)
import random  # bruges til at vælge en tilfældig fil til test
import numpy as np  # numeriske arrays (pad, mean/std, stack, argmax)
import librosa  # audio-loading + feature-extraction (mel-spektrogram, dB)

from sklearn.model_selection import train_test_split  # split i train/val
from sklearn.preprocessing import LabelEncoder  # labels tekst -> heltal (0..K-1)

import tensorflow as tf  # deep learning + tf.data pipeline
from tensorflow import keras  # Keras API til modeller
from tensorflow.keras import layers  # lag som Conv2D, MaxPool, Dense osv.


# --------------------------
# 1) Indstillinger (hyperparametre)
# --------------------------

DATA_DIR = "bird_audio_dataset"  # mappen hvor dine data ligger (DATA_DIR/klasse/fil.wav)
CLASS_NAMES = None  # None = find klasser automatisk fra undermapper i DATA_DIR

SR = 22050  # sample rate; alle filer resamples til denne frekvens (ens input), dette er angivet som mest optimalt for fugelyde og giver angiveligt de mest optimale mel-spektrogrammer for menneskelig høresans 
CLIP_SECONDS = 4.0  # alle klip gøres præcis 4 sekunder (pad/truncate) for fast input-størrelse

N_MELS = 128  # antal mel-bånd (spektrogrammets "højde")
N_FFT = 1024  # FFT-vinduestørrelse (frekvensopløsning)
HOP_LENGTH = 256  # hop mellem vinduer (tidsopløsning)

BATCH_SIZE = 16  # antal samples pr. trænings-step (mindre = mere støj, større = hurtigere)
EPOCHS = 30  # max antal epochs (EarlyStopping kan stoppe før)
RANDOM_SEED = 42  # gør split reproducérbart (samme train/val hver gang)


# --------------------------
# 2) Hjælpefunktioner
# --------------------------

def list_files_and_labels(root_dir, class_names=None):  # finder alle lydfiler og deres labels
    if class_names is None:  # hvis vi ikke har fået klassenavne manuelt
        class_names = sorted([  # sorter alfabetisk for stabil rækkefølge
            d  # selve mappenavnet bliver klassenavnet
            for d in os.listdir(root_dir)  # list alt i root_dir
            if os.path.isdir(os.path.join(root_dir, d))  # behold kun mapper
        ])

    paths = []  # liste til filstier
    labels = []  # liste til labels (tekst, fx "klippedue")

    for c in class_names:  # loop over hver klassemappe
        cdir = os.path.join(root_dir, c)  # fuld sti til klassemappen
        for fn in os.listdir(cdir):  # loop over filer i klassemappen
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):  # kun lydfiler
                paths.append(os.path.join(cdir, fn))  # gem fuld sti til filen
                labels.append(c)  # gem label = mappenavn (klasse)

    return paths, labels, class_names  # returner filer, labels og klasse-rækkefølge


def load_audio_fixed(path, sr, clip_seconds):  # loader lyd og gør den fast længde
    n_samples = int(sr * clip_seconds)  # antal samples der svarer til clip_seconds

    y, _ = librosa.load(path, sr=sr, mono=True)  # load + resample + mono; _ ignoreres

    if len(y) < n_samples:  # hvis lyd er kortere end ønsket længde
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")  # pad med stilhed (0)
    else:  # hvis lyd er længere end ønsket længde
        y = y[:n_samples]  # klip ned til n_samples (første del)

    return y  # returnér 1D waveform med fast længde, fremragende når det enten er tilføjet padding eller downsized hvis filen er længere end 4 sek


def audio_to_mel_db(y, sr, n_mels, n_fft, hop_length):  # waveform -> mel-spektrogram (dB)
    mel = librosa.feature.melspectrogram(  # beregn mel-spektrogram (power)
        y=y,  # waveform
        sr=sr,  # sample rate
        n_mels=n_mels,  # antal mel-bånd
        n_fft=n_fft,  # FFT vindue
        hop_length=hop_length,  # hop mellem vinduer
        power=2.0  # power-spektrogram (energi)
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # log-skala (dB) for mere stabil læring
    return mel_db  # form: (n_mels, time_frames)


def normalize_per_sample(x, eps=1e-6):  # standardiserer ét spektrogram (mean=0, std=1)
    mu = np.mean(x)  # gennemsnit af alle værdier i spektrogrammet
    sd = np.std(x)  # standardafvigelse
    return (x - mu) / (sd + eps)  # normaliser; eps undgår division med 0


def build_cnn(input_shape, num_classes):  # bygger en CNN til spektrogram-billeder
    inputs = keras.Input(shape=input_shape)  # input-lag: (H, W, C) fx (128, T, 1)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)  # lær simple mønstre
    x = layers.MaxPool2D(2)(x)  # downsample (robust + hurtigere)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)  # flere filtre = mere kompleksitet
    x = layers.MaxPool2D(2)(x)  # downsample igen

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)  # endnu dybere features
    x = layers.MaxPool2D(2)(x)  # downsample igen

    x = layers.Flatten()(x)  # flad ud til 1D vektor (til Dense-lag)
    x = layers.Dense(128, activation="relu")(x)  # lær kombinationer af features
    x = layers.Dropout(0.3)(x)  # regularisering mod overfitting

    outputs = layers.Dense(num_classes, activation="softmax")(x)  # sandsynlighed pr. klasse
    model = keras.Model(inputs, outputs)  # saml input->output til en Keras model

    model.compile(  # vælg optimizer + loss + metrics
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),  # Adam er en god standard
        loss="sparse_categorical_crossentropy",  # labels er heltal (0..K-1)
        metrics=["accuracy"]  # vis accuracy under træning
    )

    return model  # returnér den kompilerede model


# --------------------------
# 3) Byg dataset (X, y)
# --------------------------

paths, labels_str, class_names = list_files_and_labels(DATA_DIR, CLASS_NAMES)  # find filer+labels
print("Klasser:", class_names)  # sanity check: hvilke klasser fandt vi?
print("Antal filer:", len(paths))  # sanity check: hvor mange filer fandt vi?

le = LabelEncoder()  # opret encoder til labels
y = le.fit_transform(labels_str)  # omdan label-tekster til tal (0..K-1) i samme rækkefølge

X_list = []  # her samler vi alle spektrogrammer
for p in paths:  # loop over alle lydfiler
    audio = load_audio_fixed(p, SR, CLIP_SECONDS)  # load + pad/truncate til fast længde
    mel_db = audio_to_mel_db(audio, SR, N_MELS, N_FFT, HOP_LENGTH)  # mel-spektrogram i dB
    mel_db = normalize_per_sample(mel_db)  # standardiser per sample

    mel_db = mel_db.astype(np.float32)[..., np.newaxis]  # gør dtype float32 + tilføj kanal (C=1)
    X_list.append(mel_db)  # gem spektrogrammet i listen

X = np.stack(X_list, axis=0)  # lav til én stor tensor: (N, n_mels, time, 1)
print("X shape:", X.shape, "y shape:", y.shape)  # sanity check: matcher antal samples?

X_train, X_val, y_train, y_val = train_test_split(  # split data i train/val
    X,  # inputs
    y,  # labels
    test_size=0.2,  # 20% til validation
    random_state=RANDOM_SEED,  # reproducérbart split
    stratify=y  # bevar klassefordeling i train og val
)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))  # lav dataset af train arrays
train_ds = train_ds.shuffle(1024)  # shuffle så modellen ikke ser klasser i blokke
train_ds = train_ds.batch(BATCH_SIZE)  # batcher samples
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # forbedrer performance

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))  # validation dataset
val_ds = val_ds.batch(BATCH_SIZE)  # batch for eval
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)  # performance


# --------------------------
# 4) Træn model
# --------------------------

num_classes = len(class_names)  # antal klasser = antal mapper
input_shape = X.shape[1:]  # formen på ét sample: (n_mels, time, 1)

model = build_cnn(input_shape, num_classes)  # byg CNN’en med korrekt input + output størrelse
model.summary()  # print model-arkitektur (godt til rapport)

callbacks = [  # callbacks = ekstra logik under træning
    keras.callbacks.EarlyStopping(  # stopper tidligt hvis validation ikke forbedrer sig
        patience=6,  # antal epochs uden forbedring før stop
        restore_best_weights=True  # gå tilbage til bedste weights
    ),
]

history = model.fit(  # træner modellen
    train_ds,  # træningsdata
    validation_data=val_ds,  # validation-data
    epochs=EPOCHS,  # max epochs
    callbacks=callbacks  # early stopping
)

model.save("bird_cnn_melspec.keras")  # gem modellen til disk
print("Gemt som bird_cnn_melspec.keras")  # bekræft output






# --------------------------
# 5) Predict på en tilfældig fil (hurtig “virker det?” test)
# --------------------------

def predict_random_file(model, data_dir, class_names, top_k=3):  # gæt på en tilfældig lydfil
    audio_paths = []  # liste med alle lydfiler vi kan vælge imellem

    for c in class_names:  # loop over alle klasser
        cdir = os.path.join(data_dir, c)  # sti til klassemappen
        for fn in os.listdir(cdir):  # loop over filer i klassemappen
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):  # lydfil?
                audio_paths.append(os.path.join(cdir, fn))  # gem fuld sti

    if not audio_paths:  # hvis vi ikke fandt nogen filer
        raise ValueError("Fandt ingen lydfiler i DATA_DIR. Tjek mappestruktur og filendelser.")  # stop med fejl

    p = random.choice(audio_paths)  # vælg én tilfældig fil

    y_audio = load_audio_fixed(p, SR, CLIP_SECONDS)  # preprocess: load + fast længde
    mel_db = audio_to_mel_db(y_audio, SR, N_MELS, N_FFT, HOP_LENGTH)  # preprocess: mel-spektrogram
    mel_db = normalize_per_sample(mel_db).astype(np.float32)[..., np.newaxis]  # preprocess: norm + (H,W,1)

    x = np.expand_dims(mel_db, axis=0)  # tilføj batch-dimension: (1, H, W, 1)

    probs = model.predict(x, verbose=0)[0]  # predict sandsynligheder for hver klasse (første batch-element)
    pred_idx = int(np.argmax(probs))  # vælg index med højeste sandsynlighed
    pred_label = class_names[pred_idx]  # map index -> klassenavn (samme rækkefølge som class_names)

    true_label = os.path.basename(os.path.dirname(p))  # sand label = mappenavn som filen ligger i

    top = np.argsort(probs)[::-1][:top_k]  # sorter sandsynligheder desc og tag top_k

    print("Fil:", p)  # print hvilken fil vi testede
    print("Sand label (fra mappe):", true_label)  # print sand label (fra folder)
    print("Model gæt:", pred_label, f"({probs[pred_idx]*100:.1f}%)")  # print model gæt + confidence

    print("\nTop", top_k, "gæt:")  # print top_k liste
    for i in top:  # loop over top indices
        print(f"  {class_names[i]:15s}  {probs[i]*100:5.1f}%")  # print navn + %

    return p, true_label, pred_label, probs  # returnér detaljer (kan bruges til statistik)


_ = predict_random_file(model, DATA_DIR, class_names, top_k=3)  # kør én test; '_' skjuler return i notebook
