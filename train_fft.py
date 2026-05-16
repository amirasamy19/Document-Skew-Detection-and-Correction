import numpy as np
from utils import train_list, load_image, load_label
from preprocessing_fft import fft_preprocess
from model import build_model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

X, Y = [], []

for img_name in train_list:
    img = load_image(img_name)
    img = fft_preprocess(img)

    X.append(img)
    Y.append(load_label(img_name))

X = np.array(X).reshape(-1, 128, 128, 1) / 255.0
Y = np.array(Y).reshape(-1, 1)

model = build_model()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

model.fit(
    X, Y,
    epochs=50,   
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce]
)

model.save("model_fft.keras")