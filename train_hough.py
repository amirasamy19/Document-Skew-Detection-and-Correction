import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import joblib

from utils import train_list, load_image, load_label
from preprocessing_hough import hough_transform_features

X, y = [], []

for img_name in train_list:

    img = load_image(img_name)
    feat = hough_transform_features(img)
    label = load_label(img_name)

    if label is None:
        continue

    X.append(feat)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Dataset shape:", X.shape, y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_val)

mse = mean_squared_error(y_val, pred)
mae = mean_absolute_error(y_val, pred)

print("🏆 XGBoost Model Results")
print("MSE:", mse)
print("MAE:", mae)

joblib.dump(model, "hough_xgb_model.pkl")
print("Model saved ✔")