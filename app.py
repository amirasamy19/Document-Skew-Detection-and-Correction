import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from preprocessing_fft import fft_preprocess
from preprocessing_classic import classic_preprocess
from preprocessing_hough import hough_transform_features

from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


# =========================
# OCR PATH
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# HELPERS
# =========================
def prepare_image(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, (128, 128))


def rotate_image(img, angle):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def improve_for_ocr(img):
    blur = cv2.GaussianBlur(img, (3,3), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 5
    )


# =========================
# HOUGH VISUALIZATION
# =========================
def hough_with_lines(image):

    edges = canny(image, sigma=2)
    h, theta, d = hough_line(edges)
    _, angles, dists = hough_line_peaks(h, theta, d)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")

    for angle, dist in zip(angles, dists):
        x0 = dist * np.cos(angle)
        y0 = dist * np.sin(angle)
        ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), color='red')

    ax.axis("off")

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return img


# =========================
# LOAD MODELS SAFE
# =========================
@st.cache_resource
def load_models():
    fft_model = None
    classic_model = None
    hough_model = None

    try:
        fft_model = load_model("model_fft.keras")
    except:
        pass

    try:
        classic_model = load_model("model_classic.keras")
    except:
        pass

    try:
        hough_model = joblib.load("hough_xgb_model.pkl")
    except:
        pass

    return fft_model, classic_model, hough_model


fft_model, classic_model, hough_model = load_models()


# =========================
# UI
# =========================
st.title("📄 Document Skew Detection System")

file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])


# =========================
# MAIN
# =========================
if file is not None:

    # -------------------------
    # IMAGE LOAD
    # -------------------------
    image = Image.open(file).convert("L")
    img_np = np.array(image)

    st.image(image, caption="Original Image", width=400)

    # =========================
    # FFT
    # =========================
    fft_img = prepare_image(fft_preprocess(img_np))

    # =========================
    # CLASSIC
    # =========================
    seg_img, classic_img = classic_preprocess(img_np)

    seg_img = prepare_image(seg_img)
    classic_img = prepare_image(classic_img)

    # =========================
    # HOUGH FEATURES
    # =========================
    hough_feat = hough_transform_features(img_np).reshape(1, -1)

    # =========================
    # INPUTS
    # =========================
    fft_in = fft_img.reshape(1,128,128,1)/255.0
    classic_in = classic_img.reshape(1,128,128,1)/255.0

    # =========================
    # PREDICTIONS
    # =========================
    fft_angle = float(fft_model.predict(fft_in)[0][0]) if fft_model else 0
    classic_angle = float(classic_model.predict(classic_in)[0][0]) if classic_model else 0
    hough_angle = float(hough_model.predict(hough_feat)[0]) if hough_model else 0

    # =========================
    # DISPLAY PREPROCESSING
    # =========================
    st.subheader("Preprocessing")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.image(fft_img, caption="FFT")

    with c2:
        st.image(seg_img, caption="Segmentation")

    with c3:
        st.image(classic_img, caption="Classic")

    # =========================
    # HOUGH VISUALIZATION
    # =========================
    st.subheader("Hough Lines")
    st.image(hough_with_lines(img_np))

    # =========================
    # RESULTS
    # =========================
    st.subheader("Predictions")

    st.write(f"FFT: {fft_angle:.2f}")
    st.write(f"Classic: {classic_angle:.2f}")
    st.write(f"Hough: {hough_angle:.2f}")

    # =========================
    # ROTATION
    # =========================
    st.subheader("Corrected Images")

    fft_corr = rotate_image(img_np, fft_angle)
    classic_corr = rotate_image(img_np, classic_angle)
    hough_corr = rotate_image(img_np, hough_angle)

    c4, c5, c6 = st.columns(3)

    with c4:
        st.image(fft_corr, caption="FFT Corrected")

    with c5:
        st.image(classic_corr, caption="Classic Corrected")

    with c6:
        st.image(hough_corr, caption="Hough Corrected")

    # =========================
    # BEST MODEL (ANGLE BASED)
    # =========================
    errors = {
        "FFT": abs(fft_angle),
        "Classic": abs(classic_angle),
        "Hough": abs(hough_angle)
    }

    best = min(errors, key=errors.get)

    st.success(f"🏆 Best Model (Angle Error): {best}")

    # ==================================================
    # ========================= OCR =====================
    # ==================================================

    st.subheader("OCR Results")

    def extract_text(img):
        return pytesseract.image_to_string(img)

    def text_score(text):
        return len([c for c in text if c.isalnum()])

    # OCR images
    fft_ocr_img = improve_for_ocr(fft_corr)
    classic_ocr_img = improve_for_ocr(classic_corr)
    hough_ocr_img = improve_for_ocr(hough_corr)

    # text extraction
    fft_text = extract_text(fft_ocr_img)
    classic_text = extract_text(classic_ocr_img)
    hough_text = extract_text(hough_ocr_img)

    c7, c8, c9 = st.columns(3)

    with c7:
        st.image(fft_ocr_img, caption="FFT OCR")
        st.text_area("FFT Text", fft_text, height=150)

    with c8:
        st.image(classic_ocr_img, caption="Classic OCR")
        st.text_area("Classic Text", classic_text, height=150)

    with c9:
        st.image(hough_ocr_img, caption="Hough OCR")
        st.text_area("Hough Text", hough_text, height=150)

    # =========================
    # BEST MODEL (OCR BASED)
    # =========================
    scores = {
        "FFT": text_score(fft_text),
        "Classic": text_score(classic_text),
        "Hough": text_score(hough_text)
    }

    best_ocr = max(scores, key=scores.get)

    st.subheader("Final Decision (OCR Based)")
    st.write(scores)

    st.success(f"🏆 Best Model (OCR Quality): {best_ocr}")
