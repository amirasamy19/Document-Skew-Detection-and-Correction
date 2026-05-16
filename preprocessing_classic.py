import cv2
import numpy as np

def classic_preprocess(img):

    img = cv2.GaussianBlur(img, (3,3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    seg = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    kernel = np.ones((2,2), np.uint8)
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel)

    lap = cv2.Laplacian(seg, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))

    enhanced = cv2.addWeighted(seg, 0.7, lap, 0.3, 0)

    return seg, enhanced