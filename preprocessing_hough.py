import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks

def hough_transform_features(image):

 
    edges = canny(image, sigma=2)

    tested_angles = np.deg2rad(np.arange(80.0, 100.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    _, angles, _ = hough_line_peaks(h, theta, d)

    if len(angles) == 0:
        return np.zeros(18, dtype=np.float32)

    angles_deg = np.rad2deg(angles)

    hist, _ = np.histogram(
        angles_deg,
        bins=18,
        range=(-90, 90)
    )

    hist = hist.astype(np.float32)

    hist = hist / (np.linalg.norm(hist) + 1e-6)

    return hist


def estimate_angle(img):

    edges = canny(img, sigma=2)

    tested_angles = np.deg2rad(np.arange(80.0, 100.0))
    h, theta, d = hough_line(edges, theta=tested_angles)

    _, angles, _ = hough_line_peaks(h, theta, d)

    if len(angles) == 0:
        return 0.0, 0.0

    angle = np.mean(np.rad2deg(angles))

    fixed = -(90 - angle)

    return angle, fixed