import numpy as np
import cv2

def fft_preprocess(img):

    img = img.astype(np.float32)

    # FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # mask low frequency
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2

    mask = np.zeros((rows, cols))
    r = 30
    mask[crow-r:crow+r, ccol-r:ccol+r] = 1

    fshift = fshift * mask

    # inverse FFT
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift))
    img_back = np.abs(img_back)

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return img_back.astype(np.uint8)