# Document Skew Detection and Correction

Hybrid image processing system for document skew detection and correction using CNN, FFT, Hough Transform, XGBoost, and OCR.

---

## Project Overview

This project focuses on detecting and correcting skewed scanned document images using a hybrid approach that combines image processing, machine learning, and deep learning techniques.

The system improves document readability and OCR performance by automatically estimating and correcting the skew angle.

---

## Problem Statement

Scanned documents often suffer from rotation due to improper scanning or camera angles, which negatively affects:
- Readability
- OCR accuracy
- Document analysis systems

This project aims to automatically detect and correct document skew with high accuracy.

---

## Technologies Used

- Python
- OpenCV
- CNN
- XGBoost
- FFT Processing
- Hough Transform
- OCR
- NumPy
- Matplotlib

---

## Applied Techniques

### Classical Image Processing
- Gaussian Filtering
- CLAHE Contrast Enhancement
- Adaptive Thresholding
- Morphological Operations

### FFT Processing
- Frequency domain analysis
- Dominant text orientation enhancement

### Deep Learning
- CNN regression models for skew angle prediction

### Machine Learning
- XGBoost regression using Hough Transform features

### OCR Evaluation
- Text extraction evaluation after skew correction

---

## Dataset

Dataset contains 600 scanned document images with:
- Different skew angles
- Noise and speckles
- Various document layouts

---

## Results

- Hough Transform + XGBoost achieved the best performance.
- OCR accuracy improved significantly after correction.
- The corrected documents showed better text alignment and readability.

---

## Challenges

- Model selection optimization
- CNN hyperparameter tuning
- Overfitting prevention
- Multi-pipeline system integration

---

## Team Members

- Raghad Hani
- Khadija Mohamed
- Aml Mohamed
- Amira Samy
- Hala Mahmoud

---

## Author

Amira Samy  
AI Student at Pharos University
