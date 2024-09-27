import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the grayscale image
gray_image = cv2.imread('doggy.jpeg', cv2.IMREAD_GRAYSCALE)

# Compute histogram
histogram, _ = np.histogram(gray_image, bins=256, range=(0, 256))

# Normalize histogram
histogram = histogram.astype(float) / sum(histogram)

# Variables for thresholding
total_pixels = len(gray_image.ravel())
sum_total = np.sum(np.arange(256) * histogram)

max_variance = 0
threshold = 0

# Iterate through all possible thresholds
for t in range(256):
    # Class probabilities
    w0 = np.sum(histogram[:t])
    w1 = np.sum(histogram[t:])
    
    if w0 == 0 or w1 == 0:
        continue
    
    # Class means
    mu0 = np.sum(np.arange(t) * histogram[:t]) / w0
    mu1 = np.sum(np.arange(t, 256) * histogram[t:]) / w1
    
    # Class variances
    variance = w0 * w1 * ((mu0 - mu1) ** 2)
    
    # Update threshold if variance is maximum
    if variance > max_variance:
        max_variance = variance
        threshold = t

# Apply thresholding to create binary image
binary_image = np.where(gray_image >= threshold, 255, 0).astype(np.uint8)

# Plot histogram of the image with threshold line
plt.figure(figsize=(10, 6))
plt.bar(range(len(histogram)), histogram, width=1.0, color='b')
plt.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Intensity Bins')
plt.ylabel('Frequency')
plt.title('Intensity Distribution of the Given Image')
plt.legend()
plt.show()

# Display the original and binary images
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Binary Image (Otsu Thresholding)', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
