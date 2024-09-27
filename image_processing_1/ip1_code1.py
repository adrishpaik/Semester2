import cv2
import matplotlib.pyplot as plt
import numpy as np

def histogram_equalization(image):
    # Calculate histogram
    hist = np.zeros(256, dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            hist[intensity] += 1
            
    plt.bar(range(len(hist/(image.shape[0]*image.shape[1]))),hist/(image.shape[0]*image.shape[1]))
    plt.xlabel('Intensity Bins')
    plt.ylabel('frequency')
    plt.title('Intensity distribution of the given image')
    plt.show()

    # Calculate cumulative distribution function (CDF)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]
    
    plt.bar(range(len(cdf/(image.shape[0]*image.shape[1]))),cdf/(image.shape[0]*image.shape[1]))
    plt.xlabel('Intensity Bins')
    plt.ylabel('frequency')
    plt.title('Intensity distribution after histogram equalisation ')
    plt.show()
    
    # Perform histogram equalization
    cdf_min = np.min(cdf)
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            intensity = image[i, j]
            equalized_intensity = (255*(cdf[intensity] - cdf_min)) // (image.shape[0] * image.shape[1])
            equalized_image[i, j] = equalized_intensity
    #print(equalized_image)
    return equalized_image

# Load grayscale image
image = cv2.imread('monkey.jpg', cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_image = histogram_equalization(image)

# Display original and equalized images
cv2.imshow('Original Image', image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
