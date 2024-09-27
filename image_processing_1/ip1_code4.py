import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hsi(rgb_image):
    r, g, b = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]

    intensity = (r + g + b) / 3.0

    minimum = np.minimum.reduce([r, g, b])
    sum_rgb = r + g + b
    #to encounter division by zero using that clipin denominator
    saturation = np.where(sum_rgb == 0, 0, 1 - 3 * minimum / np.clip(sum_rgb, 1e-10, None))

    #to encounter division by zero using that clipin denominator when R = G = B
    arg = 0.5 * ((r - g) + (r - b)) / np.maximum(1e-10, np.sqrt((r - g)**2 + (r - b) * (g - b)))
    hue = np.arccos(np.clip(arg, -1, 1))
    hue[np.isnan(hue)] = 0  # Replace NaN values with 0

    return hue / (2 * np.pi), saturation, intensity

def hsi_to_rgb(hsi_image):
    hue, saturation, intensity = hsi_image

    r= intensity * (1 - saturation)
    g= intensity * (1 + (saturation * np.cos(2 * np.pi * hue) / np.clip(np.cos(np.pi / 3 - 2 * np.pi * hue), 1e-10, None)))
    b= 3 * intensity - (r + g)

                                                            
    mask = (0 <= hue) & (hue < 1/3)
    b[mask] = intensity[mask] * (1 - saturation[mask])
    r[mask] = intensity[mask] * (1 + (saturation[mask] * np.cos(2 * np.pi * hue[mask]) / np.clip(np.cos(np.pi / 3 - 2 * np.pi * hue[mask]), 1e-10, None)))
    g[mask] = 3 * intensity[mask] - (r[mask] + b[mask])

    mask = (1/3 <= hue) & (hue < 2/3)
    r[mask] = intensity[mask] * (1 - saturation[mask])
    g[mask] = intensity[mask] * (1 + (saturation[mask] * np.cos(2 * np.pi * (hue[mask] - 1/3)) / np.clip(np.cos(np.pi / 3 - 2 * np.pi * (hue[mask] - 1/3)), 1e-10, None)))
    b[mask] = 3 * intensity[mask] - (r[mask] + g[mask])

    mask = (2/3 <= hue) & (hue < 1)
    g[mask] = intensity[mask] * (1 - saturation[mask])
    b[mask] = intensity[mask] * (1 + (saturation[mask] * np.cos(2 * np.pi * (hue[mask] - 2/3)) / np.clip(np.cos(np.pi / 3 - 2 * np.pi * (hue[mask] - 2/3)), 1e-10, None)))
    r[mask] = 3 * intensity[mask] - (g[mask] + b[mask])

    return np.clip([r, g, b] * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

# Load the RGB image
rgb_image = cv2.imread('col.jpg')

# Convert RGB to HSI
hsi_image = rgb_to_hsi(rgb_image)


# Convert HSI back to RGB
reconstructed_rgb_image = hsi_to_rgb(hsi_image)

plt.figure(figsize=(12, 10))
# Display the original RGB image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')

# Display the reconstructed RGB image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(reconstructed_rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Reconstructed RGB Image from HSI')
plt.axis('off')
plt.show()


