import cv2
import numpy as np
import matplotlib.pyplot as plt

#Perform 2D convolution between an image and a kernel
def convolve2D(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]
    
    #for padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Allocate memory for the output image
    output = np.zeros_like(image, dtype="float32")
    
    # Pad the borders of the input image
    image_padded = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REPLICATE)
    
    for y in range(pad_height, image_height + pad_height):
        for x in range(pad_width, image_width + pad_width):
            #region of interest (ROI) from the padded image
            roi = image_padded[y - pad_height:y + pad_height + 1, x - pad_width:x + pad_width + 1]
            # Perform the actual convolution by element-wise multiplication between ROI and kernel, then summing the matrix
            k = np.sum(roi * kernel)
            # Store the convolved value in the output image
            output[y - pad_height, x - pad_width] = k
    
    # Rescale the output image to be in the range [0, 255]
    output = np.clip(output, 0, 255)
    output = output.astype("uint8")
    
    return output

# Load grayscale image
image = cv2.imread('doggy.jpeg', cv2.IMREAD_GRAYSCALE)

# Define kernels
box_kernel = np.ones((3,3), np.float32) / 9  # Box filter kernel

#gaussian kernel row x columnT(kernel_size,standard_dev)
gaussian_kernel = cv2.getGaussianKernel(3,0.071)@ cv2.getGaussianKernel(3,0.071).T

# Laplacian kernel
laplacian_kernel = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])  

# Prewitt kernel for x-direction
prewitt_kernel_x = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]]) 

# Prewitt kernel for y-direction 
prewitt_kernel_y = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]])  

# Sobel kernel for x-direction
sobel_kernel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

# Sobel kernel for y-direction  
sobel_kernel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])  

# Perform 2-D convolution with different kernels
box_filtered = convolve2D(image,box_kernel)
gaussian_filtered = convolve2D(image,gaussian_kernel)
laplacian_filtered = convolve2D(image, laplacian_kernel)
prewitt_filtered_x = convolve2D(image, prewitt_kernel_x)
prewitt_filtered_y = convolve2D(image, prewitt_kernel_y)
sobel_filtered_x = convolve2D(image, sobel_kernel_x)
sobel_filtered_y = convolve2D(image,sobel_kernel_y)

# Display the results
plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title('Box Filtered')
plt.imshow(box_filtered, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title('Gaussian Filtered')
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title('Laplacian Filtered')
plt.imshow(laplacian_filtered, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.title('Prewitt Filtered (X-direction)')
plt.imshow(prewitt_filtered_x, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.title('Prewitt Filtered (Y-direction)')
plt.imshow(prewitt_filtered_y, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.title('Sobel Filtered (X-direction)')
plt.imshow(sobel_filtered_x, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.title('Sobel Filtered (Y-direction)')
plt.imshow(sobel_filtered_y, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
