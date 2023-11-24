import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load Color Image
image = cv2.imread('capybara.jpeg', 0)

# Define Filters
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]])

shift = np.array([
    [0, 0, 0],
    [0, 0, 10],
    [0, 20, 0]])

gaussian = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]]) / 16

motion_blur = np.array([
    [1, 0, 0, 0, 0, 0, 1],  
    [0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1]]) / 9

sharpness = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])

vertical_sobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

border = np.array([
    [-1, -1, 0], 
    [-1, 8, -1], 
    [0, -1, -1]])

invert = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

def handler_convolve(img, filter):
    img_height, img_width = img.shape
    filter_height, filter_width = filter.shape
    
    out_height = img_height - filter_height + 1
    out_width = img_width - filter_width + 1

    output_image = np.zeros((out_height, out_width))

    for h in range(out_height):
        for w in range(out_width):
            st = img[h:h + filter_height, w:w + filter_width]
            output_image[h, w] = np.sum(st * filter)

    return output_image

identity_image = handler_convolve(image, identity)
shifted_image = handler_convolve(image, shift)
gaussian_image = handler_convolve(image, gaussian)
motion_blur_image = handler_convolve(image, motion_blur)
sharpness_image = handler_convolve(image, sharpness)
vertical_sobel_image = handler_convolve(image, vertical_sobel)
border_image = handler_convolve(image, border)
invert_image = handler_convolve(image, invert)

fig, axs = plt.subplots(3, 4)
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 1].imshow(identity_image, cmap='gray')
axs[0, 1].set_title('Identity Filter')
axs[0, 2].imshow(shifted_image, cmap='gray')
axs[0, 2].set_title('Shift Filter')
axs[0, 3].imshow(gaussian_image, cmap='gray')
axs[0, 3].set_title('Gaussian Filter')

axs[1, 0].imshow(motion_blur_image, cmap='gray')
axs[1, 0].set_title('Motion Blur Filter')
axs[1, 1].imshow(sharpness_image, cmap='gray')
axs[1, 1].set_title('Sharpness Filter')
axs[1, 2].imshow(vertical_sobel_image, cmap='gray')
axs[1, 2].set_title('Vertical Sobel Filter')
axs[1, 3].imshow(border_image, cmap='gray')
axs[1, 3].set_title('Border Filter')

axs[2, 0].imshow(invert_image, cmap='gray')
axs[2, 0].set_title('Invert Filter')

plt.tight_layout()
plt.show()