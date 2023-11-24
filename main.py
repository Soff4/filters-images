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

horizontal_sobel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])

border = np.array([
    [-1, -1, 0], 
    [-1, 8, -1], 
    [0, -1, -1]])

invert = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]])

kernel_filer = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]], dtype=np.uint8)

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

def handler_erosion(img, filter):
    img_rows, img_cols = img.shape
    filter_rows, filter_cols = filter.shape
    
    eroded_image = np.zeros((img_rows, img_cols), dtype=np.uint8)

    for r in range(1, img_rows - 1):
        for c in range(1, img_cols - 1):
            min_val = 255

            for filter_r in range(filter_rows):
                for filter_c in range(filter_cols):
                    val = img[r - 1 + filter_r][c - 1 + filter_c] - filter[filter_r][filter_c]

                    if val < min_val:
                        min_val = val

            eroded_image[r][c] = min_val

    return eroded_image

def handler_dilation(img, filter):
    img_rows, img_cols = img.shape
    filter_rows, filter_cols = filter.shape
    
    dilated_image = np.zeros((img_rows, img_cols), dtype=np.uint8)

    for r in range(1, img_rows - 1):
        for c in range(1, img_cols - 1):
            max_val = 0

            for filter_r in range(filter_rows):
                for filter_c in range(filter_cols):
                    val = img[r - 1 + filter_r][c - 1 + filter_c] + filter[filter_r][filter_c]

                    if val > max_val:
                        max_val = val

            dilated_image[r][c] = max_val

    return dilated_image

def handler_opening(img, filter):
    erosion_img = handler_erosion(img, filter)
    opening_image = handler_dilation(erosion_img, filter)
    return opening_image

def handler_closing(img, filter):
    dilation_img = handler_dilation(img, filter)
    closing_image = handler_erosion(dilation_img, filter)
    return closing_image

def handler_edges(img):
    img_rows, img_cols = img.shape

    gradient_x = np.zeros((img_rows - 2, img_cols - 2))
    gradient_y = np.zeros((img_rows - 2, img_cols - 2))
    gradient_mg = np.zeros((img_rows - 2, img_cols - 2))

    for r in range(img_rows - 2):
        for c in range(img_cols - 2):
            window = img[r:r+3, c:c+3]
            gradient_x[r, c] = np.sum(window * horizontal_sobel)
            gradient_y[r, c] = np.sum(window * vertical_sobel)
            gradient_mg[r, c] = np.sqrt(gradient_x[r, c]**2 + gradient_y[r, c]**2)

    return gradient_mg

def handler_show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

identity_image = handler_convolve(image, identity)
shifted_image = handler_convolve(image, shift)
gaussian_image = handler_convolve(image, gaussian)
motion_blur_image = handler_convolve(image, motion_blur)
sharpness_image = handler_convolve(image, sharpness)
vertical_sobel_image = handler_convolve(image, vertical_sobel)
border_image = handler_convolve(image, border)
invert_image = handler_convolve(image, invert)
eroded_image = handler_erosion(image, kernel_filer)
dilated_image = handler_dilation(image, kernel_filer)
opening_image = handler_opening(image, kernel_filer)
closing_image = handler_closing(image, kernel_filer)
edges_of_image = handler_edges(image)

handler_show_image(edges_of_image, 'Edges of Image')