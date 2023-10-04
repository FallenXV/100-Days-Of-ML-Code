import cv2
import numpy as np
from matplotlib import pyplot as plt

# Important note: OpenCV stores the colour info in BGR format

# Read Image in full colour
image = cv2.imread('datasets/input.jpg')
cv2.imshow('image', image)

# Read the image in greyscale
image2 = cv2.imread('datasets/input.jpg', 0)
cv2.imshow('image2', image2)

# Convert the image to HSV format
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imshow('HSV Image', hsv_image)
# cv2.imshow('Hue Channel', hsv_image[:, :, 0])
# cv2.imshow('Saturation Channel', hsv_image[:, :, 1])
# cv2.imshow('Value Channel', hsv_image[:, :, 2])

# Convert the image to Grayscale (alternative method to obtain grayscale)
# grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('grey_image', grey_image)

# Split the image into 3 images, Blue, Green and Red
B, G, R = cv2.split(image)
zeros = np.zeros(image.shape[:2], dtype="uint8")
# cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
# cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
# cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

# Plotting histogram of the image
plt.hist(image2.ravel(), 256, [0, 256])
# plt.show()

# Thresholding
retval, threshold = cv2.threshold(image2, 120, 255, cv2.THRESH_BINARY)
# cv2.imshow('threshold', threshold)

# Histogram Equalization
histogram_equalization = cv2.equalizeHist(image2)
# cv2.imshow('Histogram Equalization', histogram_equalization)
# plt.hist(histogram_equalization.ravel(), 256, [0, 256])

# Image Resize
(height, width) = image.shape[:2]
resized_image = cv2.resize(image, (int(width / 2), int(height / 2)), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('Resized Image', resized_image)
cv2.imwrite('datasets/resized_image.jpg', resized_image)

# Image Addition
test = cv2.imread('datasets/input2.jpg')
resized_test = cv2.resize(test, (1500, 1000), interpolation=cv2.INTER_CUBIC)
weightedSum = cv2.addWeighted(image, 0.5, resized_test, 0.4, 0)
cv2.imshow('Weighted Image', weightedSum)

# Image Convolution and Blurring
kernel_3x3 = np.ones((3, 3), np.float32) / 9  # 9 is the sum of the kernel
blurred = cv2.filter2D(image, -1, kernel_3x3)  # Convolution: -1 is the depth of the output image
cv2.imshow('3x3 Kernel Blurring', blurred)
kernel_7x7 = np.ones((7, 7), np.float32) / 49  # 49 is the sum of the kernel
blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kernel Blurring', blurred2)

plt.show()
print(image.shape)  # shows the dimensions and colour info flag of the image
if cv2.waitKey(0) & 0xFF == 27:
    cv2.destroyAllWindows()
