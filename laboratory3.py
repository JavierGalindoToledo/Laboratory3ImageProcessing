import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("__119.png", 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
fft_img=img = 20*np.log(np.abs(fshift))

plt.imshow(img, cmap = 'gray')
plt.title("Original image")
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(fft_img, cmap = 'gray')
plt.title("Fourier filter")
plt.xticks([]), plt.yticks([])
plt.show()

kernel3 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
mask = cv.filter2D(img, -1, kernel3)

fshift_masked = np.multiply(fshift, mask) / 255

back_ishift_masked = np.fft.ifftshift(fshift_masked)
img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))

img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)

plt.imshow(img_filtered, cmap = 'gray')
plt.title("Result")
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(20, 20))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap = 'gray')
plt.title("Original image")
plt.subplot(1, 3, 2)
plt.imshow(fft_img, cmap = 'gray')
plt.title("Fourier filter")
plt.subplot(1, 3, 3)
plt.imshow(img_filtered, cmap = 'gray')
plt.title("The result after sharpening on \nthe Fourier filter")