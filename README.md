# Laboratory3ImageProcessing
## Original Image
![output1](https://user-images.githubusercontent.com/65180398/144722527-1ec22eed-6a60-48aa-8fe9-47f2de5045e3.png)
## Fourier Filter
![output2](https://user-images.githubusercontent.com/65180398/144722679-c9e608cf-00c6-4f74-9ad3-8c43d10b8cfa.png)
### Sharpening
kernel3 = np.array([[0, -1,  0],
                   [-1,  5, -1],
                    [0, -1,  0]])
mask = cv.filter2D(img, -1, kernel3)


### applying mask
fshift_masked = np.multiply(fshift, mask) / 255

### Reverse transformation
back_ishift_masked = np.fft.ifftshift(fshift_masked)
img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))

### combining the components to form an image again
img_filtered = np.abs(img_filtered).clip(0,255).astype(np.uint8)
## Image with the filter applied
![output3](https://user-images.githubusercontent.com/65180398/144722696-bbcf1220-ef63-47fb-bffa-64620523428f.png)
