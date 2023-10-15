import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('/home/zeelb/multiresolutionblend/examples/earth.png')
image2 = cv2.imread('/home/zeelb/multiresolutionblend/examples/moon.png')

image1 = cv2.flip(image1, 1)

image1 = cv2.resize(image1, (512, 512))
image2 = cv2.resize(image2, (512, 512))

image1 = image1[10:498, 10:498]
image1 = cv2.resize(image1, (512, 512))


image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


num_levels = 6

width = 512
height = 512

# Create a 2D numpy array for the gradient
gradient = np.linspace(0, 1, width)
mask = np.tile(gradient, (height, 1))



# Create Gaussian pyramids for both images
gaussian_pyr1 = [image1]
gaussian_pyr2 = [image2]

plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')


plt.subplot(1, 3, 2)
plt.imshow(image2, cmap='gray')




for i in range(num_levels):
    image1 = cv2.pyrDown(image1)
    image2 = cv2.pyrDown(image2)
    # plt.subplot(1, 7, i+1)
    # plt.imshow(image1, cmap='gray')
    # plt.axis('off')
    gaussian_pyr1.append(image1)
    gaussian_pyr2.append(image2)

# print(len(gaussian_pyr1))
# print(gaussian_pyr1[0].shape)
# print(gaussian_pyr1[-1].shape)


# Create Laplacian pyramids for both images
laplacian_pyr1 = [gaussian_pyr1[num_levels]]
laplacian_pyr2 = [gaussian_pyr2[num_levels]]

for i in range(num_levels, 0, -1):
    image1 = cv2.pyrUp(gaussian_pyr1[i])
    laplacian1 = cv2.subtract(gaussian_pyr1[i - 1], image1)
    laplacian_pyr1.append(laplacian1)

    image2 = cv2.pyrUp(gaussian_pyr2[i])
    laplacian2 = cv2.subtract(gaussian_pyr2[i - 1], image2)
    laplacian_pyr2.append(laplacian2)
    # plt.subplot(2, 6, 7 - i)
    # plt.imshow(laplacian1, cmap='gray')
    # plt.axis('off')

# for i in range(len(laplacian_pyr1)):
#     plt.subplot(2, 7, i+1)
#     plt.imshow(laplacian_pyr1[i], cmap='gray')
#     plt.axis('off')
#     plt.subplot(2, 7, i+8)
#     plt.imshow(laplacian_pyr2[i], cmap='gray')
#     plt.axis('off')

# print(len(laplacian_pyr1))
# print(laplacian_pyr1[0].shape)
# print(laplacian_pyr1[-1].shape)

# Create a mask to blend the left halves of the images
mask = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
# Set the left half of the mask to 1
mask[:, :mask.shape[1] // 2, :] = 1
# plt.imshow(1 - mask, cmap='gray')

# Blend the Laplacian pyramids using the mask
blended_pyr = []
for lap1, lap2, i in zip(laplacian_pyr1, laplacian_pyr2, range(num_levels + 1)):
    mask = cv2.resize(mask, (lap1.shape[1], lap1.shape[0]))
    print(lap1.shape)
    
    blended_lap = (lap1  * mask) + (lap2 * (1 - mask))
    plt.subplot(1, 7, i+1)
    plt.imshow(blended_lap, cmap='gray')
    plt.axis('off')
    blended_pyr.append(blended_lap)

# for i in range(len(blended_pyr)):
#     plt.subplot(1, 7, i+1)
#     plt.imshow(blended_pyr[i], cmap='gray')
#     plt.axis('off')


# Reconstruct the blended image from the blended Laplacian pyramid
# blended_image = blended_pyr[0]
# for i in range(0, num_levels):
#     blended_image = cv2.pyrUp(blended_image)
#     print(blended_image.shape)
#     print(blended_pyr[i+1].shape)
#     blended_image += blended_pyr[i+1]

blended_image = blended_pyr[0]
for i in range(num_levels):

    blended_image = cv2.pyrUp(blended_image)
    blended_image += blended_pyr[i+1]

    # plt.subplot(1, 7, i+1)
    # plt.imshow(blended_image.astype('uint8'), cmap='gray')
    # plt.axis('off')

# plt.imshow(blended_image.astype('uint8'))
# plt.axis('off')
# The final blended image is in 'blended_image'
# cv2.imwrite('blended_image.jpg', blended_image)
# plt.subplot(1, 3, 1)
# plt.title('Direct blending')
# plt.imshow(direct_blend.astype('uint8'), cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.title('Alpha blending')
# plt.imshow(alpha_blend.astype('uint8'), cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Multiresolution blending')
# plt.imshow(blended_image.astype('uint8'), cmap='gray')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.title('Multiresolution blending')
# plt.imshow(blended_image.astype('uint8'), cmap='gray')
# plt.axis('off')