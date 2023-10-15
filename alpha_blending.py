import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('/home/zeelb/multiresolutionblend/examples/earth.png')
image2 = cv2.imread('/home/zeelb/multiresolutionblend/examples/moon.png')

image1 = cv2.resize(image1, (512, 512))
image2 = cv2.resize(image2, (512, 512))

image1 = image1[10:498, 10:498]
image1 = cv2.resize(image1, (512, 512))


image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)




mask = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
# Set the left half of the mask to 1
mask[:, :mask.shape[1] // 2, :] = 1

b_mask = cv2.GaussianBlur(mask, (15, 15), 100)



direct_blend = (image1 * mask) + (image2 * (1 - mask))
alpha_blend = (image1 * b_mask) + (image2 * (1 - b_mask)) 

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Direct blending')
plt.imshow(direct_blend.astype('uint8'), cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Alpha blending')
plt.imshow(alpha_blend.astype('uint8'))
plt.axis('off')