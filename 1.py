import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_img(img,title,cmap=None):
    plt.imshow(img,cmap)
    plt.title(f"{title}, {img.shape}")
    plt.axis('off')

image = cv2.imread("/content/iron-man-new-suit-for-avengers-infinity-war-artwork-hk.jpg")
assert image is not None, "Failed to load the image"
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.subplot(2,2,1)
plot_img(image,"original image")


gray_img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
plt.subplot(2,2,2)
plot_img(gray_img,"grayscale image",'gray')


resized_img = cv2.resize(image,(640,480))
plt.subplot(2,2,3)
plot_img(resized_img,"Resized Image")

cropped_img = image[0:50,200:250]
plt.subplot(2,2,4)
plot_img(cropped_img,"Cropped Image")

plt.tight_layout()
