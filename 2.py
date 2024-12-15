import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("/content/iron-man-new-suit-for-avengers-infinity-war-artwork-hk.jpg")
assert image is not None, "Failed to load the image"
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def plot_img(img,title,cmap=None,r=0,c=0,p=0):
  plt.subplot(r,c,p)
  plt.imshow(img)
  plt.title(title)
  plt.axis("off")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_ , binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)

sure_bg = cv2.dilate(opening,kernel,iterations=3)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

_, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

_,markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown==255]=0

markers = cv2.watershed(image, markers)
image[markers == -1] = [0,0,255]

markers_display = cv2.convertScaleAbs(markers)

plot_img(img,"original image",None,2,2,1)
plot_img(binary,"binary image",None,2,2,2)
plot_img(markers_display,"markers image",None,2,2,3)
# plot_img(simage,"segmented image",None,2,2,4)
plt.tight_layout()
