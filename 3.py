import cv2
import numpy as np

image = cv2.imread("/content/iron-man-new-suit-for-avengers-infinity-war-artwork-hk.jpg")


h,w = image.shape[:2]
translation_matrix = np.float32([[1,0,100],[0,1,50]])
translated_image = cv2.warpAffine(image,translation_matrix,(w,h))

scaled_image = cv2.resize(image,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)

center = (w//2,h//2)
rotation_matrix = cv2.getRotationMatrix2D(center,45,1)
rotated_image = cv2.warpAffine(image,rotation_matrix,(w,h))

pts_1 = np.float32([[50,50],[200,50],[50,200]])
pts_2 = np.float32([[10,100],[200,50],[100,250]])
affine_matrix = cv2.getAffineTransform(pts_1,pts_2)
affine_image = cv2.warpAffine(image,affine_matrix,(w,h))

pts1 = np.float32([[0,0],[w-1,0],[0,h-1],[w-1,h-1]])
pts2 = np.float32([[0,0],[w-1,0],[int(0.33*w),h-1],[int(0.66*w),h-1]])
prespective_matrix = cv2.getPerspectiveTransform(pts1,pts2)
prespective_image = cv2.warpPerspective(image,prespective_matrix,(w,h))


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 10))

plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(232), plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB)), plt.title('Translated Image')
plt.subplot(233), plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)), plt.title('Scaled Image')
plt.subplot(234), plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image')
plt.subplot(235), plt.imshow(cv2.cvtColor(affine_image, cv2.COLOR_BGR2RGB)), plt.title('Affine Image')
plt.subplot(236), plt.imshow(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB)), plt.title('Perspective Image')

plt.show()
