import cv2

# original img
img = cv2.imread('acqui1_im_95_png.rf.4f9b8f3372d67dc99250844f375b76af.jpg')
cv2.imshow('Original', img)
cv2.waitKey(0)

# grayscale and blur for better edge
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

# Sobel Edge Detection X
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
# Y
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
# XY
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# canny edge detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()
