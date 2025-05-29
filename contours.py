'''
from https://docs.opencv.org/4.11.0/dd/d49/tutorial_py_contour_features.html
The thing they skip over is that finding contours is done with grayscale/b&w
images but all the drawing of contours is done on color images. 
The first 3 stanzas below convert the color image into a thresholded grayscale
which contains only black or white pixels and then find its contours. The 4th 
stanza converts the b&w image back into a color image that gets the contours
drawn upon in the 5th stanza.
After the divider, cnt is set to contours[0] and then cnt is used in most of
circle or rectangles or ellipse.
NOTE: It is very easy to blow up with divide by zero error if the image has
extraneous dots outside main body of contoured image
'''
import numpy as np
import cv2 as cv

def show_img(img, win_title):
    #print(img.shape) #verify if color or grayscale image
    cv.imshow(win_title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

original_image = cv.imread('resources/img/colorsquare.jpg')
assert original_image is not None, "file could not be read, check with os.path.exists()"
show_img(original_image, 'Original Color Image')

gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
show_img(gray_image, 'Gray-Scale Image')

ret,bw_thresh_image = cv.threshold(gray_image,127,255,0)
contours,hierarchy = cv.findContours(bw_thresh_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
show_img(bw_thresh_image, 'Black & White Thresholded Image')

# Draw the contours on a COLOR image (can't draw contour colors on B&W image)
color_thresh_image = cv.cvtColor(bw_thresh_image, cv.COLOR_GRAY2BGR)
#show_img(contour_color_image, 'Color version of B&W Image')

saved_color_image = color_thresh_image.copy()
cv.drawContours(color_thresh_image, contours, -1, (0, 255, 0), 3)
contour_image = color_thresh_image.copy()
show_img(contour_image, 'Contours drawn on Color version of B&W Image')

# ===========
cnt = contours[0]
M = cv.moments(cnt)
#print( M )

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv.circle(contour_image, (cx, cy), 5, (0, 0, 255), -1) #modifies image
show_img(contour_image, 'Contours Color Image with center point')
#print(f"centroid: {cx}, {cy}")

area = cv.contourArea(cnt)
print(f"area: {area}")

perimeter = cv.arcLength(cnt,True)
print(f"Perimeter: {perimeter}")

epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
print("epsilon set to 10% of arc = 4 xy points")
#print(approx)

epsilon = 0.01*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
print("epsilon set to 01% of arc = ~22 xy points")
#print(approx)

misc_img = np.zeros((500, 500, 3), dtype=np.uint8)
# Draw the contours
cv.drawContours(misc_img, contours, -1, (0, 255, 0), 3)
show_img(misc_img, 'Contours Drawn on Misc Image')

bounded_image = saved_color_image.copy()
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(bounded_image, (x,y), (x+w, y+h), (0,255,0),2)
show_img(bounded_image, 'Image with Bounding Rectange')

rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
#print(box)
box = np.int64(box)
# print(box)
cv.drawContours(bounded_image, [box], 0, (0,0,255),2)
show_img(bounded_image, 'Image with Bounded and Rotated Rectangle')

(x,y), radius = cv.minEnclosingCircle(cnt)
print((x,y), radius)
center = (int(x), int(y))
radius = int(radius)
cv.circle(bounded_image, center, radius, (255,0,0),2)
show_img(bounded_image, "Image Bounded by Circle and Rectangles")

ellipse = cv.fitEllipse(cnt)
cv.ellipse(bounded_image, ellipse, (0,255,0),2)
show_img(bounded_image, "Image Bounded by Ellipse, Circle, and Rectangles")
