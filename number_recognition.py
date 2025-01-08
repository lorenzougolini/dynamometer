from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2

DIGITS_PATTERNS = {
    (1, 1, 1, 0, 1, 1, 1): 0,
	(0, 0, 1, 0, 0, 1, 0): 1,
	(1, 0, 1, 1, 1, 1, 0): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

img = cv2.imread('numbers.png')

img = imutils.resize(img, height=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7,7), 3)

r = cv2.selectROI("Select the area", img)
cropped = blurred[int(r[1]):int(r[1]+r[3]),
              int(r[0]):int(r[0]+r[2])]

cv2.imshow("Cropped Image", cropped)

thresh = cv2.threshold(cropped, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cv2.imshow("Thresholded", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitsCnts = []

# for i, c in enumerate(cnts):
#     # Get the bounding box of the contour
#     (x, y, w, h) = cv2.boundingRect(c)

#     # Create a copy of the image to avoid modifying the original
#     contour_image = cropped.copy()

#     # Draw the current contour
#     cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 2)  # Green color, 2-pixel thickness
#     cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box in red

#     # Display the image with the contour
#     # print("Press 'a' to add this contour, any other key to skip, or 'q' to quit.")
#     cv2.imshow("Contour Viewer", contour_image)
#     print(f"Contour {i + 1}/{len(cnts)}")

#     # Wait for user input
#     key = cv2.waitKey(0)

#     if key == ord('q'):
#         break
#     elif key == ord('a'):
#         digitsCnts.append(c)
#         print(f"Contour added - Width: {w}, Height: {h}")

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)

    if (w >= 50 and w <= 65) and (h >= 120 and h <= 150):
        digitsCnts.append(c)

print(f"found digits: {len(digitsCnts)}")

contour_image = cropped.copy()
for c in digitsCnts:
    (x, y, w, h) = cv2.boundingRect(c)


    cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 2) 
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow("Contour Viewer", contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()