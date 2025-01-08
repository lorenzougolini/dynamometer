from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import random
import numpy as np

def find_most_similar(on_pattern, digit_patterns):
    min_distance = float('inf')
    best_match = None

    for pattern, digit in digit_patterns.items():
        # Calculate Hamming distance
        distance = sum(a != b for a, b in zip(on_pattern, pattern))
        
        # Update best match if this pattern is more similar
        if distance < min_distance:
            min_distance = distance
            best_match = digit

    return best_match

def adjust_contours(contours):
    merged_boxes = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    bounding_boxes.sort(key=lambda b: b[0])
    
    skip_indices = set()
    for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
        if i in skip_indices:
            continue

        for j, (x2, y2, w2, h2) in enumerate(bounding_boxes):
            if i == j or j in skip_indices:
                continue

            vertical_gap = min(abs(y1 - (y2+h2)), abs(y2 - (y1+h1)))
            if (abs((x1+w1) - (x2+w2)) < max(w1, w2) * 0.4) and (vertical_gap >= 0 and vertical_gap <= max(h1, h2) * 0.2):
                x_min = min(x1, x2)
                y_min = min(y1, y2)
                x_max = max(x1 + w1, x2 + w2)
                y_max = max(y1 + h1, y2 + h2)
                
                merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

                skip_indices.add(i)
                skip_indices.add(j)
                break
        else:
            # This block executes only if no break was hit (i.e., no merge happened)
            merged_boxes.append((x1, y1, w1, h1))
    
    return merged_boxes


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

# img = cv2.imread('image.png')
vidcap = cv2.VideoCapture("dyn_video4.mp4")
totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
randomFrameNumber = random.randint(0, int(totalFrames))
vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
success, image = vidcap.read()
if success:
    img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

img = imutils.resize(img, height=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, spark_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
# blurred_mask = cv2.GaussianBlur(spark_mask, (15, 15), 0)
# darker_image = cv2.addWeighted(img, 1.0, np.zeros_like(img), 0, -50)
# image_with_reduced_spark = cv2.bitwise_and(darker_image, darker_image, mask=blurred_mask)
# image_without_spark = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(blurred_mask))
# cv2.imshow("No spark", image_without_spark)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

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

    # if (w >= 35 and w <= 45) and (h >= 80 and h <= 90):
    if w <= 45 and (h >= 35 and h <= 90):
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

digitsCnts = contours.sort_contours(digitsCnts, method='left-to-right')[0]
boxes = adjust_contours(digitsCnts)
for (x, y, w, h) in boxes:
    cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Merged Boxes", cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()

digits = []
for x,y,w,h in boxes:
    if w < 30:
        digits.append(1)
        continue
    roi = thresh[y:y+h, x:x+w]
    
    cv2.imshow("Digit ROI", roi)

    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW*0.25), int(roiH*0.15))
    dHC = int(roiH*0.05)

    segments = [
        ((0,0), (w,dH)), # top
        ((0,0), (dW,h//2)), # top left
        ((w-dW,0), (w,h//2)), # top right
        ((0,(h//2)-dHC), (w,(h//2)+dHC)), # center
        ((0,h//2), (dW,h)), # bottom left
        ((w-dW,h//2), (w,h)), # bottom right
        ((0,h-dH), (w,h)) # bottom 
    ]
    on = [0]*len(segments)
    
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        segROI = roi[yA:yB, xA:xB]
        tot = cv2.countNonZero(segROI)
        area = (xB-xA)*(yB-yA)

        if tot/float(area) > 0.5:
            on[i] = 1
    
    print(on)
    cv2.waitKey(0)
    cv2.destroyWindow("Digit ROI")

    digit = find_most_similar(tuple(on), DIGITS_PATTERNS)
    digits.append(digit)
    cv2.rectangle(cropped, (x,y), (x+w, y+h), (0, 255, 0), 1)
    cv2.putText(cropped, str(digit), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

print(f"{digits}")

cv2.waitKey(0)
cv2.destroyAllWindows()