from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

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
    
    skip_indices = set()
    for i, (x1, y1, w1, h1) in enumerate(bounding_boxes):
        if i in skip_indices:
            continue
        
        # is_contained = False
        # for mx, my, mw, mh in merged_boxes:
        #     if (x1 >= mx and y1 >= my) and (x1 + w1 <= mx + mw and y1 + h1 <= my + mh):
        #         is_contained = True
        # if is_contained: 
        #     break

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

def reorder_box_points(box):
    box = sorted(box, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
    top_two = sorted(box[:2], key=lambda p: p[0])  # Sort top points by x
    bottom_two = sorted(box[2:], key=lambda p: p[0])  # Sort bottom points by x
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]], dtype=np.float32)

def find_LCD_roi(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    eq = cv2.equalizeHist(img)
    blurred = cv2.GaussianBlur(eq, (7,7), 3)
    edged = cv2.Canny(blurred, 50, 150, 255)

    # fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    # axs[0].imshow(gray, cmap='gray')
    # axs[0].set_title('Grayscale Image')
    # axs[1].imshow(eq, cmap='gray')ccccc
    # axs[1].set_title('Equalized Image')
    # axs[2].imshow(blurred, cmap='gray')
    # axs[2].set_title('Blurred Image')
    # axs[3].imshow(edged, cmap='gray')
    # axs[3].set_title('Edged Image')
    # for ax in axs:
    #     ax.axis('off')
    # plt.show()

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    found = False
    for c in cnts:
        rect = cv2.minAreaRect(c)
        (center_x, center_y), (width, height), angle = rect
        # print(f"Center: {(center_x, center_y)}, dims: {(width, height)}, angle: {angle}")
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        rect_area = rect[1][0] * rect[1][1]

        # Find screen
        if rect_area > 14000:
            if (100 < center_x < 160 and 270 < center_y < 300) and (150 < width < 160 and 90 < height < 100):
                # print("Screen found.\n")
                return reorder_box_points(box), True
    return None, found

def cut_LCD_roi(img, box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[0] - box[3]))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    cropped = imutils.resize(cv2.warpPerspective(img, M, (width, height)), height=145)

    # cv2.imshow('Cropped Box', cropped)
    # cv2.waitKey(0)
    return cropped


def extract_nums(roi):
    thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("Thresholded", thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitsCnts = []


############################ TO ANALYZE EVERY DIGIT CONTOUR ############################
    # for i, c in enumerate(cnts):
    #     # Get the bounding box of the contour
    #     (x, y, w, h) = cv2.boundingRect(c)

    #     # Create a copy of the image to avoid modifying the original
    #     contour_image = roi.copy()

    #     # Draw the current contour
    #     cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 2)  # Green color, 2-pixel thickness
    #     cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding box in red

    #     # Display the image with the contour
    #     # print("Press 'a' to add this contour, any other key to skip, or 'q' to quit.")
    #     cv2.imshow("Contour Viewer", contour_image)
    #     print(f"Contour {i + 1}/{len(cnts)}")

    #     # Wait for user input
    #     key = cv2.waitKey(0)

    #     if key == ord('z'):
    #         break
    #     elif key == ord('a'):
    #         digitsCnts.append(c)
    #         print(f"Contour added - Width: {w}, Height: {h}")
#####################################################################

    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        if w <= 45 and (h >= 30 and h <= 90):
            digitsCnts.append(c)

# print(f"found digits: {len(digitsCnts)}")

# contour_image = cropped.copy()
# for c in digitsCnts:
#     (x, y, w, h) = cv2.boundingRect(c)

#     cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 2) 
#     cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# cv2.imshow("Contour Viewer", contour_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
    if len(digitsCnts) == 0:
        return None, False
    
    digitsCnts = contours.sort_contours(digitsCnts, method='left-to-right')[0]
    boxes = adjust_contours(digitsCnts)
    for (x, y, w, h) in boxes:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Merged Boxes", roi)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()

    digits = []
    # print(len(boxes))
    for x,y,w,h in boxes:
        if w < 30:
            digits.append(1)
            continue
        if h < 60:
            digits.append(0)
            continue
        im_roi = roi[y:y+h, x:x+w]
        # test_im = im_roi.copy()
    # cv2.imshow("Digit ROI", im_roi)

        (roiH, roiW) = im_roi.shape
        (dW, dH) = (int(roiW*0.3), int(roiH*0.2))
        dHC = int(roiH*0.05)

        segments = [
            ((0,0), (w,dH-dHC)), # top
            ((0,0), (dW,h//2)), # top left
            ((w-dW,0), (w,h//2)), # top right
            ((0,(h//2)-dHC), (w,(h//2)+dHC)), # center
            ((0,h//2), (dW,h)), # bottom left
            ((w-dW,h//2), (w,h)), # bottom right
            ((0,h-dH+dHC), (w,h)) # bottom 
        ]
        on = [0]*len(segments)
    
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            centerX = (xA + xB) // 2
            centerY = (yA + yB) // 2
            square_size = 10
            half_square = square_size // 2
            startX = max(xA, centerX - half_square)
            startY = max(yA, centerY - half_square)
            endX = min(xB, centerX + half_square)
            endY = min(yB, centerY + half_square)

            segROI = im_roi[yA:yB, xA:xB]
            segCenterROI = im_roi[startY:endY, startX:endX]
            avg_intensity_seg = cv2.mean(segROI)[0]
            avg_intensity_seg_center = cv2.mean(segCenterROI)[0]
            avg_intensity = np.floor(np.mean([avg_intensity_seg, avg_intensity_seg_center]))
            # print(f"Segment {i}, Avg Intensity: {avg_intensity}")
        
            threshold = 110
            if avg_intensity <= threshold:
                on[i] = 1

            # cv2.rectangle(im_roi, (xA, yA), (xB, yB), 255, 1)
            # cv2.imshow("Segment ROI", test_im)
            # cv2.waitKey(0)
            # cv2.destroyWindow("Segment ROI")
    
        # print(on)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Digit ROI")

        digit = find_most_similar(tuple(on), DIGITS_PATTERNS)
        digits.append(digit)
    # cv2.rectangle(cropped, (x,y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.putText(cropped, str(digit), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

    # print(f"{digits[0]}.{digits[1:]}")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return digits, True


# img = cv2.imread('image.png')
# vidcap = cv2.VideoCapture("dyn_video4.mp4")
# totalFrames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
# randomFrameNumber = random.randint(0, int(totalFrames))
# vidcap.set(cv2.CAP_PROP_POS_FRAMES,randomFrameNumber)
# success, image = vidcap.read()
# if success:
#     img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# img = imutils.resize(img, height=500)
# r = cv2.selectROI("Select the LCD area", img)
# extract_nums(img, r)


if __name__ == "__main__":

    cap = cv2.VideoCapture("dyn_video5.mp4")

    # select ROI
    # f = random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    # print(f"Frame {f}") 47
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    succ, img = cap.read()

    # cv2.imshow("Frame selected", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Frame selected")
    # if succ:
    #     # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    #     img = imutils.resize(img, height=700)
    #     r = cv2.selectROI("Select the LCD area", img)
    #     cv2.destroyWindow("Select the LCD area")
    
    # video loop
    start_time = time.time()
    records = {}
    prev_roi_box = np.array([[ 40, 200],
                             [250, 200],
                             [250, 350],
                             [ 40, 350]], dtype=np.float32)
    prev_record = [0, 0]
    while True:
        succ, frame = cap.read()
        if not succ:
            print("End of video or cannot read frame")
            break
        
        # img = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        img = imutils.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), height=500)
        cv2.imshow('Frame', img)

        roi_box, found = find_LCD_roi(img)
        if not found:
            screen = cut_LCD_roi(img, prev_roi_box)
        else:
            prev_roi_box = roi_box
            screen = cut_LCD_roi(img, roi_box)
        
        n, found = extract_nums(screen)
        if found:
            prev_record = n
        number = float(f"{prev_record[0]}.{''.join(list(map(str,prev_record[1:])))}")
        records[time.time()-start_time] = number   

        # while True:
        #     key = cv2.waitKey(0) & 0xFF
        #     if key == ord('c'):
        #         break  # Continue to the next frame
        #     elif key == ord('q'):
        #         cap.release()
        #         cv2.destroyAllWindows()
        #         exit()  # Exit the program

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    with open("datafile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, records.keys())
        w.writeheader()
        w.writerow(records)

    cap.release()
    cv2.destroyAllWindows()
    exit()