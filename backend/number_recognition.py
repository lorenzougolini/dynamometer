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
	(1, 0, 1, 1, 1, 0, 1): 2,
	(1, 0, 1, 1, 0, 1, 1): 3,
	(0, 1, 1, 1, 0, 1, 0): 4,
	(1, 1, 0, 1, 0, 1, 1): 5,
	(1, 1, 0, 1, 1, 1, 1): 6,
	(1, 0, 1, 0, 0, 1, 0): 7,
	(1, 1, 1, 1, 1, 1, 1): 8,
	(1, 1, 1, 1, 0, 1, 1): 9
}

def find_most_similar(on_pattern, prev_digit=None):
    '''
    Find the digit that is most similar to the given pattern.
    Args:
        on_pattern (tuple): The pattern of the lit segments.
        prev_digit (int): The previous digit to compare with.
    Returns:
        int: The most similar digit.
    '''
    result = {}
    for pattern, digit in DIGITS_PATTERNS.items():
        distance = sum(a != b for a, b in zip(on_pattern, pattern))
        result[digit] = distance
    min_dist = min(result.values())
    b_m = [digit for digit, dist in result.items() if dist == min_dist]
    if prev_digit:
        return sorted(list(zip(b_m, [abs(prev_digit-v) for v in b_m])), key=lambda x: x[1])[0][0]
    return b_m[0]

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

def reorder_box_points(box) -> np.ndarray:
    '''
    Reorder the box points in the order: top-left, top-right, bottom-right, bottom-left.
    Args:
        box (list): The list of four points representing the box.
    Returns:
        numpy.ndarray: The reordered box points.
    '''
    box = sorted(box, key=lambda p: (p[1], p[0]))  # Sort by y, then by x
    top_two = sorted(box[:2], key=lambda p: p[0])  # Sort top points by x
    bottom_two = sorted(box[2:], key=lambda p: p[0])  # Sort bottom points by x
    return np.array([top_two[0], top_two[1], bottom_two[1], bottom_two[0]], dtype=np.float32)

def find_LCD_roi(img) -> tuple:
    ''''
    Find the LCD screen in the image and return the box points of the screen.
    Args:
        img (numpy.ndarray): The input image in RGB.
    Returns:
        list: The list of four points representing the screen box.
        bool: A boolean indicating whether the screen was found
    '''
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # for blue dynanometer
    lower_blue = np.array([0, 200, 200])
    upper_blue = np.array([100, 255, 255])
    
    # for orange dynanometer
    # lower_blue = np.array([90, 50, 50])
    # upper_blue = np.array([120, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    isolated_blue = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Isolated Blue parts", isolated_blue)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Isolated Blue parts")

    # gray = cv2.cvtColor(isolated_blue, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (7,7), 3)
    # edged = cv2.Canny(blurred, 50, 150, 255)

    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(gray, cmap='gray')
    # axs[0].set_title('Grayscale Image')
    # axs[1].imshow(blurred, cmap='gray')
    # axs[1].set_title('Blurred Image')
    # axs[2].imshow(edged, cmap='gray')
    # axs[2].set_title('Edged Image')
    # for ax in axs:
    #     ax.axis('off')
    # plt.show()

    gray = cv2.cvtColor(isolated_blue, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour_img = cv2.drawContours(isolated_blue.copy(), cnts, -1, (0, 255, 0), 2)
    # cv2.imshow("Contour Image", contour_img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Contour Image")
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    if rect[1][0]*rect[1][1] < 12000:
        # print("Screen not found")
        return None, False
    box = np.array(cv2.boxPoints(rect), dtype=np.float32)

    # cv2.imshow("detected lcd box", cv2.drawContours(img.copy(), [box.astype(int)], -1, (0, 255, 0), 2))
    # cv2.waitKey(0)
    # cv2.destroyWindow("detected lcd box")

    return reorder_box_points(box), True
    # for c in cnts:
    #     rect = cv2.minAreaRect(c)
    #     (center_x, center_y), (width, height), angle = rect
    #     # print(f"Center: {(center_x, center_y)}, dims: {(width, height)}, angle: {angle}")
    #     box = cv2.boxPoints(rect)
    #     box = np.int32(box)

    #     rect_area = rect[1][0] * rect[1][1]

    #     # Find screen
    #     if rect_area > 14000:
    #         if (100 < center_x < 160 and 270 < center_y < 300) and (150 < width < 160 and 90 < height < 100):
    #             # print("Screen found")
    #             return reorder_box_points(box), True
    # print("Screen not found")
    # return None, False

def cut_LCD_roi(img, box) -> np.ndarray:
    '''
    Cut the LCD screen from the image using the box points.
    Args:
        img (numpy.ndarray): The input image in RGB.
        box (list): The list of four points representing the screen box in clockwise order.
    Returns:
        numpy.ndarray: The cropped LCD screen in RGB.
    '''
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    cropped = imutils.resize(cv2.warpPerspective(img, M, (width, height)), height=300)

    # cv2.imshow('Cropped Box', cropped)
    # cv2.waitKey(0)
    return cropped

def enlarge_roi_box(roi_box, scale=1.1):
    """
    Enlarges the ROI box (list of four points) by a given scale factor.
    Args:
        roi_box (list): A list of four points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
        scale (float): The scale factor by which to enlarge the box (default is 1.1).
    Returns:
        list: The enlarged ROI box as a list of four points [(x1, y1), ...].
    """
    center = np.mean(roi_box, axis=0)  # Compute the center of the box
    enlarged_box = [
        tuple((point - center) * scale + center) for point in roi_box
    ]
    return np.array([(int(x), int(y)) for x, y in enlarged_box], dtype=np.float32)


def extract_nums(roi, prev_record) -> tuple:
    '''
    Extract the numbers from the LCD screen.
    Args:
        roi (numpy.ndarray): The cropped LCD screen in RGB.
        prev_record (tuple): The previous record of the numbers.
    Returns:
        tuple: The extracted numbers as a tuple (int, int, int).
    '''
    new_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    # new_roi = cv2.GaussianBlur(new_roi, (7,7), 5)
    for i in range(3):
        new_roi = cv2.GaussianBlur(new_roi, (7,7), 5)
    thresh = cv2.threshold(new_roi, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Thresholded", thresh)
    
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

    # chosen_cnts = roi.copy()
    for c in cnts:
        (x,y,w,h) = cv2.boundingRect(c)
        # color = (0, 0, 255)
        if 10 <= w <= 90 and 65 <= h <= 185:
            digitsCnts.append(c)
            # print(f"Contour added - Width: {w}, Height: {h}")
            # color = (0, 255, 0)
        # else:
            # print(f"Contour NOT added - Width: {w}, Height: {h}")
        # cv2.rectangle(chosen_cnts, (x, y), (x + w, y + h), color, 2)
    # print("\n")
    # cv2.imshow("Chosen contours", chosen_cnts)

############### plot the picked contours ################## 
    # contour_image = roi.copy()
    # for c in digitsCnts:
    #     (x, y, w, h) = cv2.boundingRect(c)

    #     cv2.drawContours(contour_image, [c], -1, (0, 255, 0), 2) 
    #     cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("Contour Viewer", contour_image)
###########################################################

    if len(digitsCnts) == 0:
        return None, False
    
    digitsCnts = contours.sort_contours(digitsCnts, method='left-to-right')[0]
    boxes = adjust_contours(digitsCnts)
    for (x,y,w,h) in boxes:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Merged Boxes", roi)

    digits = []
    for i, (x,y,w,h) in enumerate(boxes):
        if w < 30:
            digits.append(1)
            continue
        if h < 85:
            digits.append(0)
            continue
        im_roi = new_roi[y:y+h, x:x+w]

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
    
        for (j, ((xA, yA), (xB, yB))) in enumerate(segments):
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

            # segment_display = im_roi.copy()
            # cv2.rectangle(segment_display, (xA, yA), (xB, yB), (255, 0, 0), 2)  # Blue box for the segment
            # cv2.rectangle(segment_display, (startX, startY), (endX, endY), (0, 0, 255), 2)  # Red box for the center
            # cv2.imshow(f"Segment", segment_display)
            # print(f"Segment {j}, Avg Intensity: {avg_intensity}")
            # key = cv2.waitKey(0)
        
            threshold = 100
            if avg_intensity < threshold:
                on[j] = 1
        # print(f"{on}")

        # print(on)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Digit ROI")

        if i < len(prev_record):
            prev_digit = prev_record[i]
        digit = find_most_similar(tuple(on), prev_digit)
        digits.append(digit)
    
    # cv2.rectangle(cropped, (x,y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.putText(cropped, str(digit), (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

    # print(f"{digits[0]}.{''.join(list(map(str,digits[1:])))}")
    # cv2.waitKey(0)
    # cv2.destroyWindow("Merged Boxes")
    if len(digits) < 3:
        return None, False
    return digits, True


def main(path=None, plot=True):
    
    cap = cv2.VideoCapture(path)

    # f = random.randint(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    # cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    # print(f"Frame {f}") 77, 92
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # succ, img = cap.read()
    
    # video loop
    start_time = time.time()
    records = {}
    # prev_roi_rect = cv2.selectROI("Select the LCD area", img)
    # prev_roi_box = np.array([[prev_roi_rect[0], prev_roi_rect[1]],
    #                 [prev_roi_rect[0] + prev_roi_rect[2], prev_roi_rect[1]],
    #                 [prev_roi_rect[0] + prev_roi_rect[2], prev_roi_rect[1] + prev_roi_rect[3]],
    #                 [prev_roi_rect[0], prev_roi_rect[1] + prev_roi_rect[3]]
    # ], dtype=np.float32)
    # cv2.destroyWindow("Select the LCD area")
    prev_record = [0, 0]
    while True:
        succ, img = cap.read()
        if not succ:
            # print("End of video or cannot read frame")
            break
        
        img = imutils.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), height=500)
        # cv2.imshow('Frame', img)

        roi_box, screen_found = find_LCD_roi(img)
        if screen_found:
            screen = cut_LCD_roi(img, roi_box)
            prev_roi_box = roi_box
        else:
            screen = cut_LCD_roi(img, prev_roi_box)
        
        n, number_found = extract_nums(screen, prev_record)
        if number_found:
            prev_record = n
        number = float(f"{prev_record[0]}.{''.join(list(map(str,prev_record[1:])))}")
        t = round(time.time()-start_time, 2)
        records[t] = number   
        # print(f"Frame: {cap.get(cv2.CAP_PROP_POS_FRAMES)}, Number: {number}")
        # print(f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}, Time: {t%.2}, Number: {number}")

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

    # with open("datafile.csv", "w", newline="") as f:
    #     w = csv.DictWriter(f, records.keys())
    #     w.writeheader()
    #     w.writerow(records)
    cap.release()
    cv2.destroyAllWindows()
    # if plot:
    #     plot_data(records)
    
    # exit()
    return records


if __name__ == "__main__":
    main("./dyn_video5.mp4", True)