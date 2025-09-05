import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: camera not found"); exit()

# Default HSV ranges 
RANGES = {
    "red1": (np.array([0, 120, 70]),   np.array([10, 255, 255])),
    "red2": (np.array([170, 120, 70]), np.array([180, 255, 255])),
    "yellow": (np.array([18, 100, 100]), np.array([35, 255, 255])),
    "green": (np.array([40, 70, 70]),   np.array([85, 255, 255]))
}

# BGR colors for drawing 
DRAW_COLOR = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0)
}

AREA_THRESH = 500           
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SHOW_MASKS = False

print("Keys: ESC exit | s = print center HSV | m = toggle masks | [ / ] = lower/raise area threshold")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (800, 600))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # build masks
    red_mask1 = cv2.inRange(hsv, RANGES["red1"][0], RANGES["red1"][1])
    red_mask2 = cv2.inRange(hsv, RANGES["red2"][0], RANGES["red2"][1])
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    yellow_mask = cv2.inRange(hsv, RANGES["yellow"][0], RANGES["yellow"][1])
    green_mask = cv2.inRange(hsv, RANGES["green"][0], RANGES["green"][1])

    # smooth masks
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, KERNEL)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, KERNEL)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, KERNEL)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, KERNEL)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, KERNEL)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, KERNEL)

    # Detect and draw per color
    for mask, label in [(red_mask, "RED"), (yellow_mask, "YELLOW"), (green_mask, "GREEN")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < AREA_THRESH:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            color = DRAW_COLOR[label]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    
    cv2.imshow("Detector", frame)
    if SHOW_MASKS:
        cv2.imshow("Mask - RED", red_mask)
        cv2.imshow("Mask - YELLOW", yellow_mask)
        cv2.imshow("Mask - GREEN", green_mask)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break
    elif key == ord('m'):
        SHOW_MASKS = not SHOW_MASKS
    elif key == ord('s'):  # for tuning
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        print("Center HSV:", hsv[cy, cx])
    elif key == ord(']'):
        AREA_THRESH += 100
        print("AREA_THRESH ->", AREA_THRESH)
    elif key == ord('['):
        AREA_THRESH = max(100, AREA_THRESH - 100)
        print("AREA_THRESH ->", AREA_THRESH)

cap.release()
cv2.destroyAllWindows()
