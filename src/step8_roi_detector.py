import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: camera not found"); exit()

RANGES = {
    "red1": (np.array([0, 120, 70]),   np.array([10, 255, 255])),
    "red2": (np.array([170, 120, 70]), np.array([180, 255, 255])),
    "yellow": (np.array([18, 100, 100]), np.array([35, 255, 255])),
    "green": (np.array([40, 70, 70]),   np.array([85, 255, 255]))
}

DRAW_COLOR = {
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "GREEN": (0, 255, 0)
}

AREA_THRESH = 500
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# define ROI as top-middle rectangle
def get_roi(frame):
    h, w, _ = frame.shape
    x1, y1 = w//3, 0         # left, top
    x2, y2 = 2*w//3, h//2    # right, halfway down
    return (x1, y1, x2, y2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    x1, y1, x2, y2 = get_roi(frame)
    roi = frame[y1:y2, x1:x2]  # crop to ROI
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask1 = cv2.inRange(hsv, RANGES["red1"][0], RANGES["red1"][1])
    red_mask2 = cv2.inRange(hsv, RANGES["red2"][0], RANGES["red2"][1])
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, RANGES["yellow"][0], RANGES["yellow"][1])
    green_mask = cv2.inRange(hsv, RANGES["green"][0], RANGES["green"][1])

    masks = {"RED": red_mask, "YELLOW": yellow_mask, "GREEN": green_mask}

    for label, mask in masks.items():
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > AREA_THRESH:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x, y), (x + w, y + h), DRAW_COLOR[label], 2)
                cv2.putText(roi, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, DRAW_COLOR[label], 2)

    # draw ROI box on main frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    frame[y1:y2, x1:x2] = roi  # overlay roi back

    cv2.imshow("ROI Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
