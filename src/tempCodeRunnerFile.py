import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# HSV ranges
color_ranges = {
    "Red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([170, 100, 100]), np.array([180, 255, 255]))
    ],
    "Yellow": [(np.array([15, 100, 100]), np.array([35, 255, 255]))],
    "Green": [(np.array([40, 100, 100]), np.array([85, 255, 255]))]
}

# Colors for drawing
draw_colors = {
    "Red": (0, 0, 255),
    "Yellow": (0, 255, 255),
    "Green": (0, 255, 0)
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, ranges in color_ranges.items():
        mask_total = None
        for (lower, upper) in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 500 < area < 5000:  # filter very small + very large
                (x, y, w, h) = cv2.boundingRect(cnt)

                # shape filtering: check circularity
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.3:  # too irregular, skip
                    continue

                # draw bounding box + label
                cv2.rectangle(frame, (x, y), (x + w, y + h), draw_colors[color], 2)
                cv2.putText(frame, color, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[color], 2)

    cv2.imshow("Traffic Light Detection - Cleaned", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
