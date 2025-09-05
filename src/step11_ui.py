import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    roi = frame[0:int(h/2), int(w/4):int(3*w/4)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # HSV ranges (tuned later)
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])

    # Masks
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Count non-zero pixels
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # Decide state
    state = "NONE"
    if red_pixels > 500:
        state = "STOP"
    elif yellow_pixels > 500:
        state = "READY"
    elif green_pixels > 500:
        state = "GO"

    # Draw ROI box
    cv2.rectangle(frame, (int(w/4), 0), (int(3*w/4), int(h/2)), (255, 0, 0), 2)

    # --- Traffic Light UI (top-right corner) ---
    x, y = w - 100, 50
    radius = 20
    thickness = -1  # filled circles

    # Red circle
    cv2.circle(frame, (x, y), radius, (0, 0, 255) if state == "STOP" else (50, 50, 50), thickness)
    # Yellow circle
    cv2.circle(frame, (x, y + 60), radius, (0, 255, 255) if state == "READY" else (50, 50, 50), thickness)
    # Green circle
    cv2.circle(frame, (x, y + 120), radius, (0, 255, 0) if state == "GO" else (50, 50, 50), thickness)

    # Label text
    cv2.putText(frame, f"State: {state}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
