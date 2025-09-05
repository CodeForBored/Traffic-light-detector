import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame size
    h, w, _ = frame.shape

    # ROI: top-center (blue rectangle area)
    roi = frame[0:int(h/2), int(w/4):int(3*w/4)]

    # Convert ROI to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Color ranges (use your tuned HSV later)
    red_lower = np.array([0, 100, 100])
    red_upper = np.array([10, 255, 255])

    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    green_lower = np.array([40, 100, 100])
    green_upper = np.array([90, 255, 255])

    # Create masks
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Show ROI
    cv2.imshow("ROI", roi)

    # Draw ROI box on main frame
    cv2.rectangle(frame, (int(w/4), 0), (int(3*w/4), int(h/2)), (255, 0, 0), 2)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
