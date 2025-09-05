import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: camera not found")
    quit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
  
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([40, 70, 70])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Combine with original frame to see detection
    red_detected = cv2.bitwise_and(frame, frame, mask=mask_red)
    yellow_detected = cv2.bitwise_and(frame, frame, mask=mask_yellow)
    green_detected = cv2.bitwise_and(frame, frame, mask=mask_green)

   
    cv2.imshow("Webcam", frame)
    cv2.imshow("Red Detection", red_detected)
    cv2.imshow("Yellow Detection", yellow_detected)
    cv2.imshow("Green Detection", green_detected)

    # Quit when pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
