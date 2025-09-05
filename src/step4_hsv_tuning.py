import cv2
import numpy as np

# Create a window
cv2.namedWindow("Trackbars")

# Function to do nothing (needed for trackbars)
def nothing(x):
    pass

# Create trackbars for lower and upper HSV values
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)  # Hue lower
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)  # Saturation lower
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)  # Value lower
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)  # Hue upper
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)  # Saturation upper
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)  # Value upper

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    # Apply mask
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
