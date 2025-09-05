import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: camera not found")
    quit()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
