# src/main.py
import cv2
import numpy as np

# HSV ranges 
COLOR_RANGES = {
    "red": [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([179, 255, 255]))
    ],
    "yellow": [(np.array([15, 80, 80]), np.array([40, 255, 255]))],
    "green": [(np.array([40, 50, 50]), np.array([95, 255, 255]))]
}

PIXEL_THRESHOLD = 300   
MIN_CONTOUR_AREA = 150  


def detect_state(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    counts = {"red": 0, "yellow": 0, "green": 0}

    for cname, ranges in COLOR_RANGES.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        if mask_total is None:
            continue

       
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kern)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kern)

        # count pixels for this color
        counts[cname] = cv2.countNonZero(mask_total)

        
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x, y, wc, hc = cv2.boundingRect(cnt)
            color = (0, 255, 0)
            if cname == "red":
                color = (0, 0, 255)
            elif cname == "yellow":
                color = (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x + wc, y + hc), color, 2)
            cv2.putText(frame, cname.upper(), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # decide state
    detected = max(counts, key=counts.get)
    if counts[detected] < PIXEL_THRESHOLD:
        detected = None

    return detected, frame


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # ROI
        roi_w, roi_h = 350, 350
        roi_x = (w - roi_w) // 2
        roi_y = int(h * 0.15)

        # Draw ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)

        # Process ROI
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        detected, roi_frame = detect_state(roi)
        frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = roi_frame

        # Map detection 
        label, color = "NONE", (255, 255, 255)
        if detected == "red":
            label, color = "STOP", (0, 0, 255)
        elif detected == "yellow":
            label, color = "SLOW", (0, 255, 255)
        elif detected == "green":
            label, color = "GO", (0, 255, 0)

        cv2.putText(frame, f"STATE: {label}", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

        cv2.imshow("Traffic Light Detector (ROI)", frame)

        
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
