import cv2
import numpy as np
import gradio as gr

# function to detect traffic light in uploaded video
def detect_traffic_lights(video_path):
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    # HSV ranges for traffic light colors
    color_ranges = {
        "Red": [(0, 120, 70), (10, 255, 255)],
        "Yellow": [(15, 150, 150), (35, 255, 255)],
        "Green": [(40, 50, 50), (90, 255, 255)]
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        detected_color = None
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            if cv2.countNonZero(mask) > 500:  # threshold
                detected_color = color
                break

        if detected_color:
            cv2.putText(frame, f"{detected_color} Light", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # convert BGR → RGB (for Gradio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frames.append(frame_rgb)

    cap.release()
    return output_frames

# Gradio interface
demo = gr.Interface(
    fn=detect_traffic_lights,
    inputs=gr.Video(),
    outputs=gr.Video(),
    title="Traffic Light Detection",
    description="Upload a traffic light video and see detected signals!"
)

if __name__ == "__main__":
    demo.launch()
