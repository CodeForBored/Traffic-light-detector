import cv2
import numpy as np
import gradio as gr
import tempfile
import os
import shutil
import traceback

#  CONFIG   
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


def _normalize_input_to_path(input_video):
    """
    Return a filepath that cv2.VideoCapture can open, or None.
    Handles:
      - string path
      - dicts returned by some Gradio versions
      - file-like objects (save to temp)
    """
    if input_video is None:
        return None

    #  string path 
    if isinstance(input_video, str) and os.path.exists(input_video):
        return input_video

    #  dict 
    if isinstance(input_video, dict):
        for k in ("name", "tmp_path", "filepath", "file"):
            v = input_video.get(k)
            if isinstance(v, str) and os.path.exists(v):
                return v
        
        fobj = input_video.get("file")
        if hasattr(fobj, "name") and os.path.exists(fobj.name):
            return fobj.name

    
    if hasattr(input_video, "name") and isinstance(input_video.name, str) and os.path.exists(input_video.name):
        return input_video.name

    if hasattr(input_video, "read"):
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            with open(tmp_path, "wb") as out_f:
                
                while True:
                    chunk = input_video.read(8192)
                    if not chunk:
                        break
                    out_f.write(chunk)
            return tmp_path
        except Exception:
            return None

    return None

def _annotate_frame(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    counts = {"red": 0, "yellow": 0, "green": 0}

    for cname, ranges in COLOR_RANGES.items():
        mask_total = None
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)
            mask_total = mask if mask_total is None else cv2.bitwise_or(mask_total, mask)

        if mask_total is None:
            continue

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kern)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kern)

        counts[cname] = int(cv2.countNonZero(mask_total))

        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            x,y,wc,hc = cv2.boundingRect(cnt)
            color = (0,255,0)
            if cname=="red": color=(0,0,255)
            elif cname=="yellow": color=(0,255,255)
            cv2.rectangle(frame, (x,y), (x+wc, y+hc), color, 2)
            cv2.putText(frame, cname.upper(), (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    detected = max(counts, key=counts.get)
    if counts[detected] < PIXEL_THRESHOLD:
        detected = None

    return detected, frame

def process_video(input_video):
    """
    Gradio callable: accepts uploaded video (various types),
    writes annotated .avi and returns path.
    """
    tmp_input_path = None
    try:
        video_path = _normalize_input_to_path(input_video)
        print("DEBUG: resolved input path ->", video_path)
        if video_path is None:
            raise RuntimeError("No valid uploaded file found. Please upload a valid video file (mp4/avi).")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open uploaded video at path: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"DEBUG: input w,h,fps,frames = {w},{h},{fps},{total_frames}")

        
        fd, out_path = tempfile.mkstemp(suffix=".avi")
        os.close(fd)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to open video writer.")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detected, annotated = _annotate_frame(frame)
            
            if detected == "red":
                label, color = "STOP", (0,0,255)
            elif detected == "yellow":
                label, color = "SLOW", (0,255,255)
            elif detected == "green":
                label, color = "GO", (0,255,0)
            else:
                label, color = "NONE", (255,255,255)

            cv2.putText(annotated, f"STATE: {label}", (20,40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)

            writer.write(annotated)
            frame_idx += 1

        cap.release()
        writer.release()

       
        if video_path != input_video and hasattr(input_video, "read"):
            tmp_input_path = video_path

        print("DEBUG: wrote annotated output ->", out_path)
        return out_path

    except Exception as e:
        traceback.print_exc()
        raise RuntimeError("Processing error: " + str(e))

    finally:
        
        try:
            if tmp_input_path and os.path.exists(tmp_input_path):
                os.remove(tmp_input_path)
        except Exception:
            pass

#  Gradio UI 
title = "Traffic Light Detector — Upload video"
desc = "Upload a short traffic-light video (mp4). The app will return an annotated video showing detected state (STOP / SLOW / GO)."

demo = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload video"),
    outputs=gr.Video(label="Annotated output"),
    title=title,
    description=desc,
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch(share=True)

