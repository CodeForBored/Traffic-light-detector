# debug_video_io.py
import cv2, os

# <-- CHANGE THIS to your exact full path (copy from Explorer address bar)
path = r"D:\Trafic_light_demo.mp4"

print("Input path:", path)
print("Exists?:", os.path.exists(path))

cap = cv2.VideoCapture(path)
print("cap.isOpened():", cap.isOpened())

if not cap.isOpened():
    print("ERROR: OpenCV cannot open the input file. Try opening the file in VLC or re-check the path.")
else:
    ret, frame = cap.read()
    print("first frame read:", ret)
    if ret:
        print("frame shape:", frame.shape)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    print("video w,h,fps:", w, h, fps)

    # Try writing an AVI (XVID) — often more compatible on Windows
    out_path = "debug_out.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    print("writer opened:", out.isOpened())

    # write up to 120 frames (few seconds)
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while written < 120:
        ret, frame = cap.read()
        if not ret:
            break
        # draw a small debug mark
        cv2.putText(frame, f"DBG {written}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        out.write(frame)
        written += 1

    cap.release()
    out.release()
    print("Wrote frames:", written)
    if os.path.exists(out_path):
        print("Output file size (bytes):", os.path.getsize(out_path))
    else:
        print("Output file was not created.")
