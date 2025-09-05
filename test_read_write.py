# debug_read_write.py
import cv2
import os

path = r"D:\Trafic_light_demo.mp4"   # <- keep your tested path here
print("Input path:", path)
cap = cv2.VideoCapture(path)

print("cap.isOpened():", cap.isOpened())
if not cap.isOpened():
    raise SystemExit("ERROR: Could not open input video. Check path or permissions.")

# try to read the first frame
ret, frame = cap.read()
print("first frame read:", ret)
if not ret or frame is None:
    raise SystemExit("ERROR: Couldn't read first frame from the input video.")

print("first frame type:", type(frame), "shape:", getattr(frame, "shape", None), "dtype:", getattr(frame, "dtype", None))

# determine width/height/fps robustly (use the first frame as fallback)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
fps_raw = cap.get(cv2.CAP_PROP_FPS) or 0.0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
print("raw video w,h,fps,frame_count:", w, h, fps_raw, frame_count)

# fallback to frame.shape if properties missing
if w == 0 or h == 0:
    if hasattr(frame, "shape"):
        h, w = frame.shape[:2]
        print("Fallback to frame.shape -> w,h:", w, h)
    else:
        raise SystemExit("ERROR: width/height unavailable and frame has no shape.")

# sanitize fps (must be >= 1 for writer)
fps = fps_raw if fps_raw and fps_raw >= 1.0 else 20.0
fps = float(fps)
print("using fps:", fps)

# write a couple of sample PNG frames for verification
out_dir = "debug_frames"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, "frame_000.png"), frame)
print(f"Wrote sample frame to {out_dir}/frame_000.png")

# Try MP4 first (mp4v). If writer fails, try AVI (XVID)
def try_writer(filename, fourcc_str):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    print(f"Trying writer -> {filename} using FOURCC='{fourcc_str}'; writer.isOpened():", writer.isOpened())
    return writer

out_mp4 = "test_out.mp4"
writer = try_writer(out_mp4, "mp4v")

if not writer.isOpened():
    # try alternate MP4 fourcc (may or may not help on Windows)
    writer = try_writer(out_mp4, "avc1")  # sometimes "avc1" / "H264" — may fail if not available

if not writer.isOpened():
    # fall back to .avi with XVID (commonly available)
    out_avi = "test_out.avi"
    writer = try_writer(out_avi, "XVID")
    out_filepath = out_avi
else:
    out_filepath = out_mp4

if not writer.isOpened():
    raise SystemExit("ERROR: Could not open any VideoWriter. You likely lack a matching codec on your system.")

# If we already consumed the first frame earlier, re-seek to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

count = 0
max_write = 120  # write up to 120 frames for test
while count < max_write:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    # ensure frame size matches writer expectation
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h))
    cv2.putText(frame, f"TEST {count}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    writer.write(frame)
    count += 1

cap.release()
writer.release()
print(f"WROTE: {out_filepath} — frames written: {count}")
print("If the file is playable, open with VLC/MPV. If it's not playable, see notes below.")
