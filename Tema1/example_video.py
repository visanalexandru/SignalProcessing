import cv2
import jpeg
import numpy as np

vid_capture = cv2.VideoCapture("fireworks.mp4")
assert vid_capture.isOpened()

width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid_capture.get(cv2.CAP_PROP_FPS)
frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("fireworks.avi", fourcc, fps, (int(width), int(height)))

current_frame = 1

while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    if not ret:
        break

    print(f"Frame {current_frame} out of {frame_count}")

    image = np.array(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    compressed = jpeg.compress_rgb(image, quality=80)
    decompressed = jpeg.decompress_rgb(compressed)

    decompressed = cv2.cvtColor(decompressed, cv2.COLOR_RGB2BGR)

    out.write(decompressed)
    current_frame += 1


vid_capture.release()
out.release()
