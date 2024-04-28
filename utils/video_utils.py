import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path) # Captured at 24 fps
    frames = []
    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print("+------------------------------------+")
    print("Total frames in the video: ", len(frames))
    print("+------------------------------------+")

    return frames

def write_video(frames, output_path):
    print("+------------------------------------+")
    print("Saving video...")

    height, width, _ = frames[0].shape # 384 x 640
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    print("Video saved at: ", output_path)
 