import cv2
from PIL import Image

def video_loader(file_path):
    video = cv2.VideoCapture(file_path)
    frames = []
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to gray scale for ultrasound
        pil_img = Image.fromarray(frame , 'L') # convert to PIL Image
        frames.append(pil_img)
    video.release()
    return frames
