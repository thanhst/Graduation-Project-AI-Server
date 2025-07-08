from rtp_process import rtpimageproc
import ctypes
import cv2
img = cv2.imread("a2.jpg")
print("Shape:", img.shape, "Dtype:", img.dtype)
emotion = rtpimageproc.infer_emotion_from_image(img)
print("Dự đoán cảm xúc:", emotion)
