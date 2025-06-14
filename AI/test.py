import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main.module.CNN import CNNFER,train,test

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNFER().to(device)
model.load_state_dict(torch.load("./main/model/cnn_fer.pth", map_location=device,weights_only=True))
model.eval()

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được webcam.")
    exit()
print("✅ Webcam đang chạy... Nhấn 'q' để thoát.")

# Bộ phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        try:
            roi_tensor = transform(roi_color).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(roi_tensor)
                _, predicted = torch.max(outputs, 1)
                emotion = emotion_labels[predicted.item()]

            # Vẽ box và label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("⚠️ Lỗi xử lý khuôn mặt:", e)

    # Hiển thị frame
    cv2.imshow("Emotion Recognition", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng
cap.release()
cv2.destroyAllWindows()