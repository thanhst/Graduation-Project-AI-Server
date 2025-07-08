from ai.rtp_process import rtpimageproc
import torch
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from module.CNN import CNNFER
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
import numpy as np
import time
from torchvision.datasets.folder import default_loader
import pandas as pd
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Sad','Surprise' ,'Neutral',]

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def test_pytorch_single(model, dataset, device):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for img, label in dataset:
            input_tensor = img.unsqueeze(0).to(device)
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
            if pred == label:
                correct += 1
    end_time = time.time()

    acc = correct / len(dataset)
    all_preds_str = [emotion_labels[p] for p in all_preds]
    all_labels_str = [emotion_labels[l] for l in all_labels]

    print(f"PyTorch Accuracy: {acc:.4f}")
    print(f"PyTorch Time: {end_time - start_time:.2f}s")
    print(classification_report(all_labels_str, all_preds_str, target_names=emotion_labels))

    return acc, end_time - start_time

def test_onnx_single(dataset):
    import time
    import numpy as np
    from torchvision.transforms.functional import to_pil_image
    from sklearn.metrics import classification_report

    correct = 0
    all_preds = []
    all_labels = []

    start_time = time.time()
    for img_tensor, label in dataset:
        img_rgb = img_tensor.convert("RGB")
        img_np = np.array(img_rgb)
        pred_str = rtpimageproc.infer_emotion_from_image(img_np)

        if pred_str not in emotion_labels:
            print(f"Unknown predicted label: {pred_str}")
            continue

        pred = emotion_labels.index(pred_str)

        all_preds.append(pred)
        all_labels.append(label)

        if pred == label:
            correct += 1

    end_time = time.time()
    acc = correct / len(all_labels)

    all_preds_str = [emotion_labels[p] for p in all_preds]
    all_labels_str = [emotion_labels[l] for l in all_labels]

    print(f"ONNX Accuracy: {acc:.4f}")
    print(f"ONNX Time: {end_time - start_time:.2f}s")
    print(classification_report(all_labels_str, all_preds_str, target_names=emotion_labels))

    return acc, end_time - start_time


def plot_accuracy_comparison(acc1, acc2):
    print("\nBảng độ chính xác:")
    df_acc = pd.DataFrame({
        "Model": ["PyTorch", "ONNX"],
        "Accuracy": [acc1, acc2]
    })
    print(df_acc.to_string(index=False))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(["PyTorch", "ONNX"], [acc1, acc2], color=['#4C72B0', '#55A868'])
    plt.gca().bar_label(bars, fmt='%.3f', padding=3)
    plt.title("Độ chính xác của mô hình")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_time_comparison(t1, t2, total_images):
    avg_time1 = t1 / total_images
    avg_time2 = t2 / total_images

    print("\nBảng thời gian trung bình trên mỗi ảnh:")
    df_time = pd.DataFrame({
        "Model": ["PyTorch", "ONNX"],
        "Avg Time per Image (s)": [avg_time1, avg_time2]
    })
    print(df_time.to_string(index=False))

    # Biểu đồ
    plt.figure(figsize=(6, 4))
    bars = plt.bar(["PyTorch", "ONNX"], [avg_time1, avg_time2], color=['#4C72B0', '#55A868'])
    plt.gca().bar_label(bars, fmt='%.3f', padding=3)
    plt.title("Thời gian trung bình trên mỗi ảnh")
    plt.ylabel("Seconds")
    plt.ylim(0, 0.05)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cpu")
    model = CNNFER().to(device)
    model.load_state_dict(torch.load("./model/cnn_fer.pth", map_location=device, weights_only=True))

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = ImageFolder(root='FER/test', transform=test_transform)
    raw_dataset = ImageFolder(root='FER/test', loader=default_loader)

    print("\nTesting PyTorch model:")
    acc1, t1 = test_pytorch_single(model, test_dataset, device)

    print("\nTesting ONNX model (via rtpimageproc):")
    acc2, t2 = test_onnx_single(raw_dataset)

    print(f"\nSo sánh:")
    print(f"PyTorch  → Accuracy: {acc1:.4f}, Time: {t1:.2f}s")
    print(f"ONNX     → Accuracy: {acc2:.4f}, Time: {t2:.2f}s")

    total_images = len(test_dataset)

    plot_accuracy_comparison(acc1, acc2)
    plot_time_comparison(t1, t2, total_images)


if __name__ == "__main__":
    main()
