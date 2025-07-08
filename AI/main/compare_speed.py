import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import time
import os
import numpy as np
from torchvision.models import ResNet18_Weights
from torchvision.models import MobileNet_V2_Weights
import matplotlib.pyplot as plt

torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def benchmark_total_time(model, name, dataset):
    count = 0
    start = time.time()
    with torch.no_grad():
        for images, _ in dataset:
            tensor = images.unsqueeze(0).to(device)
            _ = model(tensor)
            count += 1
    end = time.time()
    total_time = end - start
    avg_time = total_time / count
    return total_time, avg_time

if __name__ == "__main__":
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ImageFolder(root="FER/test", transform=transform)
    print(len(dataset))

    resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device).eval()
    mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device).eval()
    names = []
    total_times = []
    avg_times = []

    for model, name in [(resnet18, "ResNet-18"), (mobilenet_v2, "MobileNetV2")]:
        t_total, t_avg = benchmark_total_time(model, name, dataset)
        names.append(name)
        total_times.append(t_total)
        avg_times.append(t_avg)
    plt.ylim(0,1)
    plt.figure(figsize=(8, 5))
    plt.bar(names, avg_times, color=['steelblue', 'orange'])
    plt.ylabel("Avg Time per Image (s)")
    plt.title("Benchmark Inference Time (CPU, Single Thread, No MKLDNN)")
    for i, v in enumerate(avg_times):
        plt.text(i, v + 0.0005, f"{v:.4f}s", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("benchmark_result.png")
