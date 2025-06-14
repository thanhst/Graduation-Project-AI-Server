import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from module.CNN import CNNFER,train,test

# ---------------------- Main ----------------------
def main():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    if device.type == 'cuda':
        print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA is available: {torch.cuda.is_available()}")
    else:
        print("⚠️ WARNING: Running on CPU")
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = ImageFolder(root='FER/train', transform=train_transform)
    test_dataset  = ImageFolder(root='FER/test',  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)
    
    model = CNNFER().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, device, epochs=100)
    test(model, test_loader, device)
    # Save PyTorch model
    torch.save(model.state_dict(), './model/cnn_fer.pth')

    # Export to ONNX
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    model.eval()
    torch.onnx.export(model, dummy_input, './model/cnn_fer.onnx',
                    input_names=['input'], output_names=['output'],
                    opset_version=17)
    print("Exported model to cnn_fer.onnx")

if __name__ == '__main__':
    main()
