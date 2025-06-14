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
    model = CNNFER().to(device)
    model.load_state_dict(torch.load("./model/cnn_fer.pth", map_location=device,weights_only=True))
    model.eval()
    dummy_input = torch.randn(1, 1, 48, 48).to(device)
    torch.onnx.export(model, dummy_input, './model/cnn_fer.onnx',
                    input_names=['input'], output_names=['output'],
                    opset_version=17)
    print("Exported model to cnn_fer.onnx")