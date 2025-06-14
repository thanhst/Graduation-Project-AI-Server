import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ---------------------- CNN Model ----------------------
class CNNFER(nn.Module):
    def __init__(self):
        super(CNNFER, self).__init__()
        self.features = nn.Sequential(
            self.conv_block(1, 64,num_conv=1),
            self.conv_block(64, 128,num_conv=3),
            self.conv_block(128, 256,num_conv=2),
            self.conv_block(256, 512,num_conv=3),
            self.conv_block(512, 512,num_conv=4),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 7)
        )

    def conv_block(self, in_ch, out_ch,num_conv = 2):
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU())

        for _ in range(num_conv - 1):
            layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

# ---------------------- Train Function ----------------------
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 5
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", dynamic_ncols=True,unit="batch")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(dataloader):.4f} - Acc: {accuracy:.2f}%")
        
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping")
            break
        

# ---------------------- Test ----------------------
def test(model, dataloader, device):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = sum(p == t for p, t in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total
    print(f"✅ Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=emotion_labels, yticklabels=emotion_labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotion_labels))
