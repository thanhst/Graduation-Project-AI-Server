import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from module.CNN import CNNFER
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd

def test(model, dataloader, device, class_names):
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
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    macro = report["macro avg"]
    print(f"\n🔎 Macro Average — Precision: {macro['precision']:.4f}, Recall: {macro['recall']:.4f}, F1-score: {macro['f1-score']:.4f}")

    # 📊 Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 📋 Classification Report
    print("\n📊 Classification Report:")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    for cls in class_names:
        precision = report[cls]['precision']
        recall = report[cls]['recall']
        f1 = report[cls]['f1-score']
        print(f"{cls:<10s} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")
    

    # 📈 Vẽ biểu đồ đường (Line Chart)
    df_report = pd.DataFrame(report).transpose()
    df_plot = df_report.loc[class_names, ['precision', 'recall', 'f1-score']]
    
    plt.figure(figsize=(10, 6))
    for metric in df_plot.columns:
        plt.plot(class_names, df_plot[metric], marker='o', label=metric.capitalize())
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xlabel('Emotion Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = CNNFER().to(device)

    model.load_state_dict(torch.load("./model/cnn_fer.pth", map_location=device,weights_only=True))

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise' ]
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = ImageFolder(root='FER/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test(model, test_loader, device, emotion_labels)


if __name__ == "__main__":
    main()
