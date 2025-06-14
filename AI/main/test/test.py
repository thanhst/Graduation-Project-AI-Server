import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from module.CNN import CNNFER
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

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

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNFER().to(device)

    model.load_state_dict(torch.load("./model/cnn_fer.pth", map_location=device,weights_only=True))

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy','Neutral', 'Sad', 'Surprise' ]
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor()
    ])
    
    test_dataset = ImageFolder(root='FER/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test(model, test_loader, device, emotion_labels)


if __name__ == "__main__":
    main()
