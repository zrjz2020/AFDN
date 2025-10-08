import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.metrics import accuracy_score


class BoeingDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.labels = [f.replace('.jpg', '.txt') for f in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(img_path).convert('RGB')

        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                raise ValueError(f"标签文件 {label_path} 为空")
            label = int(float(lines[0].strip().split()[0]))  # 将 '0' 或 '1'（可能为 '0.0' 或 '1.0'）转换为整数

            for line in lines[1:]:
                if line.strip():
                    other_label = int(float(line.strip().split()[0]))
                    if other_label != label:
                        raise ValueError(f"标签文件 {label_path} 包含多个不同的类 ID: {label} 和 {other_label}")

        if self.transform:
            image = self.transform(image)

        return image, label

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_img_dir = r"your path"
train_label_dir = r"your path"
val_img_dir = r"your path"
val_label_dir = r"your path"
query_img_dir = r"your path"
output_dir = r"your path"

train_dataset = BoeingDataset(train_img_dir, train_label_dir, train_transform)
val_dataset = BoeingDataset(val_img_dir, val_label_dir, val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.vgg16(pretrained=True)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)  # 假设二分类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


def validate_model(model, val_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return predictions, true_labels


def predict_model(model, query_img_dir, transform):
    model.eval()
    predictions = []
    image_files = [f for f in os.listdir(query_img_dir) if f.endswith('.jpg')]
    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(query_img_dir, img_file)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            predictions.append((img_file, pred.item()))
    return predictions


train_model(model, train_loader, criterion, optimizer, num_epochs=10)
validate_model(model, val_loader)


predictions = predict_model(model, query_img_dir, val_transform)
df = pd.DataFrame(predictions, columns=['Image', 'Prediction'])
output_path = os.path.join(output_dir, 'vgg16_predictions.csv')
df.to_csv(output_path, index=False)
print(f"预测结果已保存至: {output_path}")
