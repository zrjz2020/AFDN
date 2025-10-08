import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict


class BoeingFewShotDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, n_way=2, k_shot=3, k_query=2):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        # 支持 .jpg, .jpeg, .png 后缀
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.labels = [f.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') for f in self.images]
        self.class_to_images = defaultdict(list)

        for img, lbl in zip(self.images, self.labels):
            label_path = os.path.join(label_dir, lbl)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        print(f"警告: 标签文件 {lbl} 为空，跳过 (位于 {label_dir})")
                        continue
                    # 提取第一行的类 ID
                    first_line = lines[0].strip()
                    if not first_line:
                        print(f"警告: 标签文件 {lbl} 第一行为空，跳过 (位于 {label_dir})")
                        continue
                    label = int(float(first_line.split()[0]))  # 取第一列并转换为整数
                    # 检查所有行的类 ID 是否一致
                    for line in lines[1:]:
                        if line.strip():
                            other_label = int(float(line.strip().split()[0]))
                            if other_label != label:
                                print(f"警告: 标签文件 {lbl} 包含不同类 ID ({label} 和 {other_label})，跳过 (位于 {label_dir})")
                                break
                    else:  # 如果所有行的类 ID 一致
                        self.class_to_images[label].append(img)
            except Exception as e:
                print(f"处理标签文件 {lbl} 时出错: {e} (位于 {label_dir})")
                continue

        print(f"数据集目录: {image_dir}")
        for cls, imgs in self.class_to_images.items():
            print(f"类别 {cls}: {len(imgs)} 张图像")

        if len(self.class_to_images) < n_way:
            raise ValueError(f"数据集 {image_dir} 的可用类别数 ({len(self.class_to_images)}) 小于 n_way ({n_way})")
        for cls, imgs in self.class_to_images.items():
            if len(imgs) < k_shot + k_query:
                raise ValueError(
                    f"数据集 {image_dir} 中类别 {cls} 的图像数 ({len(imgs)}) 不足以支持 k_shot ({k_shot}) 和 k_query ({k_query})")

    def __len__(self):
        return 300  # 每个 epoch 生成 1000 个 episode

    def __getitem__(self, idx):
        support_set, query_set = [], []
        classes = np.random.choice(list(self.class_to_images.keys()), self.n_way, replace=False)
        for cls in classes:
            support_imgs = np.random.choice(self.class_to_images[cls], self.k_shot, replace=False)
            for img in support_imgs:
                img_path = os.path.join(self.image_dir, img)
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                        if not isinstance(image, torch.Tensor):
                            raise ValueError(f"支持集图像 {img} 不是张量: {type(image)} (位于 {self.image_dir})")
                        if image.dim() != 3 or image.shape != (3, 84, 84):
                            raise ValueError(f"支持集图像 {img} 的张量形状不正确: {image.shape} (位于 {self.image_dir})")
                    support_set.append((image, int(cls)))  # 确保 cls 是整数
                except Exception as e:
                    print(f"加载支持集图像 {img} 失败: {e} (位于 {self.image_dir})")
                    continue
            query_candidates = [img for img in self.class_to_images[cls] if img not in support_imgs]
            if len(query_candidates) < self.k_query:
                query_imgs = np.random.choice(self.class_to_images[cls], self.k_query, replace=True)
            else:
                query_imgs = np.random.choice(query_candidates, self.k_query, replace=False)
            for img in query_imgs:
                img_path = os.path.join(self.image_dir, img)
                try:
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                        if not isinstance(image, torch.Tensor):
                            raise ValueError(f"查询集图像 {img} 不是张量: {type(image)} (位于 {self.image_dir})")
                        if image.dim() != 3 or image.shape != (3, 84, 84):
                            raise ValueError(f"查询集图像 {img} 的张量形状不正确: {image.shape} (位于 {self.image_dir})")
                    query_set.append((image, int(cls)))  # 确保 cls 是整数
                except Exception as e:
                    print(f"加载查询集图像 {img} 失败: {e} (位于 {self.image_dir})")
                    continue

        if len(support_set) < self.n_way * self.k_shot or len(query_set) < self.n_way * self.k_query:
            raise ValueError(
                f"Episode {idx} 数据不足: 支持集大小={len(support_set)} (预期 {self.n_way * self.k_shot}), 查询集大小={len(query_set)} (预期 {self.n_way * self.k_query})")
        return support_set, query_set


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super(ProtoNet, self).__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):
        support_features = self.backbone(support_images)
        query_features = self.backbone(query_images)
        prototypes = []
        for cls in torch.unique(support_labels):
            mask = support_labels == cls
            prototype = support_features[mask].mean(0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        dists = torch.cdist(query_features, prototypes)
        return -dists  # 使用负距离作为 logits


transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_img_dir = r"your path"
train_label_dir = r"your path"
support_img_dir = r"your path"
support_label_dir = r"your path"
query_img_dir = r"your path"
output_dir = r"your path"

try:
    train_dataset = BoeingFewShotDataset(train_img_dir, train_label_dir, transform, n_way=2, k_shot=3, k_query=2)
    support_dataset = BoeingFewShotDataset(support_img_dir, support_label_dir, transform, n_way=2, k_shot=3, k_query=2)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
except Exception as e:
    print(f"数据集加载失败: {e}")
    exit()


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 21 * 21, 64)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = ConvNet().to(device)
model = ProtoNet(backbone).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_protonet(model, train_loader, criterion, optimizer, num_episodes=10):
    model.train()
    for episode in range(num_episodes):
        running_loss = 0.0
        for batch in train_loader:
            try:
                support_set, query_set = batch  # 解包 DataLoader 返回的 batch

                print(f"Episode {episode + 1}: 支持集 = {[(x[0].shape, x[1]) for x in support_set]}")
                print(f"Episode {episode + 1}: 查询集 = {[(x[0].shape, x[1]) for x in query_set]}")

                support_tensors = []
                for x in support_set:
                    tensor = x[0]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"支持集包含非张量数据: {type(tensor)}")
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)  # 移除额外的批次维度
                    if tensor.dim() != 3 or tensor.shape != (3, 84, 84):
                        raise ValueError(f"支持集张量形状不正确: {tensor.shape}")
                    support_tensors.append(tensor.unsqueeze(0))
                support_images = torch.cat(support_tensors, dim=0).to(device)

                support_labels = torch.tensor(
                    [int(x[1]) if isinstance(x[1], (int, torch.Tensor)) else x[1] for x in support_set],
                    dtype=torch.long).to(device)

                query_tensors = []
                for x in query_set:
                    tensor = x[0]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"查询集包含非张量数据: {type(tensor)}")
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)  # 移除额外的批次维度
                    if tensor.dim() != 3 or tensor.shape != (3, 84, 84):
                        raise ValueError(f"查询集张量形状不正确: {tensor.shape}")
                    query_tensors.append(tensor.unsqueeze(0))
                query_images = torch.cat(query_tensors, dim=0).to(device)

                query_labels = torch.tensor(
                    [int(x[1]) if isinstance(x[1], (int, torch.Tensor)) else x[1] for x in query_set],
                    dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(support_images, support_labels, query_images)
                loss = criterion(outputs, query_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            except Exception as e:
                print(f"Episode {episode + 1} 处理失败: {e}")
                continue
        if (episode + 1) % 100 == 0:
            print(f"Episode [{episode + 1}/{num_episodes}], Loss: {running_loss / len(train_loader):.4f}")


def validate_protonet(model, support_dataset, num_episodes=2):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(num_episodes):
            try:
                support_set, query_set = support_dataset[0]  # 获取一个 episode
                # 调试：检查支持集和查询集的结构
                print(f"验证集支持集: {[(x[0].shape, x[1]) for x in support_set]}")
                print(f"验证集查询集: {[(x[0].shape, x[1]) for x in query_set]}")

                # 确保支持集中的每个张量是 3 维
                support_tensors = []
                for x in support_set:
                    tensor = x[0]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"支持集包含非张量数据: {type(tensor)}")
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)  # 移除额外的批次维度
                    if tensor.dim() != 3 or tensor.shape != (3, 84, 84):
                        raise ValueError(f"支持集张量形状不正确: {tensor.shape}")
                    support_tensors.append(tensor.unsqueeze(0))
                support_images = torch.cat(support_tensors, dim=0).to(device)

                support_labels = torch.tensor(
                    [int(x[1]) if isinstance(x[1], (int, torch.Tensor)) else x[1] for x in support_set],
                    dtype=torch.long).to(device)

                query_tensors = []
                for x in query_set:
                    tensor = x[0]
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"查询集包含非张量数据: {type(tensor)}")
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)  # 移除额外的批次维度
                    if tensor.dim() != 3 or tensor.shape != (3, 84, 84):
                        raise ValueError(f"查询集张量形状不正确: {tensor.shape}")
                    query_tensors.append(tensor.unsqueeze(0))
                query_images = torch.cat(query_tensors, dim=0).to(device)

                query_labels = torch.tensor(
                    [int(x[1]) if isinstance(x[1], (int, torch.Tensor)) else x[1] for x in query_set],
                    dtype=torch.long).to(device)

                outputs = model(support_images, support_labels, query_images)
                _, predicted = torch.max(outputs, 1)
                total += query_labels.size(0)
                correct += (predicted == query_labels).sum().item()
            except Exception as e:
                print(f"验证集处理失败: {e}")
                continue
    accuracy = correct / total if total > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")


def predict_protonet(model, support_dataset, query_img_dir, transform):
    model.eval()
    predictions = []
    # 支持 .jpg, .jpeg, .png 后缀
    image_files = [f for f in os.listdir(query_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    try:
        support_set, _ = support_dataset[0]  # 获取一个支持集

        print(f"预测支持集: {[(x[0].shape, x[1]) for x in support_set]}")

        support_tensors = []
        for x in support_set:
            tensor = x[0]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"支持集包含非张量数据: {type(tensor)}")
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)  # 移除额外的批次维度
            if tensor.dim() != 3 or tensor.shape != (3, 84, 84):
                raise ValueError(f"支持集张量形状不正确: {tensor.shape}")
            support_tensors.append(tensor.unsqueeze(0))
        support_images = torch.cat(support_tensors, dim=0).to(device)

        support_labels = torch.tensor(
            [int(x[1]) if isinstance(x[1], (int, torch.Tensor)) else x[1] for x in support_set], dtype=torch.long).to(
            device)
    except Exception as e:
        print(f"加载支持集失败: {e}")
        return predictions

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(query_img_dir, img_file)
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)
                outputs = model(support_images, support_labels, image)
                _, pred = torch.max(outputs, 1)
                predictions.append((img_file, pred.item()))
            except Exception as e:
                print(f"预测图像 {img_file} 失败: {e}")
                continue
    return predictions


try:
    train_protonet(model, train_loader, criterion, optimizer, num_episodes=300)
    validate_protonet(model, support_dataset, num_episodes=30)
except Exception as e:
    print(f"训练或验证失败: {e}")

try:
    predictions = predict_protonet(model, support_dataset, query_img_dir, transform)
    df = pd.DataFrame(predictions, columns=['Image', 'Prediction'])
    output_path = os.path.join(output_dir, 'protonet_predictions.csv')
    df.to_csv(output_path, index=False)
    print(f"预测结果已保存至: {output_path}")
except Exception as e:
    print(f"预测失败: {e}")
