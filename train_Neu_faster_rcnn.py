import os
import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import yaml
import gc


with open('dataneuAug.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)


class NeuDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        self.class_names = cfg['names']  # 从 YAML 文件加载类别名称

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法加载图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # 读取标签
        boxes = []
        labels = []
        if os.path.exists(label_path):
            if os.path.getsize(label_path) == 0:
                print(f"Warning: Label file {label_path} is empty")
            else:
                try:
                    with open(label_path, 'r', encoding='utf-8') as f:
                        for line in f.read().strip().split('\n'):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    class_id = int(float(parts[0]))
                                    if class_id not in range(len(self.class_names)):
                                        continue
                                    center_x, center_y, width, height = map(float, parts[1:5])
                                    x_min = (center_x - width / 2) * w
                                    y_min = (center_y - height / 2) * h
                                    x_max = (center_x + width / 2) * w
                                    y_max = (center_y + height / 2) * h
                                    x_min, y_min = max(0, x_min), max(0, y_min)
                                    x_max, y_max = min(w, x_max), min(h, y_max)
                                    if x_max > x_min and y_max > y_min:
                                        boxes.append([x_min, y_min, x_max, y_max])
                                        labels.append(class_id + 1)  # Faster R-CNN 标签从 1 开始（0 为背景）
                                except (IndexError, ValueError) as e:
                                    print(f"Warning: Invalid label format in {label_path}: {str(e)}")
                except UnicodeDecodeError as e:
                    print(f"Warning: Failed to decode {label_path} with UTF-8: {str(e)}")
        else:
            print(f"Warning: Label file {label_path} does not exist")

        img = transforms.ToTensor()(img)
        if self.transform:
            img = self.transform(img)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }
        return img, target

def get_next_exp_dir(base_path="runs/train"):
    os.makedirs(base_path, exist_ok=True)
    exp_dirs = [d for d in os.listdir(base_path) if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))]
    exp_nums = [int(d.replace("exp", "")) for d in exp_dirs if d.replace("exp", "").isdigit()]
    next_exp_num = max(exp_nums, default=0) + 1
    return os.path.join(base_path, f"exp{next_exp_num}")

def train_model(data_dir=cfg['path'], epochs=20, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集路径
    train_img_dir = os.path.join(data_dir, cfg['train'])
    train_label_dir = os.path.join(data_dir, cfg['train'].replace('images', 'labels'))
    val_img_dir = os.path.join(data_dir, cfg['val'])
    val_label_dir = os.path.join(data_dir, cfg['val'].replace('images', 'labels'))

    # 检查数据路径
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"数据目录不存在: {d}")

    # 数据集
    train_dataset = NeuDataset(train_img_dir, train_label_dir)
    val_dataset = NeuDataset(val_img_dir, val_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)))

    # 加载 Faster R-CNN 模型
    num_classes = len(cfg['names']) + 1  # 类别数 + 背景
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # 训练
    exp_dir = get_next_exp_dir()
    os.makedirs(os.path.join(exp_dir, "weights"), exist_ok=True)
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 检查目标数据
            for t in targets:
                if t["boxes"].shape[0] == 0:
                    print("Warning: Empty boxes in batch, skipping loss computation")
                    continue

            try:
                loss_dict = model(images, targets)
                if isinstance(loss_dict, list):
                    print(f"Error: loss_dict is a list in training: {loss_dict}")
                    continue
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                train_loss += losses.item()
            except Exception as e:
                print(f"Loss computation failed: {str(e)}")
                continue
        train_loss = train_loss / max(1, len(train_loader))

        # 验证
        val_loss = 0.0
        with torch.no_grad():
            model.train()  # 临时切换到训练模式以计算损失
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                try:
                    loss_dict = model(images, targets)
                    if isinstance(loss_dict, list):
                        print(f"Error: loss_dict is a list in validation: {loss_dict}")
                        continue
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                except Exception as e:
                    print(f"Validation loss computation failed: {str(e)}")
                    continue
            model.eval()  # 恢复 eval 模式
        val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "weights/best.pt"))

        lr_scheduler.step()

    print(f"训练完成，最佳模型保存到 {os.path.join(exp_dir, 'weights/best.pt')}")
    return exp_dir

if __name__ == "__main__":
    try:
        exp_dir = train_model()
        print(f"训练结果保存到 {exp_dir}")
    except Exception as e:
        print(f"训练失败: {str(e)}")
    # Clean up
    torch.cuda.empty_cache()  # 释放显存
    gc.collect()  # 强制垃圾回收