import os
import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
import yaml
import gc


# 自定义数据集
class BoeingDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
        # 从 YAML 文件加载类别名称
        try:
            with open('./datasets/dataBoeingAug.yaml', 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            self.class_names = cfg['names']
        except UnicodeDecodeError as e:
            print(f"Error decoding YAML file: {e}")
            raise

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

        # 读取 YOLO 格式标签
        boxes = []
        labels = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(float(parts[0]))
                                if class_id not in [0, 1]:
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
                            except (IndexError, ValueError):
                                print(f"Warning: Invalid label format in {label_path}")
            except UnicodeDecodeError as e:
                print(f"Error decoding label file {label_path}: {e}")
        else:
            print(f"Warning: Label file {label_path} is missing or empty")

        # 转换为张量
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
    """
    获取下一个实验目录
    """
    os.makedirs(base_path, exist_ok=True)
    exp_dirs = [d for d in os.listdir(base_path) if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))]
    exp_nums = [int(d.replace("exp", "")) for d in exp_dirs if d.replace("exp", "").isdigit()]
    next_exp_num = max(exp_nums, default=0) + 1
    return os.path.join(base_path, f"exp{next_exp_num}")


def print_common_classes(neu_yaml_path='./datasets/dataneuAug.yaml', boeing_yaml_path='./datasets/dataBoeingAug.yaml'):
    """
    打印两个 YAML 文件中的共同类别
    """
    try:
        with open(neu_yaml_path, 'r', encoding='utf-8') as f:
            neu_cfg = yaml.safe_load(f)
        with open(boeing_yaml_path, 'r', encoding='utf-8') as f:
            boeing_cfg = yaml.safe_load(f)

        # 从字典中提取类别名称（值）
        neu_classes = set(neu_cfg['names'].values())
        boeing_classes = set(boeing_cfg['names'].values())
        common_classes = neu_classes.intersection(boeing_classes)

        if common_classes:
            print(f"Common classes between {neu_yaml_path} and {boeing_yaml_path}: {', '.join(common_classes)}")
        else:
            print(f"No common classes found between {neu_yaml_path} and {boeing_yaml_path}")

        return common_classes
    except Exception as e:
        print(f"Error reading YAML files for class comparison: {e}")
        return set()


def map_pretrained_weights(model, pretrained_weights, neu_class_names, boeing_class_names, device):
    """
    映射 Neu 数据集的预训练权重到 Boeing 数据集模型，保留共同类别的知识
    """
    try:
        state_dict = torch.load(pretrained_weights, map_location=device)
        model_dict = model.state_dict()

        # neu_class_names 和 boeing_class_names 是字典，键是索引，值是类别名称
        # 创建类别名称到索引的映射（Faster R-CNN 标签从 1 开始，0 为背景）
        neu_classes = {name: idx + 1 for idx, name in neu_class_names.items()}
        boeing_classes = {name: idx + 1 for idx, name in boeing_class_names.items()}
        common_classes = set(neu_classes.keys()).intersection(set(boeing_classes.keys()))

        # 初始化新的权重和偏置张量
        cls_score_key = 'roi_heads.box_predictor.cls_score.weight'
        cls_bias_key = 'roi_heads.box_predictor.cls_score.bias'
        bbox_pred_key = 'roi_heads.box_predictor.bbox_pred.weight'
        bbox_pred_bias_key = 'roi_heads.box_predictor.bbox_pred.bias'

        if cls_score_key in state_dict:
            pretrained_weight = state_dict[cls_score_key]
            pretrained_bias = state_dict[cls_bias_key]
            new_weight = torch.zeros((len(boeing_class_names) + 1, pretrained_weight.size(1)),
                                     dtype=pretrained_weight.dtype)
            new_bias = torch.zeros(len(boeing_class_names) + 1, dtype=pretrained_bias.dtype)

            # 复制背景类权重 (index 0)
            new_weight[0] = pretrained_weight[0]
            new_bias[0] = pretrained_bias[0]

            # 映射共同类别的权重
            for class_name, boeing_idx in boeing_classes.items():
                if class_name in common_classes:
                    neu_idx = neu_classes[class_name]
                    new_weight[boeing_idx] = pretrained_weight[neu_idx]
                    new_bias[boeing_idx] = pretrained_bias[neu_idx]
                else:
                    # 对于非共同类别，保持随机初始化（已在 zeros 中处理）
                    pass

            model_dict[cls_score_key] = new_weight
            model_dict[cls_bias_key] = new_bias

        if bbox_pred_key in state_dict:
            pretrained_weight = state_dict[bbox_pred_key]
            pretrained_bias = state_dict[bbox_pred_bias_key]
            new_weight = torch.zeros((len(boeing_class_names) * 4 + 4, pretrained_weight.size(1)),
                                     dtype=pretrained_weight.dtype)
            new_bias = torch.zeros(len(boeing_class_names) * 4 + 4, dtype=pretrained_bias.dtype)

            # 复制背景类权重 (index 0-3)
            new_weight[0:4] = pretrained_weight[0:4]
            new_bias[0:4] = pretrained_bias[0:4]

            # 映射共同类别的边界框回归权重
            for class_name, boeing_idx in boeing_classes.items():
                if class_name in common_classes:
                    neu_idx = neu_classes[class_name]
                    pretrained_start_idx = neu_idx * 4
                    new_start_idx = boeing_idx * 4
                    new_weight[new_start_idx:new_start_idx + 4] = pretrained_weight[
                                                                  pretrained_start_idx:pretrained_start_idx + 4]
                    new_bias[new_start_idx:new_start_idx + 4] = pretrained_bias[
                                                                pretrained_start_idx:pretrained_start_idx + 4]
                else:
                    # 对于非共同类别，保持随机初始化
                    pass

            model_dict[bbox_pred_key] = new_weight
            model_dict[bbox_pred_bias_key] = new_bias

        # 加载其他匹配的权重（骨干网络、RPN 等）
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"已加载预训练权重并映射共同类别: {pretrained_weights}")
        if common_classes:
            print(f"保留的共同类别: {', '.join(common_classes)}")
        else:
            print("无共同类别，分类层部分随机初始化")
        return model
    except Exception as e:
        print(f"加载预训练权重失败 {pretrained_weights}: {e}")
        print("使用随机初始化")
        return model


def train_model(data_dir=None, epochs=100, batch_size=8, pretrained_weights="your path/best.pt"):
    """
    训练 Faster R-CNN 模型
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 YAML 配置文件
    try:
        with open('./datasets/dataBoeingAug.yaml', 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except UnicodeDecodeError as e:
        print(f"Error decoding YAML file: {e}")
        raise

    # 加载 Neu 数据集的 YAML 文件以获取类别名称
    try:
        with open('./datasets/dataneuAug.yaml', 'r', encoding='utf-8') as f:
            neu_cfg = yaml.safe_load(f)
    except UnicodeDecodeError as e:
        print(f"Error decoding Neu YAML file: {e}")
        raise

    # 打印共同类别
    print_common_classes('./datasets/dataneuAug.yaml', './datasets/dataBoeingAug.yaml')

    # 数据集路径
    if data_dir is None:
        data_dir = cfg['path']
    train_img_dir = os.path.join(data_dir, cfg['train'])
    train_label_dir = os.path.join(data_dir, cfg['train'].replace('images', 'labels'))
    val_img_dir = os.path.join(data_dir, cfg['val'])
    val_label_dir = os.path.join(data_dir, cfg['val'].replace('images', 'labels'))

    # 检查数据路径
    for d in [train_img_dir, train_label_dir, val_img_dir, val_label_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"数据目录不存在: {d}")

    # 数据集
    train_dataset = BoeingDataset(train_img_dir, train_label_dir)
    val_dataset = BoeingDataset(val_img_dir, val_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=lambda x: tuple(zip(*x)))

    # 加载 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(cfg['names']) + 1)  # 2 类 + 背景
    if os.path.exists(pretrained_weights):
        try:
            model = map_pretrained_weights(model, pretrained_weights, neu_cfg['names'], cfg['names'], device)
        except Exception as e:
            print(f"加载预训练权重失败 {pretrained_weights}: {e}")
            print("使用随机初始化")
    else:
        print(f"警告: 预训练权重文件 {pretrained_weights} 不存在，使用随机初始化")
    model.to(device)

    # 优化器（降低学习率以进行微调）
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

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
            torch.save(model.state_dict(), os.path.join(exp_dir, "your path/best.pt"))

        lr_scheduler.step()

    print(f"训练完成，最佳模型保存到 {os.path.join(exp_dir, 'your path/best.pt')}")
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


