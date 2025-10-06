import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_label_file(label_path):
    """读取标注文件，返回所有边界框的坐标列表（center_x, center_y, width, height）"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # 提取边界框坐标，忽略class_id
            center_x, center_y, width, height = map(float, parts[1:5])
            bboxes.append((center_x, center_y, width, height))
        return bboxes if bboxes else None
    except Exception as e:
        print(f"错误：读取{label_path}时发生错误：{str(e)}")
        return None


def crop_image(image, bbox, img_width, img_height):
    """根据边界框裁剪图像"""
    if bbox is None:
        return None
    center_x, center_y, width, height = bbox
    # 将归一化坐标转换为像素坐标
    x_min = int((center_x - width / 2) * img_width)
    x_max = int((center_x + width / 2) * img_width)
    y_min = int((center_y - height / 2) * img_height)
    y_max = int((center_y + height / 2) * img_height)
    # 确保坐标有效
    x_min, x_max = max(0, x_min), min(img_width, x_max)
    y_min, y_max = max(0, y_min), min(img_height, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    return image.crop((x_min, y_min, x_max, y_max))


def extract_embedding(image_path, bboxes):
    try:
        # 读取图像
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size

        embeddings = []
        for i, bbox in enumerate(bboxes):
            # 裁剪图像
            cropped_img = crop_image(img, bbox, img_width, img_height)
            if cropped_img is None:
                print(f"无效的边界框 {i} 在 {image_path}")
                continue

            # 预处理裁剪后的图像
            img_tensor = preprocess(cropped_img).unsqueeze(0).to(device)

            # 提取embedding
            with torch.no_grad():
                embedding = model(img_tensor)

            # 展平embedding
            embeddings.append((i, embedding.squeeze().cpu().numpy()))

        return embeddings if embeddings else None

    except Exception as e:
        print(f"错误：处理{image_path}时发生错误：{str(e)}")
        return None


def main():
    image_dir = "your path"
    label_dir = "your path"
    output_dir = "your path"

    os.makedirs(output_dir, exist_ok=True)

    valid_image_extensions = ('.jpg', '.jpeg', '.png')
    valid_label_extension = '.txt'

    # 遍历图像目录
    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(valid_image_extensions):
            img_path = os.path.join(image_dir, img_name)
            # 构建对应的标注文件路径
            label_name = Path(img_name).stem + valid_label_extension
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(label_path):
                print(f"错误：未找到{img_name}对应的标注文件：{label_path}")
                continue

            try:
                # 读取所有边界框
                bboxes = read_label_file(label_path)
                if bboxes is None:
                    print(f"错误：{img_name}的标注文件无有效边界框")
                    continue

                # 提取所有边界框的embedding
                embeddings = extract_embedding(img_path, bboxes)
                if embeddings is None:
                    print(f"错误：{img_name}无法提取embedding")
                    continue

                # 保存每个边界框的embedding
                for bbox_index, embedding in embeddings:
                    output_path = os.path.join(output_dir, f"{Path(img_name).stem}_embedding_{bbox_index}.txt")
                    np.savetxt(output_path, embedding, fmt='%.6f')
                    print(f"已处理：{img_name} 的边界框 {bbox_index}，保存至 {output_path}")
            except Exception as e:
                print(f"错误：处理{img_name}时发生错误：{str(e)}")


if __name__ == "__main__":
    main()
