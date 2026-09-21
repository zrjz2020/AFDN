import os
import numpy as np
from pathlib import Path
from PIL import Image

import feature_extractor as fe


def read_label_file(label_path):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            center_x, center_y, width, height = map(float, parts[1:5])
            bboxes.append((class_id, center_x, center_y, width, height))
        return bboxes if bboxes else None
    except Exception as e:
        print(f"错误：读取{label_path}时发生错误：{str(e)}")
        return None


def crop_image(image, bbox, img_width, img_height):
    if bbox is None:
        return None
    class_id, center_x, center_y, width, height = bbox
    x_min = int((center_x - width / 2) * img_width)
    x_max = int((center_x + width / 2) * img_width)
    y_min = int((center_y - height / 2) * img_height)
    y_max = int((center_y + height / 2) * img_height)
    x_min, x_max = max(0, x_min), min(img_width, x_max)
    y_min, y_max = max(0, y_min), min(img_height, y_max)
    if x_max <= x_min or y_max <= y_min:
        return None
    return image.crop((x_min, y_min, x_max, y_max))


def extract_embedding(image_path, bboxes):
    try:
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size

        embeddings = []
        for i, bbox in enumerate(bboxes):
            cropped_img = crop_image(img, bbox, img_width, img_height)
            if cropped_img is None:
                print(f"无效的边界框 {i} 在 {image_path}")
                continue

            embedding = fe.extract(cropped_img)

            embeddings.append((i, bbox[0], embedding))

        return embeddings if embeddings else None

    except Exception as e:
        print(f"错误：处理{image_path}时发生错误：{str(e)}")
        return None


def main():
    image_dir = "./datasets/BoeingFewShot/S/images"
    label_dir = "./datasets/BoeingFewShot/S/labels"
    output_root = "./datasets/BoeingFewShot/embeddings"
    CLASS_DIR = {0: "S", 1: "T"}
    output_dir = os.path.join(output_root, "S_Embeddings")

    os.makedirs(output_dir, exist_ok=True)
    for d in CLASS_DIR.values():
        os.makedirs(os.path.join(output_root, d), exist_ok=True)

    valid_image_extensions = ('.jpg', '.jpeg', '.png')
    valid_label_extension = '.txt'

    for img_name in os.listdir(image_dir):
        if img_name.lower().endswith(valid_image_extensions):
            img_path = os.path.join(image_dir, img_name)
            label_name = Path(img_name).stem + valid_label_extension
            label_path = os.path.join(label_dir, label_name)

            if not os.path.exists(label_path):
                print(f"错误：未找到{img_name}对应的标注文件：{label_path}")
                continue

            try:
                bboxes = read_label_file(label_path)
                if bboxes is None:
                    print(f"错误：{img_name}的标注文件无有效边界框")
                    continue

                embeddings = extract_embedding(img_path, bboxes)
                if embeddings is None:
                    print(f"错误：{img_name}无法提取embedding")
                    continue

                stem = Path(img_name).stem
                for bbox_index, class_id, embedding in embeddings:
                    if class_id in CLASS_DIR:
                        class_out_path = os.path.join(output_root, CLASS_DIR[class_id],
                                                      f"{stem}_embedding_{bbox_index}.txt")
                        np.savetxt(class_out_path, embedding, fmt='%.6f')
                    else:
                        print(f"警告：{img_name} 边界框 {bbox_index} 的类别 id {class_id} 未定义，跳过分流")
                    output_path = os.path.join(output_dir, f"{stem}_embedding_{bbox_index}.txt")
                    np.savetxt(output_path, embedding, fmt='%.6f')
                    print(f"已处理：{img_name} 的边界框 {bbox_index} (class={class_id})，保存至 {output_path}")
            except Exception as e:
                print(f"错误：处理{img_name}时发生错误：{str(e)}")


if __name__ == "__main__":
    main()
