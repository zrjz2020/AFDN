import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional
from scipy.spatial.distance import cosine


ROOT_DIR = r"your path"
VALID_EXTS = (".jpg", ".jpeg", ".png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # 输出形状 (B, 2048, 1, 1)
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@torch.no_grad()
def extract_embedding_for_image(img_path: str) -> Optional[np.ndarray]:
    """对单张图像提取 2048 维 embedding（展平）。出错返回 None。"""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"跳过：无法打开图像 {img_path}，原因：{e}")
        return None

    try:
        img_tensor = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        feat = model(img_tensor)  # (1, 2048, 1, 1)
        emb = feat.squeeze().detach().cpu().numpy()  # (2048,)
        return emb
    except Exception as e:
        print(f"跳过：提取 {img_path} 的 embedding 失败，原因：{e}")
        return None


def process_subfolder(folder_path: str):
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"不是目录：{folder}")
        return

    while True:
        # 收集该子文件夹内所有图片路径
        img_paths = [str(p) for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in VALID_EXTS]

        if not img_paths:
            print(f"无有效图片：{folder}")
            return

        if len(img_paths) == 1:
            # 只剩一张图片，保存其 embedding
            emb = extract_embedding_for_image(img_paths[0])
            if emb is not None:
                out_name = f"final_embedding_{folder.name}.txt"
                out_path = folder / out_name
                try:
                    np.savetxt(out_path, emb, fmt="%.6f")
                    print(f"已保存最后一张图片的 embedding：{out_path}")
                except Exception as e:
                    print(f"保存失败：{out_path}，原因：{e}")
            else:
                print(f"无法提取最后一张图片的 embedding：{img_paths[0]}")
            return


        embs = []
        valid_img_paths = []
        for p in img_paths:
            emb = extract_embedding_for_image(p)
            if emb is not None:
                embs.append(emb)
                valid_img_paths.append(p)

        if not embs:
            print(f"无有效 embedding：{folder}")
            return

        # 计算平均 embedding
        embs_np = np.stack(embs, axis=0)  # (N, 2048)
        ave_emb = embs_np.mean(axis=0)    # (2048,)

        # 计算每张图片的 embedding 与平均 embedding 的余弦相似度
        similarities = [1 - cosine(emb, ave_emb) for emb in embs]  # 余弦相似度 = 1 - 余弦距离
        min_similarity_idx = np.argmin(similarities)  # 相似度最低的索引

        # 删除相似度最低的图片
        try:
            os.remove(valid_img_paths[min_similarity_idx])
            print(f"已删除最不相似的图片：{valid_img_paths[min_similarity_idx]}，余弦相似度：{similarities[min_similarity_idx]:.6f}")
        except Exception as e:
            print(f"删除失败：{valid_img_paths[min_similarity_idx]}，原因：{e}")
            return


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        print(f"根目录不存在：{root}")
        return

    # 仅遍历一级子文件夹
    subfolders = [str(p) for p in root.iterdir() if p.is_dir()]
    if not subfolders:
        print(f"目录下无子文件夹：{root}")
        return

    print(f"开始处理，共 {len(subfolders)} 个子文件夹...")
    for sf in subfolders:
        print(f"处理子文件夹：{sf}")
        process_subfolder(sf)

    print("全部完成。")


if __name__ == "__main__":
    main()