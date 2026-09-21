import os
import sys
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import crops
import feature_extractor as fe
from GetEmbedding import read_label_file, crop_image

RCNN_RUNS = os.path.join(PROJECT_ROOT, "runs", "train")
S_IMG_DIR = os.path.join(PROJECT_ROOT, "datasets", "BoeingFewShot", "S", "images")
S_LBL_DIR = os.path.join(PROJECT_ROOT, "datasets", "BoeingFewShot", "S", "labels")
OUT_PATH = os.path.join(PROJECT_ROOT, "weights", "afdn_model.pt")

CONFIG = {
    "conf_thr": 0.5,
    "max_crops": 6,
    "nms_iou": 0.5,
    "sim_threshold": 0.6,
    "class_names": {1: "scratches", 2: "stain"},
}


def build_prototypes():
    id2name = {0: "scratches", 1: "stain"}
    buckets = {"scratches": [], "stain": []}
    for fn in os.listdir(S_IMG_DIR):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        stem = os.path.splitext(fn)[0]
        lbl_path = os.path.join(S_LBL_DIR, stem + ".txt")
        if not os.path.isfile(lbl_path):
            continue
        bboxes = read_label_file(lbl_path)
        if not bboxes:
            continue
        img = Image.open(os.path.join(S_IMG_DIR, fn)).convert("RGB")
        W, H = img.size
        for bb in bboxes:
            if bb[0] not in id2name:
                continue
            cropped = crop_image(img, bb, W, H)
            if cropped is None:
                continue
            buckets[id2name[bb[0]]].append(fe.extract(cropped))
    protos = {}
    for name, embs in buckets.items():
        if not embs:
            raise RuntimeError(f"Support 集无类别 {name} 的 GT 框")
        v = np.mean(np.stack(embs), axis=0)
        protos[name] = v / (np.linalg.norm(v) + 1e-8)
        print(f"[原型] {name}: {len(embs)} 个 GT crop")
    return protos


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = crops.find_latest_rcnn_best(base=RCNN_RUNS)
    print(f"[R-CNN] 权重: {weights_path}")
    rcnn_sd = torch.load(weights_path, map_location="cpu")

    fe.extract(Image.new("RGB", (224, 224)))
    dinov2_sd = fe._model.state_dict()
    print(f"[DINOv2] 权重: {fe._WEIGHTS}")

    protos = build_prototypes()

    package = {
        "version": "afdn-v1",
        "config": CONFIG,
        "rcnn_state_dict": rcnn_sd,
        "dinov2_state_dict": dinov2_sd,
        "prototypes": {k: torch.from_numpy(v.astype(np.float32)) for k, v in protos.items()},
        "rcnn_num_classes": 3,
        "emb_dim": 768,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(package, OUT_PATH)
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"\n✅ 打包完成: {OUT_PATH}")
    print(f"   文件大小: {size_mb:.1f} MB")
    print(f"   原型维度: scratches={protos['scratches'].shape}, stain={protos['stain'].shape}")


if __name__ == "__main__":
    main()
