import argparse
import csv
import gc
import io
import builtins
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn



sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
_orig_print = builtins.print
def _print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _orig_print(*args, **kwargs)
builtins.print = _print_flush


CLASS_NAMES = {1: "scratches", 2: "stain"}



def _class_wise_nms(boxes, labels, scores, iou_thr=0.5):
    keep = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        c_boxes = boxes[cls_idx]
        c_scores = scores[cls_idx]
        order = np.argsort(-c_scores)
        suppressed = np.zeros(len(cls_idx), dtype=bool)
        for _i in range(len(order)):
            i = order[_i]
            if suppressed[i]:
                continue
            keep.append(cls_idx[i])
            x1 = c_boxes[i, 0]; y1 = c_boxes[i, 1]; x2 = c_boxes[i, 2]; y2 = c_boxes[i, 3]
            area_i = (x2 - x1) * (y2 - y1)
            for _j in range(_i + 1, len(order)):
                j = order[_j]
                if suppressed[j]:
                    continue
                xx1 = max(x1, c_boxes[j, 0]); yy1 = max(y1, c_boxes[j, 1])
                xx2 = min(x2, c_boxes[j, 2]); yy2 = min(y2, c_boxes[j, 3])
                w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
                inter = w * h
                area_j = (c_boxes[j, 2] - c_boxes[j, 0]) * (c_boxes[j, 3] - c_boxes[j, 1])
                iou = inter / max(1e-8, area_i + area_j - inter)
                if iou > iou_thr:
                    suppressed[j] = True
    keep.sort(key=lambda i: -scores[i])
    return keep


def _expand_bbox(x1, y1, x2, y2, W, H, expand_ratio=0.1):
    bw = x2 - x1
    bh = y2 - y1
    ex = bw * expand_ratio
    ey = bh * expand_ratio
    nx1 = int(max(0, x1 - ex))
    ny1 = int(max(0, y1 - ey))
    nx2 = int(min(W, x2 + ex))
    ny2 = int(min(H, y2 + ey))
    if nx2 - nx1 <= 0 or ny2 - ny1 <= 0:
        return int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
    return nx1, ny1, nx2, ny2


def _imread(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _imwrite(path, img):
    ext = Path(path).suffix.lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True



def load_faster_rcnn_model(weights_path, device):
    try:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=3)
        state = torch.load(weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(state)
        if missing or unexpected:
            print(f"[WARN] R-CNN load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        model.to(device)
        model.eval()
        print(f"Loaded Faster R-CNN weights from {weights_path}")
        return model
    except Exception as e:
        raise ValueError(f"加载 Faster R-CNN 模型失败: {str(e)}")





def _save_crop_and_record(bgr_img, bbox, label, score, model_type, output_dir,
                          base_name, ext, idx):
    if score is None or score <= 0:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    H, W = bgr_img.shape[:2]
    x1, y1, x2, y2 = _expand_bbox(x1, y1, x2, y2, W, H, expand_ratio=0.08)
    w = x2 - x1; h = y2 - y1
    if w <= 16 or h <= 16:
        print(f"skip {model_type} crop: w={w} h={h} too small")
        return None
    ratio = w / max(1, h); rinv = h / max(1, w)
    if ratio > 50 or rinv > 50:
        print(f"skip {model_type} crop: aspect_ratio={max(ratio, rinv):.2f} >50:1")
        return None
    cropped = bgr_img[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    class_name = CLASS_NAMES.get(int(label), "unknown")
    if class_name == "unknown":
        return None
    label_text = f"{model_type}_{class_name}_{float(score):.2f}"
    out_path = os.path.join(output_dir, base_name, f"{base_name}_{idx}_{label_text}{ext}")
    os.makedirs(os.path.join(output_dir, base_name), exist_ok=True)
    if not _imwrite(out_path, cropped):
        print(f"[WARN] failed to write crop {out_path}")
        return None
    print(f"saved {out_path}")
    return (class_name, float(score), [int(x1), int(y1), int(x2), int(y2)], "Detected")


def _save_fallback(bgr_img, image_path, output_dir):
    name = os.path.basename(image_path)
    base, ext = os.path.splitext(name)
    img_dir = os.path.join(output_dir, base)
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, name)
    _imwrite(out_path, bgr_img)
    print(f"saved fallback (no detection) -> {out_path}")
    return [("unknown", 0.0, None, "No detections")]



def predict_rcnn(model, bgr_img, image_path, output_dir, device,
                 conf_thr, max_crops, nms_iou):
    img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
    tensor = tensor.to(device).float().div_(255.0)
    with torch.no_grad():
        preds = model([tensor])[0]
    boxes = preds["boxes"].detach().cpu().numpy()
    labels = preds["labels"].detach().cpu().numpy().astype(np.int64)
    scores = preds["scores"].detach().cpu().numpy().astype(np.float64)
    print(f"  rcnn raw boxes for {os.path.basename(image_path)}: {len(boxes)}")
    valid_mask = (scores > conf_thr)
    for i in range(len(labels)):
        if int(labels[i]) not in CLASS_NAMES:
            valid_mask[i] = False
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return []
    boxes = boxes[valid_idx]; labels = labels[valid_idx]; scores = scores[valid_idx]
    keep = _class_wise_nms(boxes, labels, scores, iou_thr=nms_iou)
    boxes = boxes[keep]; labels = labels[keep]; scores = scores[keep]
    print(f"  rcnn valid after NMS/conf: {len(boxes)}")
    order = np.argsort(-scores)[:max_crops]
    base, ext = os.path.splitext(os.path.basename(image_path))
    results = []
    for rank, k in enumerate(order, start=1):
        rec = _save_crop_and_record(bgr_img, boxes[k], labels[k], scores[k],
                                    "rcnn", output_dir, base, ext, rank)
        if rec is not None:
            results.append(rec)
    return results





def find_latest_rcnn_best(base="runs/train"):
    if not os.path.isdir(base):
        return None
    candidates = []
    for d in sorted(os.listdir(base)):
        exp_dir = os.path.join(base, d)
        if not (d.startswith("exp") and d[3:].isdigit() and os.path.isdir(exp_dir)):
            continue
        p = os.path.join(exp_dir, "weights", "best.pt")
        if not os.path.isfile(p):
            continue
        candidates.append((int(d[3:]), p))
    if not candidates:
        return None
    device = torch.device("cpu")
    for _, p in sorted(candidates, reverse=True):
        try:
            sd = torch.load(p, map_location=device)
            k = "roi_heads.box_predictor.cls_score.weight"
            if k in sd and sd[k].shape[0] == 3:
                return p
        except Exception:
            pass
    return candidates[-1][1]


def get_next_exp_dir(base_path="runs/predict"):
    os.makedirs(base_path, exist_ok=True)
    nums = []
    for d in os.listdir(base_path):
        if d.startswith("exp") and d[3:].isdigit() and os.path.isdir(os.path.join(base_path, d)):
            nums.append(int(d[3:]))
    n = (max(nums) + 1) if nums else 1
    exp_dir = os.path.join(base_path, f"exp{n}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def predict_folder(rcnn_weights, test_dir, output_dir,
                   conf_thr=0.5, max_crops=6, nms_iou=0.5,
                   clean_output_dir=True,
                   summary_csv=True):
    test_dir = os.path.abspath(test_dir)
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试路径不存在: {test_dir}")
    print(f"[INFO] Test dir: {test_dir}")

    rcnn_weights = os.path.abspath(rcnn_weights)
    if not os.path.isfile(rcnn_weights):
        auto = find_latest_rcnn_best()
        if auto and os.path.isfile(auto):
            print(f"[INFO] 指定的 R-CNN 权重不存在，自动回退到: {auto}")
            rcnn_weights = auto
        else:
            raise FileNotFoundError(f"Faster R-CNN 权重不存在: {rcnn_weights}")


    if clean_output_dir and os.path.isdir(output_dir):
        bak = output_dir + ".bak"
        if os.path.isdir(bak):
            try:
                shutil.rmtree(bak)
            except Exception as e:
                print(f"[WARN] 无法清理旧 bak: {e}")
        try:
            shutil.move(output_dir, bak)
            print(f"[INFO] 旧 crop 目录已备份到 {bak}")
        except Exception as e:
            print(f"[WARN] move output_dir 失败，尝试清空内容: {e}")
            try:
                for name in os.listdir(output_dir):
                    p = os.path.join(output_dir, name)
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
            except Exception as e2:
                raise RuntimeError(f"无法清理输出目录 {output_dir}: {e2}") from e2
    os.makedirs(output_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)
                          if f.lower().endswith(exts)])
    if not image_paths:
        raise ValueError(f"{test_dir} 中没有找到图片")
    print(f"[INFO] Found {len(image_paths)} images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    rcnn_model = load_faster_rcnn_model(rcnn_weights, device)
    exp_dir = get_next_exp_dir()
    print(f"[INFO] Experiment dir: {exp_dir}")

    csv_rcnn = os.path.join(exp_dir, "predictions_rcnn.csv")
    preds_rcnn = []

    for i, ip in enumerate(image_paths, 1):
        name = os.path.basename(ip)
        print(f"[{i}/{len(image_paths)}] {name}")
        try:
            img = _imread(ip)
            if img is None:
                raise ValueError("cv2 无法加载")
            rrec = predict_rcnn(rcnn_model, img, ip, output_dir, device,
                                conf_thr, max_crops, nms_iou)
            if not rrec:
                rrec = _save_fallback(img, ip, output_dir)
            for cn, cf, bb, st in rrec:
                preds_rcnn.append([name, cn, f"{cf:.4f}",
                                   str(bb).replace(" ", "") if bb else "None", st])
                print(f"    rcnn -> {cn} {cf:.4f}")
            del img, rrec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            preds_rcnn.append([name, "error", str(e), "None", "Error"])

    def _write_csv(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Image", "Predicted Class", "Confidence/Error", "Bounding Box", "Status"])
            w.writerows(rows)

    _write_csv(csv_rcnn, preds_rcnn)
    print(f"[DONE] R-CNN: {csv_rcnn}  (rows={len(preds_rcnn)})")

    if summary_csv:
        n_crops = 0
        for d in os.listdir(output_dir):
            sub = os.path.join(output_dir, d)
            if os.path.isdir(sub):
                n_crops += len([x for x in os.listdir(sub) if x.lower().endswith(exts)])
        summary = os.path.join(exp_dir, "summary.txt")
        with open(summary, "w", encoding="utf-8") as f:
            f.write(f"images        = {len(image_paths)}\n")
            f.write(f"rcnn_weights  = {rcnn_weights}\n")
            f.write(f"yolo_weights  = DISABLED (YOLO 分支已注释，只采用 Faster R-CNN)\n")
            f.write(f"conf_thr      = {conf_thr}\n")
            f.write(f"max_crops/img = {max_crops}\n")
            f.write(f"nms_iou       = {nms_iou}\n")
            f.write(f"output_dir    = {output_dir}\n")
            f.write(f"total_crops   = {n_crops}\n")
        print(f"[INFO] total crops written = {n_crops}")
    print(f"[DONE] crops output -> {output_dir}")
    return preds_rcnn




def main():
    parser = argparse.ArgumentParser(description="BoeingFewShot crop 生成 (仅 Faster R-CNN，YOLO 分支已注释禁用)")
    parser.add_argument("--rcnn-weights", type=str, default=None,
                        help="R-CNN 权重路径（默认自动找最新 Boeing exp）")
    parser.add_argument("--test-dir", type=str, default="./datasets/BoeingFewShot/Q/images",
                        help="Query 图目录")
    parser.add_argument("--output-dir", type=str, default="./datasets/BoeingFewShot/crop",
                        help="crop 输出目录")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认 0.5，训练期 crop_purity 0.59 的合理阈值)")
    parser.add_argument("--max-crops", type=int, default=6,
                        help="每张图、每个模型最多保留多少 crop (默认 6)")
    parser.add_argument("--nms-iou", type=float, default=0.5,
                        help="per-class NMS IoU 阈值 (默认 0.5)")
    parser.add_argument("--no-clean", action="store_true",
                        help="不清空旧 crop 目录，在旧目录上继续追加")
    args = parser.parse_args()

    rcnn_weights = args.rcnn_weights
    if rcnn_weights is None:
        auto = find_latest_rcnn_best()
        if auto is None:
            raise SystemExit("[FATAL] 未找到任何 R-CNN 权重 (runs/train/exp*/weights/best.pt, cls_score.out_features=3)")
        rcnn_weights = auto


    predict_folder(
        rcnn_weights=rcnn_weights,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        conf_thr=args.conf,
        max_crops=args.max_crops,
        nms_iou=args.nms_iou,
        clean_output_dir=not args.no_clean,
    )


if __name__ == "__main__":
    main()
