import os
import sys
import csv
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import gc
import io
import builtins

if sys.platform.startswith("win"):
    try:
        raw_out = sys.stdout
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except Exception:
        pass

_orig_print = builtins.print
def _print_flush(*args, **kwargs):
    kwargs.setdefault("flush", True)
    return _orig_print(*args, **kwargs)
builtins.print = _print_flush

YAML_PATH = os.environ.get("NEU_YAML", os.path.join(".", "datasets", "dataneuAug.yaml"))
with open(YAML_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


class NeuDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        )
        self.class_names = class_names
        self.valid_class_ids = set(int(k) for k in class_names.keys())
        self._class_stats = {}

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        boxes = []
        labels = []
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            class_id = int(float(parts[0]))
                            if class_id not in self.valid_class_ids:
                                self._class_stats[class_id] = self._class_stats.get(class_id, 0) + 1
                                continue
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            x1, y1 = max(0.0, x1), max(0.0, y1)
                            x2, y2 = min(float(w), x2), min(float(h), y2)
                            if x2 - x1 > 1 and y2 - y1 > 1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(class_id + 1)
                        except (IndexError, ValueError) as e:
                            print(f"[WARN] Invalid label format in {label_path}: {e}")
            except UnicodeDecodeError as e:
                print(f"[WARN] Failed to decode {label_path} with UTF-8: {e}")

        img_tensor = transforms.ToTensor()(img)
        if self.transform:
            img_tensor = self.transform(img_tensor)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return img_tensor, target


def detection_collate(batch):
    return tuple(zip(*batch))


def get_next_exp_dir(base_path=os.path.join("runs", "train")):
    os.makedirs(base_path, exist_ok=True)
    exp_dirs = [
        d
        for d in os.listdir(base_path)
        if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))
    ]
    exp_nums = [int(d.replace("exp", "")) for d in exp_dirs if d.replace("exp", "").isdigit()]
    next_exp_num = max(exp_nums, default=0) + 1
    return os.path.join(base_path, f"exp{next_exp_num}")


def init_results_csv(exp_dir):
    csv_path = os.path.join(exp_dir, "results.csv")
    header = [
        "epoch",
        "train_loss",
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "val_loss",
        "map_50",
        "map_50_95",
        "lr",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return csv_path


def append_results(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs, grad_clip=1.0, log_every=50):
    model.train()
    running_loss = 0.0
    running_items = {k: 0.0 for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")}
    window_loss = 0.0
    window_batches = 0
    n_batches = 0
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=scaler.is_enabled()):
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                print(f"[ERROR] loss_dict is list at epoch {epoch + 1} batch {batch_idx}, skip")
                continue
            total_loss = sum(loss for loss in loss_dict.values())
            if not torch.isfinite(total_loss):
                print(f"[WARN] Non-finite loss at epoch {epoch + 1} batch {batch_idx}: {total_loss.item()}, skip")
                continue

        scaler.scale(total_loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        lv = float(total_loss.item())
        running_loss += lv
        window_loss += lv
        for k in running_items:
            running_items[k] += float(loss_dict.get(k, torch.tensor(0.0)).item())
        n_batches += 1
        window_batches += 1

        if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == total_batches:
            avg_w = window_loss / max(1, window_batches)
            lr_now = float(optimizer.param_groups[0]["lr"])
            print(
                f"  batch {batch_idx + 1}/{total_batches}  "
                f"window_loss_last_{log_every}={avg_w:.4f}  "
                f"lr={lr_now:.6f}"
            )
            window_loss = 0.0
            window_batches = 0

    n = max(1, n_batches)
    avg_total = running_loss / n
    avg_items = {k: v / n for k, v in running_items.items()}
    return avg_total, avg_items


@torch.no_grad()
def compute_val_loss(model, loader, device):
    model.train()
    val_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast(enabled=True):
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                continue
            total = sum(loss for loss in loss_dict.values())
        if torch.isfinite(total):
            val_loss += float(total.item())
            n_batches += 1
    return val_loss / max(1, n_batches)


@torch.no_grad()
def compute_map(model, loader, device, num_classes, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    model.eval()

    detections = {c: [] for c in range(1, num_classes)}
    gts = {c: [] for c in range(1, num_classes)}
    n_pos = {c: 0 for c in range(1, num_classes)}

    img_id = 0
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        for b_idx, out in enumerate(outputs):
            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for b, s, l in zip(boxes, scores, labels):
                l = int(l)
                if 1 <= l < num_classes:
                    detections[l].append((float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]), img_id))
            t = targets[b_idx]
            gt_boxes = t["boxes"].cpu().numpy()
            gt_labels = t["labels"].cpu().numpy()
            for b, l in zip(gt_boxes, gt_labels):
                l = int(l)
                if 1 <= l < num_classes:
                    gts[l].append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), img_id, False))
                    n_pos[l] += 1
            img_id += 1

    def _iou(boxA, boxB):
        x1 = max(boxA[0], boxB[0]); y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2]); y2 = min(boxA[3], boxB[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a1 = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        a2 = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = a1 + a2 - inter
        return inter / max(union, 1e-8)

    def _ap_single_iou(thr):
        aps = []
        for c in range(1, num_classes):
            if n_pos[c] == 0:
                continue
            dets = sorted(detections[c], key=lambda x: x[0], reverse=True)
            gt_by_img = {}
            for idx, gt in enumerate(gts[c]):
                gt_by_img.setdefault(gt[4], []).append((idx, gt))

            tp = [0] * len(dets)
            fp = [0] * len(dets)
            matched = set()

            for di, d in enumerate(dets):
                s, x1, y1, x2, y2, iid = d
                best_iou = 0.0
                best_gi = -1
                for gi, gt in gt_by_img.get(iid, []):
                    if gi in matched:
                        continue
                    iou = _iou((x1, y1, x2, y2), (gt[0], gt[1], gt[2], gt[3]))
                    if iou > best_iou:
                        best_iou = iou
                        best_gi = gi
                if best_iou >= thr and best_gi >= 0:
                    tp[di] = 1
                    matched.add(best_gi)
                else:
                    fp[di] = 1

            tp_cum = 0; fp_cum = 0
            precisions = []; recalls = []
            for di in range(len(dets)):
                tp_cum += tp[di]
                fp_cum += fp[di]
                prec = tp_cum / max(1, tp_cum + fp_cum)
                rec = tp_cum / max(1, n_pos[c])
                precisions.append(prec)
                recalls.append(rec)

            M = len(precisions)
            for i in range(M - 2, -1, -1):
                if precisions[i] < precisions[i + 1]:
                    precisions[i] = precisions[i + 1]
            ap = 0.0
            prev_rec = 0.0
            for i in range(M):
                if recalls[i] != prev_rec:
                    ap += precisions[i] * (recalls[i] - prev_rec)
                    prev_rec = recalls[i]
            aps.append(ap)
        return float(sum(aps) / len(aps)) if aps else 0.0

    map_50 = _ap_single_iou(0.5)
    map_all = [_ap_single_iou(t) for t in iou_thresholds]
    map_50_95 = float(sum(map_all) / len(iou_thresholds))
    return map_50, map_50_95


def build_scheduler(optimizer, epochs, warmup_epochs=3):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cos], milestones=[warmup_epochs]
    )


def freeze_backbone(model, freeze_until_layer=3):
    backbone = model.backbone.body
    stage_map = {"conv1": 0, "bn1": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
    for name, mod in backbone.named_children():
        stage_id = stage_map.get(name, 99)
        if stage_id <= freeze_until_layer:
            for p in mod.parameters():
                p.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    print(f"[INFO] Frozen params: {n_frozen}/{n_total} (until layer {freeze_until_layer})")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def train_model(data_dir=None, epochs=40, batch_size=8, unfreeze_epoch=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] NEU yaml: {YAML_PATH}")
    print(f"[INFO] Class names: {cfg['names']}")

    if data_dir is None:
        data_dir = cfg["path"]

    train_img_dir = os.path.join(data_dir, cfg["train"])
    train_label_dir = os.path.join(data_dir, cfg["train"].replace("images", "labels"))
    val_img_dir = os.path.join(data_dir, cfg["val"])
    val_label_dir = os.path.join(data_dir, cfg["val"].replace("images", "labels"))
    for d in (train_img_dir, train_label_dir, val_img_dir, val_label_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Data dir not found: {d}")

    train_dataset = NeuDataset(train_img_dir, train_label_dir, cfg["names"])
    val_dataset = NeuDataset(val_img_dir, val_label_dir, cfg["names"])
    has_cuda = torch.cuda.is_available()
    workers = 0 if not has_cuda else min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=has_cuda,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=has_cuda,
        collate_fn=detection_collate,
    )
    print(f"[INFO] Train imgs: {len(train_dataset)}, Val imgs: {len(val_dataset)}")
    print(f"[INFO] Workers: {workers}, batch: {batch_size}")

    num_classes = len(cfg["names"]) + 1
    print(f"[INFO] Building Faster R-CNN (num_classes={num_classes}) with ImageNet/COCO pretrained weights...")
    pretrained = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    pretrained_state = pretrained.state_dict()
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    own_state = model.state_dict()
    matched = 0; skipped = 0
    for k, v in pretrained_state.items():
        if k in own_state and own_state[k].shape == v.shape:
            own_state[k] = v.clone()
            matched += 1
        else:
            skipped += 1
    model.load_state_dict(own_state)
    print(f"[INFO] Loaded pretrained: matched {matched} keys, skipped {skipped} keys (ROI head mismatch expected)")
    del pretrained, pretrained_state
    model.to(device)

    freeze_backbone(model, freeze_until_layer=3)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = build_scheduler(optimizer, epochs=epochs, warmup_epochs=3)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    exp_dir = get_next_exp_dir()
    weights_dir = os.path.join(exp_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    csv_path = init_results_csv(exp_dir)
    print(f"[INFO] Experiment dir: {exp_dir}")

    best_map50 = -1.0
    best_val_loss = float("inf")
    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            unfreeze_all(model)
            current_lr = scheduler.get_last_lr()[0]
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(trainable, lr=current_lr, momentum=0.9, weight_decay=0.0005)
            remaining = max(1, epochs - epoch - 1)
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining, eta_min=1e-6)
            scheduler = cos
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            print(f"[INFO] Epoch {epoch + 1}: unfreeze ALL params (frozen={n_frozen}), lr={current_lr:.6f}")

        train_loss, loss_items = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, epochs, grad_clip=1.0
        )
        val_loss = compute_val_loss(model, val_loader, device)
        map_50, map_50_95 = compute_map(model, val_loader, device, num_classes=num_classes)

        current_lr = scheduler.get_last_lr()[0]
        row = [
            epoch + 1,
            f"{train_loss:.4f}",
            f"{loss_items['loss_classifier']:.4f}",
            f"{loss_items['loss_box_reg']:.4f}",
            f"{loss_items['loss_objectness']:.4f}",
            f"{loss_items['loss_rpn_box_reg']:.4f}",
            f"{val_loss:.4f}",
            f"{map_50:.4f}",
            f"{map_50_95:.4f}",
            f"{current_lr:.6f}",
        ]
        append_results(csv_path, row)

        save_msg = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_val_loss.pt"))
        if map_50 > best_map50:
            best_map50 = map_50
            torch.save(model.state_dict(), os.path.join(weights_dir, "best.pt"))
            save_msg = "  [saved best.pt by mAP@50]"

        torch.save(model.state_dict(), os.path.join(weights_dir, "last.pt"))

        scheduler.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"mAP@50: {map_50:.4f}  "
            f"mAP@50:95: {map_50_95:.4f}  "
            f"LR: {current_lr:.6f}" + save_msg
        )

    print(f"[DONE] Best mAP@50 = {best_map50:.4f}")
    print(f"[DONE] Best model: {os.path.join(weights_dir, 'best.pt')}")
    return exp_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NEU Faster R-CNN 训练")
    parser.add_argument("--epochs", type=int, default=40, help="训练总轮数 (默认 40)")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size (默认 8)")
    parser.add_argument("--unfreeze-epoch", type=int, default=10,
                        help="在第几个 epoch 解冻 backbone 全部参数 (默认 10)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="覆盖 YAML 中 path 的数据集根目录")
    args = parser.parse_args()

    try:
        print(f"[CLI] epochs={args.epochs}  batch_size={args.batch_size}  "
              f"unfreeze_epoch={args.unfreeze_epoch}")
        exp_dir = train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unfreeze_epoch=args.unfreeze_epoch,
        )
        print(f"[INFO] Training results saved to {exp_dir}")
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
