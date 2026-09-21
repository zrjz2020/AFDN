import os
import sys
import csv
import cv2
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import gc
import io
import builtins
import math
import time

if sys.platform.startswith("win"):
    try:
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


BOEING_YAML_PATH = os.environ.get("BOEING_YAML", os.path.join(".", "datasets", "dataBoeingAug.yaml"))
NEU_YAML_PATH = os.environ.get("NEU_YAML", os.path.join(".", "datasets", "dataneuAug.yaml"))


class BoeingDataset(Dataset):
    def __init__(self, img_dir, label_dir, class_names, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.imgs = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        )
        self.class_names = class_names
        self.valid_class_ids = set(int(k) for k in class_names.keys())

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
                                continue
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = (cx - bw / 2) * w
                            y1 = (cy - bh / 2) * h
                            x2 = (cx + bw / 2) * w
                            y2 = (cy + bh / 2) * h
                            x1, y1 = max(0.0, x1), max(0.0, y1)
                            x2, y2 = min(float(w), x2), min(float(h), y2)
                            if x2 - x1 > 1.0 and y2 - y1 > 1.0:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(class_id + 1)
                        except (IndexError, ValueError) as e:
                            print(f"[WARN] Invalid label format in {label_path}: {e}")
            except UnicodeDecodeError as e:
                print(f"[WARN] Decode failed {label_path}: {e}")

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

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
        d for d in os.listdir(base_path)
        if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))
    ]
    exp_nums = [int(d.replace("exp", "")) for d in exp_dirs if d.replace("exp", "").isdigit()]
    next_exp_num = max(exp_nums, default=0) + 1
    return os.path.join(base_path, f"exp{next_exp_num}")


def find_latest_neu_best(base_path=os.path.join("runs", "train"), fallback=None,
                         expected_cls_rows=7, expected_reg_rows=28):
    if not os.path.isdir(base_path):
        pass
    else:
        candidates = []
        for d in os.listdir(base_path):
            if d.startswith("exp") and d.replace("exp", "").isdigit():
                best_path = os.path.join(base_path, d, "weights", "best.pt")
                if not os.path.isfile(best_path):
                    continue
                try:
                    sd = torch.load(best_path, map_location="cpu")
                except Exception:
                    continue
                cls = sd.get("roi_heads.box_predictor.cls_score.weight")
                reg = sd.get("roi_heads.box_predictor.bbox_pred.weight")
                if cls is not None and reg is not None and \
                        cls.dim() >= 2 and cls.size(0) == expected_cls_rows and \
                        reg.dim() >= 2 and reg.size(0) == expected_reg_rows:
                    candidates.append((int(d.replace("exp", "")), best_path))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

    if fallback and os.path.exists(fallback):
        try:
            sd = torch.load(fallback, map_location="cpu")
            cls = sd.get("roi_heads.box_predictor.cls_score.weight")
            reg = sd.get("roi_heads.box_predictor.bbox_pred.weight")
            if cls is not None and reg is not None and \
                    cls.dim() >= 2 and cls.size(0) == expected_cls_rows and \
                    reg.dim() >= 2 and reg.size(0) == expected_reg_rows:
                return fallback
        except Exception:
            pass
    return None


def init_results_csv(exp_dir, class_names=None):
    csv_path = os.path.join(exp_dir, "results.csv")
    header = [
        "epoch", "train_loss",
        "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg",
        "val_loss", "map_50", "map_50_95", "lr",
        "crop_purity", "det_recall",
    ]
    if class_names:
        for name in class_names:
            header.append(f"ap50_{name}")
            header.append(f"rec50_{name}")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
    return csv_path


def append_results(csv_path, row):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, total_epochs,
                    grad_clip=1.0, log_every=10):
    model.train()
    running_loss = 0.0
    running_items = {k: 0.0 for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")}
    window_loss = 0.0
    window_batches = 0
    n_batches = 0
    total_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device).float().div_(255.0) for img in images]
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
    return running_loss / n, {k: v / n for k, v in running_items.items()}


@torch.no_grad()
def compute_val_loss(model, loader, device):
    model.eval()
    was_training = model.training

    bn_states = {}
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_states[m] = (m.momentum, m.training)
            m.momentum = 0.0
    model.train()

    val_loss = 0.0
    n_batches = 0
    for images, targets in loader:
        images = [img.to(device).float().div_(255.0) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast(enabled=True):
            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                continue
            total = sum(loss for loss in loss_dict.values())
        if torch.isfinite(total):
            val_loss += float(total.item())
            n_batches += 1

    for m, (mom, training_flag) in bn_states.items():
        m.momentum = mom
    if not was_training:
        model.eval()
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
        images_dev = [img.to(device).float().div_(255.0) for img in images]
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

    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        a1 = (a[2] - a[0]) * (a[3] - a[1])
        a2 = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(a1 + a2 - inter, 1e-8)

    def _ap_single(thr, detail=None):
        aps = []
        for c in range(1, num_classes):
            if n_pos[c] == 0:
                if detail is not None:
                    detail[c] = {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": 0}
                continue
            dets = sorted(detections[c], key=lambda x: x[0], reverse=True)
            gt_by_img = {}
            for idx, gt in enumerate(gts[c]):
                gt_by_img.setdefault(gt[4], []).append((idx, gt))
            matched = set()
            tp = [0] * len(dets); fp = [0] * len(dets)
            for di, d in enumerate(dets):
                s, x1, y1, x2, y2, iid = d
                best_iou = 0.0; best_gi = -1
                for gi, gt in gt_by_img.get(iid, []):
                    if gi in matched:
                        continue
                    iou = _iou((x1, y1, x2, y2), (gt[0], gt[1], gt[2], gt[3]))
                    if iou > best_iou:
                        best_iou = iou; best_gi = gi
                if best_iou >= thr and best_gi >= 0:
                    tp[di] = 1; matched.add(best_gi)
                else:
                    fp[di] = 1
            if detail is not None:
                tp_c = sum(1 for di in range(len(dets)) if tp[di] and dets[di][0] >= 0.5)
                fp_c = sum(1 for di in range(len(dets)) if fp[di] and dets[di][0] >= 0.5)
                detail[c] = {
                    "ap": 0.0,
                    "precision": tp_c / max(1, tp_c + fp_c),
                    "recall": tp_c / n_pos[c],
                    "n_gt": n_pos[c],
                }
            tp_cum = fp_cum = 0
            prec = []; rec = []
            for di in range(len(dets)):
                tp_cum += tp[di]; fp_cum += fp[di]
                prec.append(tp_cum / max(1, tp_cum + fp_cum))
                rec.append(tp_cum / max(1, n_pos[c]))
            for i in range(len(prec) - 2, -1, -1):
                if prec[i] < prec[i + 1]:
                    prec[i] = prec[i + 1]
            ap = 0.0; prev_r = 0.0
            for i in range(len(prec)):
                if rec[i] != prev_r:
                    ap += prec[i] * (rec[i] - prev_r); prev_r = rec[i]
            if detail is not None:
                detail[c]["ap"] = ap
            aps.append(ap)
        return float(sum(aps) / len(aps)) if aps else 0.0

    def _crop_metrics(iou_thr=0.5, conf_thr=0.5):
        all_dets = []
        for c in range(1, num_classes):
            for (s, x1, y1, x2, y2, iid) in detections[c]:
                if s >= conf_thr:
                    all_dets.append((s, c, x1, y1, x2, y2, iid))
        all_dets.sort(key=lambda x: x[0], reverse=True)
        all_gts = []
        for c in range(1, num_classes):
            for (x1, y1, x2, y2, iid, _used) in gts[c]:
                all_gts.append([x1, y1, x2, y2, iid, c, False])
        gt_by_img = {}
        for g in all_gts:
            gt_by_img.setdefault(g[4], []).append(g)
        n_matched = 0; n_correct_cls = 0
        for (s, c, x1, y1, x2, y2, iid) in all_dets:
            best_iou = 0.0; best = None
            for g in gt_by_img.get(iid, []):
                if g[6]:
                    continue
                iou = _iou((x1, y1, x2, y2), (g[0], g[1], g[2], g[3]))
                if iou > best_iou:
                    best_iou = iou; best = g
            if best is not None and best_iou >= iou_thr:
                best[6] = True
                n_matched += 1
                if best[5] == c:
                    n_correct_cls += 1
        n_gt = len(all_gts)
        n_gt_hit = sum(1 for g in all_gts if g[6])
        return {
            "n_det": len(all_dets),
            "n_gt": n_gt,
            "crop_purity": n_correct_cls / len(all_dets) if all_dets else 0.0,
            "crop_acc_matched": n_correct_cls / n_matched if n_matched else 0.0,
            "det_recall": n_gt_hit / n_gt if n_gt else 0.0,
            "n_missed_gt": n_gt - n_gt_hit,
        }

    detail_50 = {}
    map_50 = _ap_single(0.5, detail=detail_50)
    map_50_95 = float(sum(_ap_single(t) for t in iou_thresholds) / len(iou_thresholds))
    extra = {"per_class": detail_50}
    extra.update(_crop_metrics())
    return map_50, map_50_95, extra


def build_scheduler(optimizer, epochs, warmup_epochs=2):
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
    stage_map = {"conv1": 0, "bn1": 0, "layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4}
    for name, mod in model.backbone.body.named_children():
        sid = stage_map.get(name, 99)
        if sid <= freeze_until_layer:
            for p in mod.parameters():
                p.requires_grad = False
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    n_total = sum(1 for _ in model.parameters())
    print(f"[INFO] Frozen params: {n_frozen}/{n_total} (until layer {freeze_until_layer})")


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def map_pretrained_weights(model, pretrained_weights, neu_class_names, boeing_class_names, device):
    state_dict = torch.load(pretrained_weights, map_location=device)
    model_dict = model.state_dict()

    neu_classes = {name: idx + 1 for idx, name in neu_class_names.items()}
    boeing_classes = {name: idx + 1 for idx, name in boeing_class_names.items()}
    common_classes = set(neu_classes.keys()).intersection(boeing_classes.keys())

    cls_w = "roi_heads.box_predictor.cls_score.weight"
    cls_b = "roi_heads.box_predictor.cls_score.bias"
    if cls_w in state_dict:
        pre_w = state_dict[cls_w]
        pre_b = state_dict[cls_b]
        new_w = torch.empty((len(boeing_class_names) + 1, pre_w.size(1)), dtype=pre_w.dtype)
        new_b = torch.empty(len(boeing_class_names) + 1, dtype=pre_b.dtype)
        nn.init.xavier_uniform_(new_w)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_w.unsqueeze(0) if new_w.dim() == 1 else new_w)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(new_b, -bound, bound)

        new_w[0].copy_(pre_w[0]); new_b[0].copy_(pre_b[0])
        for name, bidx in boeing_classes.items():
            if name in common_classes:
                nidx = neu_classes[name]
                new_w[bidx].copy_(pre_w[nidx]); new_b[bidx].copy_(pre_b[nidx])

        model_dict[cls_w] = new_w; model_dict[cls_b] = new_b

    reg_w = "roi_heads.box_predictor.bbox_pred.weight"
    reg_b = "roi_heads.box_predictor.bbox_pred.bias"
    if reg_w in state_dict:
        pre_w = state_dict[reg_w]
        pre_b = state_dict[reg_b]
        new_w = torch.empty((len(boeing_class_names) * 4 + 4, pre_w.size(1)), dtype=pre_w.dtype)
        new_b = torch.empty(len(boeing_class_names) * 4 + 4, dtype=pre_b.dtype)
        nn.init.xavier_uniform_(new_w)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_w.unsqueeze(0) if new_w.dim() == 1 else new_w)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(new_b, -bound, bound)

        new_w[0:4].copy_(pre_w[0:4]); new_b[0:4].copy_(pre_b[0:4])
        for name, bidx in boeing_classes.items():
            if name in common_classes:
                nidx = neu_classes[name]
                ps, ns = nidx * 4, bidx * 4
                new_w[ns:ns + 4].copy_(pre_w[ps:ps + 4]); new_b[ns:ns + 4].copy_(pre_b[ps:ps + 4])

        model_dict[reg_w] = new_w; model_dict[reg_b] = new_b

    common_updated_keys = set(k for k in [cls_w, cls_b, reg_w, reg_b] if k in state_dict)
    for k, v in state_dict.items():
        if k in common_updated_keys:
            continue
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v.clone()

    model.load_state_dict(model_dict, strict=False)
    print(f"[INFO] Loaded Neu pretrained & mapped common classes: {pretrained_weights}")
    if common_classes:
        print(f"[INFO] Preserved common classes: {', '.join(common_classes)}")
    else:
        print(f"[WARN] No common classes between Neu and Boeing; ROI head xavier init")
    return model


def train_model(data_dir=None, epochs=100, batch_size=6, unfreeze_epoch=6,
                pretrained_weights_default=os.path.join("runs", "train", "exp2", "weights", "best.pt"),
                init_from=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(BOEING_YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(NEU_YAML_PATH, "r", encoding="utf-8") as f:
        neu_cfg = yaml.safe_load(f)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Boeing yaml: {BOEING_YAML_PATH}  classes: {cfg['names']}")
    print(f"[INFO] Neu yaml:    {NEU_YAML_PATH}  classes: {neu_cfg['names']}")
    common = set(neu_cfg["names"].values()).intersection(set(cfg["names"].values()))
    print(f"[INFO] Common classes for transfer: {common or '(none)'}")

    if data_dir is None:
        data_dir = cfg["path"]

    train_img_dir = os.path.join(data_dir, cfg["train"])
    train_label_dir = os.path.join(data_dir, cfg["train"].replace("images", "labels"))
    val_img_dir = os.path.join(data_dir, cfg["val"])
    val_label_dir = os.path.join(data_dir, cfg["val"].replace("images", "labels"))
    for d in (train_img_dir, train_label_dir, val_img_dir, val_label_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Data dir not found: {d}")

    train_dataset = BoeingDataset(train_img_dir, train_label_dir, cfg["names"])
    val_dataset = BoeingDataset(val_img_dir, val_label_dir, cfg["names"])
    has_cuda = torch.cuda.is_available()
    _force_workers = os.environ.get("BOEING_NUM_WORKERS")
    if _force_workers is not None:
        workers = int(_force_workers)
    else:
        workers = 0 if not has_cuda else min(2, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=False, collate_fn=detection_collate,
        prefetch_factor=1 if workers > 0 else None,
    )
    for k, v in cfg["names"].items():
        class_names[int(k)] = str(v)
    csv_path = init_results_csv(exp_dir, class_names=class_names)
    print(f"[INFO] Experiment dir: {exp_dir}")

    best_map50 = -1.0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        _ep_t0 = time.time()
        if epoch == unfreeze_epoch:
            unfreeze_all(model)
            current_lr = scheduler.get_last_lr()[0]
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(trainable, lr=current_lr, momentum=0.9, weight_decay=0.0005)
            remaining = max(1, epochs - epoch - 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6
            )
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            print(f"[INFO] Epoch {epoch + 1}: unfreeze ALL params (frozen={n_frozen}), lr={current_lr:.6f}")

        try:
            train_loss, loss_items = train_one_epoch(
                model, train_loader, optimizer, scaler, device, epoch, epochs, grad_clip=1.0, log_every=10,
            )
        except (OSError, RuntimeError) as spawn_err:
            msg = str(spawn_err)
            if "WinError 5" in msg or "拒绝访问" in msg or "access is denied" in msg.lower():
                print(f"[WARN] Train DataLoader worker spawn failed ({msg}); fallback to num_workers=0 and continue.")
                workers = 0
                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=0, pin_memory=has_cuda, collate_fn=detection_collate,
                )
                train_loss, loss_items = train_one_epoch(
                    model, train_loader, optimizer, scaler, device, epoch, epochs, grad_clip=1.0, log_every=10,
                )
            else:
                raise
        val_loss = compute_val_loss(model, val_loader, device)
        map_50, map_50_95, det_extra = compute_map(model, val_loader, device, num_classes=num_classes)

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
            f"{det_extra['crop_purity']:.4f}",
            f"{det_extra['det_recall']:.4f}",
        ]
        for c in range(1, num_classes):
            d = det_extra["per_class"].get(c, {"ap": 0.0, "recall": 0.0})
            row.append(f"{d['ap']:.4f}")
            row.append(f"{d['recall']:.4f}")
        append_results(csv_path, row)

        for c in range(1, num_classes):
            d = det_extra["per_class"].get(c, {"ap": 0.0, "precision": 0.0, "recall": 0.0, "n_gt": 0})
            print(f"  class[{class_names[c - 1]}]  AP@50={d['ap']:.4f}  "
                  f"P@0.5={d['precision']:.4f}  R@0.5={d['recall']:.4f}  (n_gt={d['n_gt']})")
        print(f"  crop-level: purity={det_extra['crop_purity']:.4f} "
              f"(crops w/ correct class / all crops)  det_recall={det_extra['det_recall']:.4f} "
              f"(missed GT={det_extra['n_missed_gt']}/{det_extra['n_gt']})  "
              f"acc_matched={det_extra['crop_acc_matched']:.4f} (n_det={det_extra['n_det']})")

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

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss: {val_loss:.4f}  "
            f"mAP@50: {map_50:.4f}  "
            f"mAP@50:95: {map_50_95:.4f}  "
            f"LR: {current_lr:.6f}"
            f"  [{time.time() - _ep_t0:.1f}s/epoch]" + save_msg
        )

    print(f"[DONE] Best mAP@50 = {best_map50:.4f}")
    print(f"[DONE] Best model: {os.path.join(weights_dir, 'best.pt')}")
    print(f"[INFO] Training results saved to {exp_dir}")
    return exp_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Boeing Faster R-CNN 训练")
    parser.add_argument("--epochs", type=int, default=100, help="训练总轮数 (默认 100)")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="batch size (默认 6；Boeing 大图 batch=8 显存峰值 ~10.2/12GB 会触发 WDDM 溢写到系统 RAM)")
    parser.add_argument("--unfreeze-epoch", type=int, default=6,
                        help="在第几个 epoch 解冻 backbone 全部参数 (默认 6)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="覆盖 YAML 中 path 的数据集根目录")
    parser.add_argument("--pretrained", type=str,
                        default=os.path.join("runs", "train", "exp2", "weights", "best.pt"),
                        help="Neu 预训练权重 fallback 路径（默认 runs/train/exp2/weights/best.pt）")
    parser.add_argument("--init-from", type=str, default=None,
                        help="以指定 Boeing checkpoint（同 3 类架构）做完整初始化，跳过 COCO+Neu 迁移")
    args = parser.parse_args()

    try:
        print(f"[CLI] epochs={args.epochs}  batch_size={args.batch_size}  "
              f"unfreeze_epoch={args.unfreeze_epoch}"
              + (f"  init_from={args.init_from}" if args.init_from else ""))
        exp_dir = train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            unfreeze_epoch=args.unfreeze_epoch,
            pretrained_weights_default=args.pretrained,
            init_from=args.init_from,
        )
        print(f"[INFO] Final exp dir: {exp_dir}")
    except Exception as e:
        print(f"[FATAL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
