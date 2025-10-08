import os
import cv2
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from ultralytics import YOLO
import csv
import numpy as np

class_names = {1: "scratches", 2: "stain"}


def load_faster_rcnn_model(weights_path, device):
    try:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=3)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Loaded Faster R-CNN weights from {weights_path}")
        return model
    except Exception as e:
        raise ValueError(f"加载 Faster R-CNN 模型失败: {str(e)}")


def load_yolo_model(weights_path):
    try:
        model = YOLO(weights_path)
        print(f"Loaded YOLO weights from {weights_path}")
        return model
    except Exception as e:
        raise ValueError(f"加载 YOLO 模型失败: {str(e)}")


def predict_faster_rcnn(model, image_path, device, max_crops=4, threshold=0.2):
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if img_rgb is None:
        raise ValueError(f"无法加载图像: {image_path}")

    img_tensor = transforms.ToTensor()(img_rgb).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    bboxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    print(f"Faster R-CNN raw boxes for {os.path.basename(image_path)}: {len(bboxes)}")

    valid_indices = [i for i, score in enumerate(scores) if score > threshold and labels[i] in class_names]
    print(f"Number of valid boxes before limiting (rcnn): {len(valid_indices)}")
    if len(valid_indices) > max_crops:
        sorted_indices = np.argsort([scores[i] for i in valid_indices])[-max_crops:]
        valid_indices = [valid_indices[j] for j in sorted_indices]
        print(f"Number of boxes after limiting (rcnn): {len(valid_indices)}")

    results = []
    for i in valid_indices:
        bbox = bboxes[i].tolist()
        label = labels[i]
        score = scores[i]
        results.append((class_names[label], score, bbox, "Detected"))

    if not results:
        results = [("unknown", 0.0, None, "No detections")]

    return results


def predict_yolo(model, image_path, max_crops=4):
    try:
        results = model.predict(image_path, imgsz=320)
        predictions = []
        bboxes = []
        labels = []
        confidences = []
        for result in results:
            if result.boxes:
                print(f"YOLO raw boxes for {os.path.basename(image_path)}: {len(result.boxes)}")
                for box in result.boxes:
                    confidence = box.conf.item()
                    if confidence < 0.1:
                        continue
                    class_id = int(box.cls.item()) + 1  # YOLO 标签从 0 开始，调整为从 1 开始以匹配 Faster R-CNN
                    class_name = class_names.get(class_id, "unknown")
                    # 跳过未知类别的预测
                    if class_name == "unknown":
                        continue
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
                    bboxes.append(bbox)
                    labels.append(class_id)
                    confidences.append(confidence)
                    predictions.append((class_name, confidence, bbox, "Detected"))

        valid_indices = list(range(len(confidences)))  # 已经过滤 >0.1
        print(f"Number of valid boxes before limiting (yolo): {len(valid_indices)}")
        if len(valid_indices) > max_crops:
            sorted_indices = np.argsort(confidences)[-max_crops:]
            valid_indices = [valid_indices[j] for j in sorted_indices]
            print(f"Number of boxes after limiting (yolo): {len(valid_indices)}")

        yolo_results = []
        for i in valid_indices:
            bbox = bboxes[i]
            label = labels[i]
            score = confidences[i]
            yolo_results.append((class_names.get(label, 'unknown'), score, bbox, "Detected"))

        if not yolo_results:
            yolo_results = [("unknown", 0.0, None, "No detections")]

        return yolo_results
    except Exception as e:
        raise ValueError(f"YOLO 预测失败: {str(e)}")


def save_crops(img, img_output_dir, base_name, ext, predictions_info, model_type, start_num=1):
    saved_count = 0
    for idx, (class_name, score, bbox, status) in enumerate(predictions_info):
        if status == "Detected" and bbox is not None:
            x_min, y_min, x_max, y_max = map(int, bbox)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img.shape[1], x_max)
            y_max = min(img.shape[0], y_max)
            width = x_max - x_min
            height = y_max - y_min
            if width > 0 and height > 0:
                cropped_img = img[y_min:y_max, x_min:x_max]
                if cropped_img.size > 0:
                    label_text = f"{model_type}_{class_name}_{score:.2f}"
                    crop_num = start_num + idx
                    output_path = os.path.join(img_output_dir, f"{base_name}_{crop_num}_{label_text}{ext}")
                    cv2.imwrite(output_path, cropped_img)
                    print(f"保存裁切图像到 {output_path}")
                    saved_count += 1
    return saved_count


def get_next_exp_dir(base_path="runs/predict"):
    os.makedirs(base_path, exist_ok=True)
    exp_dirs = [d for d in os.listdir(base_path) if d.startswith("exp") and os.path.isdir(os.path.join(base_path, d))]
    exp_nums = [int(d.replace("exp", "")) for d in exp_dirs if d.replace("exp", "").isdigit()]
    next_exp_num = max(exp_nums, default=0) + 1
    exp_dir = os.path.join(base_path, f"exp{next_exp_num}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def predict_folder(yolo_weights, rcnn_weights, test_dir="./data/BoeingFewShot/Q/images",
                   output_dir=r"./data/BoeingFewShot/crop"):

    test_dir = os.path.abspath(test_dir)  # 转换为绝对路径
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试路径 {test_dir} 不存在")
    print(f"Loaded test directory: {test_dir}")


    yolo_weights = os.path.abspath(yolo_weights)
    rcnn_weights = os.path.abspath(rcnn_weights)
    if not os.path.exists(yolo_weights):
        raise FileNotFoundError(f"YOLO 权重文件不存在: {yolo_weights}")
    if not os.path.exists(rcnn_weights):
        raise FileNotFoundError(f"Faster R-CNN 权重文件不存在: {rcnn_weights}")


    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if
                   f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_paths:
        raise ValueError(f"文件夹 {test_dir} 中没有找到 .jpg 或 .png 图像文件")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = load_yolo_model(yolo_weights)
    rcnn_model = load_faster_rcnn_model(rcnn_weights, device)

    exp_dir = get_next_exp_dir()
    output_csv_yolo = os.path.join(exp_dir, "predictions_yolo.csv")
    output_csv_rcnn = os.path.join(exp_dir, "predictions_rcnn.csv")

    predictions_yolo = []
    predictions_rcnn = []

    for image_path in image_paths:
        img_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(img_name)
        img_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(img_output_dir, exist_ok=True)
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法加载图像 {image_path}")
            continue

        try:
            current_count = 0

            yolo_results = predict_yolo(yolo_model, image_path, max_crops=4)

            yolo_saved = save_crops(img, img_output_dir, base_name, ext, yolo_results, 'yolo', start_num=1)
            current_count += yolo_saved

            for class_name, confidence, bbox, status in yolo_results:
                bbox_str = str(bbox) if bbox else "None"
                predictions_yolo.append([img_name, class_name, f"{confidence:.4f}", bbox_str, status])
                print(f"YOLO - 图像: {img_name}, 预测: {class_name}, 置信度: {confidence:.4f}, 边界框: {bbox_str}, 状态: {status}")

            if current_count < 4:
                remaining_crops = 4 - current_count
                rcnn_start_num = current_count + 1
                rcnn_results = predict_faster_rcnn(rcnn_model, image_path, device, max_crops=remaining_crops, threshold=0.1)

                rcnn_saved = save_crops(img, img_output_dir, base_name, ext, rcnn_results, 'rcnn', start_num=rcnn_start_num)
                current_count += rcnn_saved

                for class_name, confidence, bbox, status in rcnn_results:
                    bbox_str = str(bbox) if bbox else "None"
                    predictions_rcnn.append([img_name, class_name, f"{confidence:.4f}", bbox_str, status])
                    print(f"Faster R-CNN - 图像: {img_name}, 预测: {class_name}, 置信度: {confidence:.4f}, 边界框: {bbox_str}, 状态: {status}")
            else:
                predictions_rcnn.append([img_name, "skipped", "N/A", "None", "Skipped (YOLO sufficient)"])

            if current_count == 0:
                original_path = os.path.join(img_output_dir, f"{base_name}_original{ext}")
                cv2.imwrite(original_path, img)
                print(f"保存原图到 {original_path} (文件夹为空)")

        except Exception as e:
            print(f"预测图像 {image_path} 失败: {str(e)}")
            predictions_yolo.append([img_name, "error", str(e), "None", "Error"])
            predictions_rcnn.append([img_name, "error", str(e), "None", "Error"])

    with open(output_csv_yolo, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted Class", "Confidence/Error", "Bounding Box", "Status"])
        writer.writerows(predictions_yolo)

    with open(output_csv_rcnn, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Predicted Class", "Confidence/Error", "Bounding Box", "Status"])
        writer.writerows(predictions_rcnn)

    print(f"YOLO 预测完成，结果已保存到 {output_csv_yolo}")
    print(f"Faster R-CNN 预测完成，结果已保存到 {output_csv_rcnn}")
    print(f"裁切图像已保存到 {output_dir}")
    return predictions_yolo, predictions_rcnn


if __name__ == "__main__":
    yolo_weights = "your path"
    rcnn_weights = "your path"
    test_dir = "your path"
    output_dir = "your path"
    try:
        predictions_yolo, predictions_rcnn = predict_folder(yolo_weights, rcnn_weights, test_dir, output_dir)
    except Exception as e:
        print(f"预测失败: {str(e)}")
