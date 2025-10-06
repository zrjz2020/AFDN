from ultralytics import YOLO
import torch
import gc
import os

if __name__ == '__main__':
    last_model_path = r"your path/best.pt"
    model_path = last_model_path if os.path.exists(last_model_path) else "./yolov8n.pt"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")

    yolo = YOLO(model_path, task="detect")
    yolo.train(
        data='./datasets/dataBoeingAug.yaml',
        epochs=800,
        imgsz=300,
        batch=174,
        device=0
    )
    print('训练完毕')

    torch.cuda.empty_cache()
    gc.collect()