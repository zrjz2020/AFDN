import os
from ultralytics import YOLO
import torch
import gc

if __name__ == '__main__':
    # Path to the last trained model
    last_model_path = r"your path\best.pt"

    # Check if last trained model exists, otherwise use yolov8n.pt
    model_path = last_model_path if os.path.exists(last_model_path) else "./yolov8n.pt"

    # Initialize model
    yolo = YOLO(model_path, task="detect")

    # Train the model
    yolo.train(
        data='./dataneuAug.yaml',
        epochs=200,
        imgsz=200,
        batch=256,
        device=0
    )
    print('训练完毕')

    # Clean up
    torch.cuda.empty_cache()  # 释放显存
    gc.collect()  # 强制垃圾回收