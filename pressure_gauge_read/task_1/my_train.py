import torch

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('my_yolov8x-pose.yaml')  # .load('yolov8n-pose.pt')  # 从YAML构建并传输权重

    # 训练模型
    results = model.train(data="my_data.yaml", epochs=100, imgsz=640,
                          # device=torch.device("cpu"),
                          batch=3)
    # 模型验证
    model.val()


