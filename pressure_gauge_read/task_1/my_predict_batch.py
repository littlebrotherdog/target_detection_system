import os
import sys

from ultralytics import YOLO

args = {
    "images_folder_path": "../DATASETS/images/test/",
    "model_path": "runs/pose/train4/weights/best.pt"
}


# 创建图片路径列表
image_path_list = []
if os.path.exists(args['images_folder_path']):
    for parent_folder, _, file_names in os.walk(args['images_folder_path']):
        for file_name in file_names:
            if file_name.endswith(".jpg"):
                image_path_list.append(parent_folder+file_name)
if not image_path_list:
    sys.exit(1)

# 加载模型
model = YOLO(args['model_path'])  # 预训练的 YOLOv8n 模型
# 在图片列表上运行批量推理
results = model(image_path_list, save=True, imgsz=640)  # 返回 Results 对象列表
