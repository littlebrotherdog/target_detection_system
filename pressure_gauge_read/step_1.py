import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from paddle_ocr.img_ocr import ocr_img

curr_dir = Path(os.path.abspath(__file__)).parent
# 模型1（识别出图片中的指针）的存放路径：
model_1_path = os.path.join(curr_dir, "datas/models/task1_best.pt")
# 模型2（识别出图片中的数标）的存放路径
model_2_path = os.path.join(curr_dir, "datas/models/task2_best.pt")
# 加载模型
task_1_model = YOLO(model_1_path)  # 预训练的 YOLOv8n 模型
task_1_objs_labels = task_1_model.names
task_2_model = YOLO(model_2_path)  # 预训练的 YOLOv8n 模型
task_2_objs_labels = task_2_model.names

# 忽略代码中所有 print 语句，可删除
def print(*args, **kwargs):
    pass


def run(image_path, save_path="./run/main_1/", save_json_result=False):
    image_name, _ = str(image_path.split("/")[-1]).split(".")
    # print(image_name)
    save_path = os.path.join(save_path, image_name)
    # print(save_path)
    # 结果保存位置
    if save_json_result:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path + "/ocr", exist_ok=True)
        os.makedirs(save_path + "/un_ocr", exist_ok=True)
    # 在图片列表上运行批量推理
    with torch.no_grad():
        # 图片中指针检测结果
        task_1_results = task_1_model(source=[image_path], save=False, imgsz=640, iou=0)  # 返回 Results 对象列表
        # 图片中数标检测结果
        task_2_results = task_2_model(source=[image_path], save=False, imgsz=640, iou=0)  # 返回 Results 对象列表

    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 先处理数标的检测结果
    for result in task_2_results:
        boxes = result.boxes.cpu().numpy()
        number_result = {}
        # 遍历每个框
        delete_list = []

        for index, box in enumerate(boxes.data):
            l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom（分别是框的左上角和右下角的xy坐标值）
            conf, class_id = box[4:]  # confidence, class
            # 将识别出的每一个数字图片截取出来，后续进行ocr识别
            frame_ocr = frame[t:b, l:r]
            ocr_txt, ocr_score = ocr_img(frame_ocr)
            # 如果OCR没有检测结果
            if ocr_txt is None:
                delete_list.append(index)
                cv2.imwrite(f"{save_path}/un_ocr/{index}.jpg", frame_ocr)
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 3)
                continue
            if save_json_result:
                cv2.imwrite(f"{save_path}/ocr/{index}_ocr_{ocr_txt}.jpg", frame_ocr)
            res_dict = {
                "index": index,
                "ocr_txt": ocr_txt,
                "ocr_score": ocr_score,
                "xy": (int(l), int(t), int(r), int(b))
            }
            number_result[index] = res_dict

            if save_json_result:
                # 绘制框
                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 3)
                # 绘制类别+置信度（格式：98.1%）
                cv2.putText(frame, f"{ocr_txt} {ocr_score * 100:.1f}% {index} ",
                            (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

        # 遍历 keypoints
        keypoints = result.keypoints  # Keypoints object for pose outputs
        keypoints = keypoints.cpu().numpy()  # convert to numpy array
        # print(keypoints.data)
        # draw keypoints, set first keypoint is red, second is blue
        for index, keypoint in enumerate(keypoints.data):
            if index in delete_list:
                continue

            for i in range(len(keypoint)):
                # print(i, keypoint[i])
                x, y, _ = keypoint[i]
                x, y = int(x), int(y)
                # print(x, y)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(frame, f"{index}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                number_result[index]['keypoint'] = (x, y)
            if len(keypoint) >= 2:
                # draw arrow line from tail to half between head and tail
                x0, y0, _ = keypoint[0]
                x1, y1, _ = keypoint[1]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 4)
        print(number_result)

    for result in task_1_results:
        pointer_result = {}
        boxes = result.boxes  # Boxes object for bbox outputs
        boxes = boxes.cpu().numpy()  # convert to numpy array
        if len(boxes) == 0:
            return None
        index, box = max(enumerate(boxes.data), key=lambda x: x[1][4])
        # print(box)
        # 遍历每个框
        # for box in boxes.data:
        l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom
        conf, id = box[4:]  # confidence, class
        id = int(id)
        # 绘制框
        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 3)
        # 绘制类别+置信度（格式：98.1%）
        cv2.putText(frame, f"{task_1_objs_labels[id]} {conf * 100:.1f}%", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        # 遍历 keypoints
        keypoints = result.keypoints  # Keypoints object for pose outputs
        keypoints = keypoints.cpu().numpy()  # convert to numpy array
        # print(keypoints.data)
        # draw keypoints, set first keypoint is red, second is blue
        # for keypoint in keypoints.data:
        keypoint_list = keypoints.data[index]

        if len(keypoint_list) < 2:
            return None

        center_x, center_y, _ = keypoint_list[0]
        head_x, head_y, _ = keypoint_list[1]

        cv2.circle(frame, (int(center_x), int(center_y)), 10, (0, 255, 0), -1)
        cv2.circle(frame, (int(head_x), int(head_y)), 10, (0, 255, 0), -1)
        # cv2.putText(frame, f"{keypoint_list[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, keypoint_color[i], 2)

        pointer_result['center'] = (int(center_x), int(center_y))
        pointer_result['head'] = (int(head_x), int(head_y))

        if len(keypoint_list) >= 2:
            # draw arrow line from tail to half between head and tail
            x0, y0, _ = keypoint_list[0]
            x1, y1, _ = keypoint_list[1]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 4)
        print(pointer_result)

    if save_json_result:
        cv2.imwrite(os.path.join(save_path, image_name + ".jpg"), frame)

    json_result = {}
    json_result["number_result"] = number_result
    json_result["pointer_result"] = pointer_result
    json_result["image_path"] = image_path

    frame = None

    if save_json_result:
        with open(os.path.join(save_path, image_name + ".json"), 'w') as file:
            # 使用 json.dump() 方法将数据写入文件
            json.dump(json_result, file, indent=4)
    return json_result


if __name__ == '__main__':
    task_1_save_path = "./run/main_1/"
    a = [d for d in os.listdir(task_1_save_path) if os.path.isdir(os.path.join(task_1_save_path, d))]
    for root, dirs, files in os.walk("./run/test_data"):
        for file in tqdm(files):
            if str(file) == '.DS_Store':
                continue
            if str(file).split(".")[0] not in a:
                file_path = os.path.join(root, file).replace("\\", "/")
                run(file_path, save_path=task_1_save_path, save_json_result=True)
