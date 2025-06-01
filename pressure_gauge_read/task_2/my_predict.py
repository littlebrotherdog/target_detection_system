import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# 加载模型
model = YOLO('runs/pose/train2/weights/best.pt')  # 预训练的 YOLOv8n 模型
objs_labels = model.names
# 在图片列表上运行批量推理
results = model(['../../DATASETS/images/test/045260206503494da112f78ead779af9.jpg'], save=False, imgsz=640)  # 返回 Results 对象列表




frame = cv2.imread("../../DATASETS/images/test/045260206503494da112f78ead779af9.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    boxes = boxes.cpu().numpy()  # convert to numpy array

    # 遍历每个框
    for box in boxes.data:
        l, t, r, b = box[:4].astype(np.int32)  # left, top, right, bottom
        conf, id = box[4:]  # confidence, class
        id = int(id)
        # 绘制框
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
        # 绘制类别+置信度（格式：98.1%）
        cv2.putText(frame, f"{objs_labels[id]} {conf * 100:.1f}%", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

    # 遍历 keypoints
    keypoints = result.keypoints  # Keypoints object for pose outputs
    keypoints = keypoints.cpu().numpy()  # convert to numpy array
    print(keypoints.data)
    # draw keypoints, set first keypoint is red, second is blue
    for keypoint in keypoints.data:
        for i in range(len(keypoint)):
            print(i, keypoint[i])
            x, y, _ = keypoint[i]
            x, y = int(x), int(y)
            print(x, y)
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            # cv2.putText(frame, f"{keypoint_list[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, keypoint_color[i], 2)

        if len(keypoint) >= 2:
            # draw arrow line from tail to half between head and tail
            x0, y0, _ = keypoint[0]
            x1, y1, _ = keypoint[1]

            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 4)



    # cv2.imshow('tiger', frame)
    # cv2.waitKey(0)

    # save image

    plt.imshow(frame)
    plt.show()
    # cv2.imwrite('frame.jpg', frame)
    # cv2.waitKey(0)
