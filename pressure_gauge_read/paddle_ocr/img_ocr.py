import math

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

import logging

logging.disable(logging.DEBUG)
# for example : `ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch",
                ocr_version="PP-OCRv4")  # need to run only once to download and load model into memory


def ocr_img(img, cls=True):
    result = ocr.ocr(img, cls=cls)
    # print(result)
    if result[0] is None:
        return None, None
    max_ = max(result, key=lambda x: x[0][0][1][0])
    return max_[0][1][0], max_[0][1][1]


def run(img, cls=False):
    image_np = np.array(img)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result = ocr.ocr(image_cv2, cls=True)
    print(result)
    result = result[0]
    if result is None:
        return None, None
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    return txts, scores

def np_detect(img):

    image_np = np.array(img)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    result = ocr.ocr(image_cv2, cls=True)


    result = result[0]
    # image = Image.open(img_path).convert('RGB')
    if result is None or len(result) == 0:
        return None
    boxes = [line[0] for line in result]

    all = 0
    for index, box in enumerate(boxes):
        a, b, c, d = box
        # 计算两点之间的水平距离和垂直距离
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        all += math.atan2(dy, dx)
        dx1 = c[0] - d[0]
        dy1 = c[1] - d[1]
        all += math.atan2(dy1, dx1)
        # 将弧度转换为角度
        # angle_deg = math.degrees(angle_rad)

    # angle_deg = math.degrees(all / len(boxes) / 2.0)
    # print(angle_deg)
    # d_360 = ((angle_deg + 360) % 360)
    # height, width, _ = img.shape
    # rotation_matrix = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=d_360, scale=1)
    # rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    return txts
    # print(txts)
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    # im_show = Image.fromarray(rotated_img)
    # im_show.show()


def nameplate_ocr_run(image):
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = ocr.ocr(image_cv2, cls=False)
    result = results[0]
    if result is None:
        return None, None
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    print(txts)
    return txts, scores


if __name__ == '__main__':
    img = Image.open("D:\workspace\yeeiee_ai\paddle_ocr\diy_kv_template_match_ocr\\test_data\\22a38f916b4e49ca868822524ea451eb.jpg")
    img.show()
    run(img)

