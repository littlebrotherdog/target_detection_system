import json
import math
import os.path
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# 解决负号'-'显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 解决plt图片中文乱码问题：使用系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Songti SC', 'Wawati TC', 'STHeiti']





def delete_repeat_ocr_txt(number_result):
    # 创建一个空的结果列表和一个临时字典保存每个唯一OCR文本的最大分数
    tmp_dict = {}
    delete_key = []
    for key, item in number_result.items():
        if item['ocr_txt'] not in tmp_dict or item['ocr_score'] > tmp_dict[item['ocr_txt']]:
            tmp_dict[item['ocr_txt']] = item['ocr_score']
    for key, item in number_result.items():
        if item['ocr_txt'] not in tmp_dict or item['ocr_score'] != tmp_dict[item['ocr_txt']]:
            delete_key.append(key)
    # 将临时字典中的结果添加到最终结果列表
    for key in delete_key:
        number_result.pop(key)
    return number_result


def sort_number_result(number_result):
    # 将字典转换为列表并按 ocr_txt 的数值大小排序
    sorted_results = sorted(number_result.values(), key=lambda x: float(x['ocr_txt']), reverse=True)
    ocr_txt_list = []
    # 更新排序后的结果的 index
    for i, result in enumerate(sorted_results, start=0):
        result['index'] = i
        ocr_txt_list.append(result['ocr_txt'])

    # 如果你想保持其为字典形式，可以将其转换回字典，但请注意此时不再是原始的键值对应关系
    new_number_result = {i: result for i, result in enumerate(sorted_results, start=0)}
    return new_number_result, ocr_txt_list
    # 输出新的结果


def run(json_result, save_path="./run/main_2/", tran=True, if_plt=True):
    os.makedirs(save_path, exist_ok=True)
    data = json_result

    origin_image_path = data['image_path']
    image_name = origin_image_path.split('/')[-1]

    number_result = data['number_result']
    # 如果字典中有相同的ocr_txt，保留score较大的那一条数据
    number_result = delete_repeat_ocr_txt(number_result)

    frame = cv2.imread(origin_image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pointer_result = data['pointer_result']
    cent_x, cent_y = pointer_result['center']
    head_x, head_y = pointer_result['head']
    print(f"指针中心关键点的坐标是：({cent_x}, {cent_y})")
    print(f"指针头部关键点的坐标是：({head_x}, {head_y})")
    cv2.line(frame, (int(cent_x), int(cent_y)), (int(head_x), int(head_y)), (255, 0, 255), 5)
    cv2.circle(frame, (int(cent_x), int(cent_y)), 10, (0, 255, 0), -1)
    cv2.circle(frame, (int(head_x), int(head_y)), 10, (0, 255, 0), -1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 7))

    # 1.找出最接近OCR结果的正确的列表
    common_ocr_degrees = [
        {"1.6": 45, "1.2": -22.5, "0.8": -90, "0.4": -157.5, "0": -225},
        {"1": 45, "0.8": -9, "0.6": -63, "0.4": -117, "0.2": -171, "0": -225},
        {"0.6": 45, "0.5": 0, "0.4": -45, "0.3": -90, "0.2": -135, "0.1": -180, "0": -225},
        {"2.5": 45, "2": -9, "1.5": -63, "1": -117, "0.5": -171, "0": -225}
    ]
    common_ocr_degree_first_end = [("1.6", "0"), ("1", "0"), ("0.6", "0"), ("2.5", "0")]
    final_right_index = None
    ocr_txt_list_list = []
    max_nums = 0
    for index, common_ocr_degree in enumerate(common_ocr_degrees):
        nums = 0
        ocr_txt_list = []
        for key, value in number_result.items():
            l, t, r, b = value["xy"]
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 255), 3)
            try:
                float_ocr_txt = float(value['ocr_txt'])
            except Exception as e:
                continue
            for common_ocr in common_ocr_degree.keys():
                if float_ocr_txt == float(common_ocr):
                    nums += 1
                    ocr_txt_list.append(common_ocr)
                    value["ocr_txt"] = common_ocr
        ocr_txt_list_list.append(ocr_txt_list)

        if nums >= 4:
            final_right_index = index
            break
        if nums > max_nums:
            max_nums = nums
            final_right_index = index

    if final_right_index is None:
        error_info = "没有找到匹配的标准列表"
        if if_plt:
            axs[0].imshow(frame)
            axs[0].text(0.5, -0.15, f'{error_info}', ha='center', va='bottom',
                        fontproperties=FontProperties(size=14), transform=axs[0].transAxes)
        fig.savefig(os.path.join(save_path, image_name), dpi=200)
        # raise ValueError(error_info)
        return None, None

    ocr_txt_list = ocr_txt_list_list[final_right_index]
    print(f"正确匹配到的刻度为{ocr_txt_list}")

    # 任务检测出的刻度数小于2，直接判定为不清晰，无法读数
    # if len(ocr_txt_list) < 0:
    #     error_info = "任务检测出的刻度数小于2，直接判定为不清晰，无法读数"
    #     if if_plt:
    #         axs[0].imshow(frame)
    #         axs[0].text(0.5, -0.15, f'{error_info}', ha='center', va='bottom',
    #                     fontproperties=FontProperties(size=14), transform=axs[0].transAxes)
    #     fig.savefig(os.path.join(save_path, image_name), dpi=200)
    #     raise ValueError(error_info)

    print(f"匹配到的标准列表：{common_ocr_degrees[final_right_index]}")

    # 2.对于读数错误的刻度，直接删除
    delete_keys = []
    for key, value in number_result.items():
        if value["ocr_txt"] not in ocr_txt_list:
            delete_keys.append(key)
    for key in delete_keys:
        l, t, r, b = number_result[key]["xy"]
        cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 3)
        number_result.pop(key)

    # 3. 计算所有数标的关键点距离中心点的平均距离
    long_list, r = [], 300
    print(number_result)
    for index, value in enumerate(number_result.values()):
        x, y = value['keypoint']
        long_list.append(math.sqrt((x - cent_x) ** 2 + (y - cent_y) ** 2))
        a, b = value['keypoint']
        cv2.circle(frame, (a, b), 10, (0, 255, 0), -1)
        r = np.mean(np.array(long_list))
    print(f"数标关键点距离中心关键点的平均距离：{r}")

    # 4.补充首尾坐标，如果没有正确匹配到首尾刻度，需要认为指定首尾刻度所在的点
    end_txt, first_txt = common_ocr_degree_first_end[final_right_index]
    if end_txt not in ocr_txt_list:
        # 假设一个点作为最后一个点，找到中心点，中心点右下角45度位置
        x = int(cent_x + r * math.cos(math.radians(45)))
        y = int(cent_y + r * math.sin(math.radians(45)))
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)
        add_dict = {"index": 99, "ocr_txt": end_txt, "keypoint": (x, y)}
        number_result[99] = add_dict

    if first_txt not in ocr_txt_list:
        # 假设一个点作为最后一个点，找到中心点，中心点右下角45度位置
        x = int(cent_x + r * math.cos(math.radians(-225)))
        y = int(cent_y + r * math.sin(math.radians(-225)))
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)
        add_dict = {"index": 100, "ocr_txt": first_txt, "keypoint": (x, y)}
        number_result[100] = add_dict

    # 重新排序
    number_result, ocr_txt_list = sort_number_result(number_result)
    print(number_result, ocr_txt_list)

    degree_list = []
    degree_360_list = []

    # OCR检测结果<4则不进行透视变换
    if len(ocr_txt_list) < 4:
        tran = False

    if tran:

        new_point = {}
        old_point = {}
        # 1. 假设所有图片最后一个点极坐标都是（45,r），得到每个刻度透视转化后应该处于的坐标
        right_change_dict = common_ocr_degrees[final_right_index]
        # 例子：{'1.6': 45, '1.2': -22.5, '0.8': -90, '0.4': -157.5, '0': -225}
        for index, (key, value) in enumerate(number_result.items()):
            # index = key = value['index']
            # value 例子： {'index': 0, 'ocr_txt': '1.2', 'ocr_score': 0.998770534992218, 'keypoint': (1248, 1127)}
            # 这里的r是使用平均长度
            degree = right_change_dict[value['ocr_txt']]
            new_x = int(cent_x + r * math.cos(math.radians(degree)))
            new_y = int(cent_y + r * math.sin(math.radians(degree)))
            number_result[key]['new_keypoint'] = (new_x, new_y)  # 新的坐标位置

            a, b = value['keypoint']
            old_point[value['ocr_txt']] = [a, b]
            new_point[value['ocr_txt']] = [new_x, new_y]
            # 画出每个数标透视转化前后后应该处于的坐标
            cv2.circle(frame, (new_x, new_y), 10, (255, 0, 0), -1)

        # 2.进行透视转换
        # 随机选择4个转换的点
        choose_list = random.sample(ocr_txt_list, 4)
        choose_list = sorted(choose_list)
        print(f"随机选择了{choose_list}进行透视转换")
        src_points = np.array([old_point[choose_list[0]], old_point[choose_list[1]],
                               old_point[choose_list[2]], old_point[choose_list[3]]], dtype=np.float32)
        dst_points = np.array([new_point[choose_list[0]], new_point[choose_list[1]],
                               new_point[choose_list[2]], new_point[choose_list[3]]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        frame_trans = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

        # 3.坐标转换
        def cvt_pos(pos, cvt_mat_t):
            u = pos[0]
            v = pos[1]
            x_ = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
                    cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
            y_ = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
                    cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
            return int(x_), int(y_)

        # 转换后指标新坐标
        new_center_x, new_center_y = cvt_pos(pointer_result['center'], M)
        new_head_x, new_head_y = cvt_pos(pointer_result['head'], M)
        r = math.atan2(new_head_y - new_center_y, new_head_x - new_center_x)
        d = math.degrees(r)
        d_360 = ((d + 360) % 360)
        # 计算所有关键点在转化后的角度，坐标等
        for index, (key, value) in enumerate(number_result.items()):
            trans_x, trans_y = cvt_pos(value['keypoint'], M)
            r1 = math.atan2(trans_y - new_center_y, trans_x - new_center_x)
            d1 = math.degrees(r1)
            d1_360 = ((d1 + 360) % 360)
            value['trans_keypoint'] = (trans_x, trans_y)  # 转换后的坐标
            value['degree'] = d1
            degree_list.append(d1)
            degree_360_list.append(d1_360)
    else:
        # 转换后指标新坐标, 这里由于没有透视转换, 所以新坐标就是旧坐标
        new_center_x, new_center_y = pointer_result['center']
        new_head_x, new_head_y = pointer_result['head']
        r = math.atan2(new_head_y - new_center_y, new_head_x - new_center_x)
        d = math.degrees(r)
        d_360 = ((d + 360) % 360)
        # 计算所有关键点在转化后的角度，坐标等
        for index, (key, value) in enumerate(number_result.items()):
            trans_x, trans_y = value['keypoint']
            r1 = math.atan2(trans_y - new_center_y, trans_x - new_center_x)
            d1 = math.degrees(r1)
            d1_360 = ((d1 + 360) % 360)
            value['trans_keypoint'] = (trans_x, trans_y)  # 转换后的坐标
            value['degree'] = d1
            degree_list.append(d1)
            degree_360_list.append(d1_360)
        frame_trans = frame

    print(degree_list)
    print(degree_360_list)
    # 找出指针在哪两个数标关键点之间
    choose_index, bet_degree_n, bet_degree_p = None, None, None
    for index, value in enumerate(degree_360_list):
        if index + 1 == len(degree_360_list):
            break
        # 如果degree_360_list[index] < degree_360_list[index+1]
        # 说明此关键点位于水平线之下，它的下一个关键点位于水平线之上
        if degree_360_list[index] < degree_360_list[index + 1]:
            # 判断指针是否在其中
            if 0 <= d_360 <= degree_360_list[index] or degree_360_list[index + 1] <= d_360 <= 360:
                bet_degree_n = 360 - degree_360_list[index + 1] + degree_360_list[index]
                if d_360 <= degree_360_list[index]:
                    bet_degree_p = 360 - degree_360_list[index + 1] + d_360
                else:
                    bet_degree_p = d_360 - degree_360_list[index + 1]
                choose_index = index
                break
        if degree_360_list[index + 1] <= d_360 <= degree_360_list[index]:
            bet_degree_n = degree_360_list[index] - degree_360_list[index + 1]
            bet_degree_p = d_360 - degree_360_list[index + 1]
            choose_index = index
            break

    if choose_index is None:
        axs[0].imshow(frame)
        # ax = axs[1]
        # ax.set_xticks([])
        # ax.set_yticks([])
        fig.savefig(os.path.join(save_path, image_name), dpi=200)
        # raise ValueError('没有找到合适的数标区间')
        # 直接赋值为 0
        axs[1].imshow(frame_trans)
        axs[1].text(0.5, -0.15, f'最终读数: 0', ha='center', va='bottom',
                    fontproperties=FontProperties(size=14),
                    transform=axs[1].transAxes)

        fig.savefig(os.path.join(save_path, image_name), dpi=200)
        plt.close(fig)
        return 0

    print("选择了区间：", choose_index + 1)

    a1, b1 = number_result[choose_index]["trans_keypoint"]
    a2, b2 = number_result[choose_index + 1]["trans_keypoint"]
    cv2.line(frame_trans, (int(new_center_x), int(new_center_y)), (int(a1), int(b1)), (255, 0, 255), 5)
    cv2.line(frame_trans, (int(new_center_x), int(new_center_y)), (int(a2), int(b2)), (255, 0, 255), 5)

    end_ocr_txt = float(number_result[choose_index]["ocr_txt"])
    start_ocr_txt = float(number_result[choose_index + 1]["ocr_txt"])
    print(f'指标刻度位于之间{start_ocr_txt}和{end_ocr_txt}之间')

    final_result = start_ocr_txt + ((end_ocr_txt - start_ocr_txt) * (bet_degree_p / bet_degree_n))
    print("最终读数", final_result)

    axs[0].imshow(frame)
    axs[0].text(0.5, -0.15, f'指标位于: {start_ocr_txt}和{end_ocr_txt}之间', ha='center', va='bottom',
                fontproperties=FontProperties(size=14),
                transform=axs[0].transAxes)

    axs[1].imshow(frame_trans)
    axs[1].text(0.5, -0.15, f'最终读数: {final_result}', ha='center', va='bottom',
                fontproperties=FontProperties(size=14),
                transform=axs[1].transAxes)

    # fig.savefig(os.path.join(save_path, image_name), dpi=200)
    plt.close(fig)

    frame = None

    return final_result, frame_trans


if __name__ == '__main__':

    task_1_save_path = "./run/main_1/"
    task_2_save_path = "./run/main_2/"

    with open(f"./run/task2_result.csv", mode="w", encoding="utf-8") as f1:

        with open("./run/right_result.csv", mode="r", encoding="utf-8") as f2:
            lines = f2.readlines()
            for line in lines:
                img_name, score = line.replace("\n", "").split(",")
                task_1_json_path = f"{task_1_save_path}{img_name}/{img_name}.json"
                print(task_1_json_path)
                try:

                    with open(task_1_json_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                        final_result, frame_trans = run(json_result=data,
                                                        save_path=task_2_save_path,
                                                        tran=True,
                                                        if_plt=True
                                                        )
                        f1.write(f"{task_1_json_path},{score},{final_result}\n")
                    file.close()
                except Exception as e:
                    f1.write(f"{task_1_json_path},{str(e)}\n")

        # for root, dirs, files in os.walk(task_1_save_path):
        #    for file in files:
        #        if file.endswith("json"):
        #            try:
        #                task_1_json_path = f"{task_1_save_path}{file.split('.')[0]}/{file}"
        #                final_result = run(json_path=task_1_json_path,
        #                                   save_path=task_2_save_path,
        #                                   tran=True,
        #                                   if_plt=True)
        #
        #                f.write(f"{task_1_json_path},{final_result}\n")
        #            except Exception as e:
        #                print("------------", e)
        #                f.write(f"{task_1_json_path},{str(e)}\n")
        #
