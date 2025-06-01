# @Author nuomuo
# @Time: 2024/4/28 17:53
from matplotlib import pyplot as plt

from step_1 import run as step_1_run
from step_2 import run as step_2_run


def predict(image_path):
    task_1_save_path = "./run/step_1/"

    json_result = step_1_run(image_path, save_json_result=False)
    print(json_result)
    if json_result is None:
        return None, None
    final_result, final_img = step_2_run(json_result, tran=False)
    return final_result, final_img


if __name__ == '__main__':
    import time
    start = time.time()
    final_result, final_img = predict("datas/test_data/0a0e9e8ac69b465ba0478984fbd8652e.jpg")
    print(final_result)
    end = time.time()
    print(end - start)
    plt.imshow(final_img)
    plt.show()

