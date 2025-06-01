
#### 前期准备
1. 下载模型
[我训练好的模型下载地址](https://pan.baidu.com/s/1eNnT_rh-i32oqg4b2uDkPQ?pwd=1213)，将模型存放在 datas/models 目录下

2. [具体方案介绍 CSDN](https://blog.csdn.net/nuomuo/article/details/136883680)

3. 环境说明：requirements.txt

#### 目录说明

```
yeeiee_ai
│   README.md
│   datas: 存放模型和图片
│───paddle_ocr: 使用 paddle ocr 进行数字识别 
│───run: 运行过程中产生的临时文件夹
└───task_1: 使用YOLOv8模型训练识别图片中的指针
│   │   my_data.txt
│   │   my_train.txt
│   └───runs: 训练过程中的一些结果
│       │   file111.txt
│       │   ...
└───task_2: 使用YOLOv8模型训练识别图片中的数标
│   │   my_data.txt
│   │   my_train.txt
│   └───runs: 训练过程中的一些结果
│       │   file111.txt
│       │   ...  
└───predict.py: 主程序
└───step_1.py: 识别图片中的指针、数标等
└───step_2.py: 计算具体的读数（包括透视变换、角度计算等）
```

