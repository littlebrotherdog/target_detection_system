# 压力表读数自动检测系统设计与实现报告    

## 一、项目背景与目标

随着工业现场监控与设备运行状态识别的需求日益增长，人工读取压力表存在效率低、误读率高等问题。为提升巡检智能化水平，本文设计并实现了一个基于 **YOLOv5** 的压力表读数自动检测系统。

本系统以现场拍摄图像为输入，自动检测并定位压力表指针位置，输出包含指针区域标注的图像，辅助实现读数分析与数字化改造。

系统目标：

- ✅ 自动检测压力表指针区域；
- ✅ 输出可视化检测结果，辅助后续读数提取；
- ✅ 具备良好的检测准确性与运行速度，适配工业部署需求。

---

## 二、系统架构设计

本系统采用模块化架构，主要包括以下四个核心模块：

### 1. 数据采集与标注模块

- 图像来源：工业现场采集或使用公开压力表图像数据集；
- 标注工具：LabelImg；
- 标注格式：YOLO 格式（class cx cy w h，归一化）。

### 2. 模型训练模块

- 框架：PyTorch + YOLOv5；
- 网络结构：YOLOv5s（轻量、高速）；
- 数据配置文件：`my_data.yaml`

```yaml
path: /data/wangac/yolov5/DATASETS
train: images/train
val: images/val
names:
  0: pointer
```

### 3. 模型推理模块

- 单张图像推理：`my_predict.py`
- 批量图像推理：`my_predict_batch.py`
- 输出：原图 + 检测框图像，用于辅助后续分析

### 4. 可视化与评估模块

- 评估指标：`mAP`、`Precision`、`Recall`；
- 支持输出检测结果图像，供人工审查使用。

---

## 三、模型说明

YOLOv5 是一种经典的一阶段目标检测器，具备以下模块：

- **Backbone**：CSPDarknet（用于图像特征提取）；
- **Neck**：PANet（多尺度融合）；
- **Head**：负责类别预测与边框回归；
- **优势**：精度与速度兼顾，适用于实际部署场景。

---

## 四、数据集说明

- 图像数量：
  - 训练集：11 张
  - 验证集：6 张（可拓展）
- 图像尺寸：640 × 640；
- 类别名称：`pointer`（压力表指针）；
- 标注格式：YOLOv5 标准 `.txt` 文本格式。

---

## 五、训练配置

训练命令如下：

```bash
python train.py --img 640 --batch 16 --epochs 100 --data my_data.yaml --weights yolov5s.pt --name pressure_pointer_detector
```

参数说明：

- 输入尺寸：640  
- 批大小：16  
- 训练轮数：100  
- 预训练模型：`yolov5s.pt`

---

## 六、系统推理测试报告

### 测试环境

- 操作系统：Ubuntu 20.04  
- PyTorch：1.10+  
- GPU：NVIDIA RTX 3080  

### 评估结论：

- ✅ 检测精度满足实际压力表识别需求；
- ⚠️ 小表盘与反光图像仍需进一步优化；
- ✅ 推理速度支持工业实时检测要求。

---

## 七、源代码结构说明

```
pressure_gauge_yolov5/
├── data/
├── images/            # 含 train/ 和 val/
├── labels/            # YOLO 格式标注
├── weights/           # best.pt 模型文件
├── my_predict.py
├── my_predict_batch.py
├── README.md
├── requirements.txt
```

### 关键脚本说明：

- `my_predict.py`: 对单张图像进行检测；
- `my_predict_batch.py`: 批量检测图像；
- `train.py`: 模型训练主脚本（YOLOv5官方）；
- `export.py`: 导出模型为 ONNX 等格式；
- `requirements.txt`: 环境依赖安装文件

```bash
pip install -r requirements.txt
```

---

## 八、系统优化方向

- ✅ 扩展压力表种类与数据集规模；
- ✅ 引入图像去反光与增强模块，提升鲁棒性；
- ✅ 引入指针角度计算模块，实现自动读数；
- ✅ 支持部署于边缘计算设备；
- ✅ 增加视频流实时识别支持。

---

## 九、结语

本文设计并实现了一个基于 YOLOv5 的压力表指针检测系统，完成了从数据采集、训练、推理到可视化评估的完整流程。系统在检测准确性与实时性方面表现良好，为后续实现“压力表自动化读数”打下坚实基础。
