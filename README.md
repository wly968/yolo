# YOLOv1 From Scratch (纯手写实现)

本项目是本人独立手写实现的 YOLOv1 目标检测模型，旨在深入理解单阶段目标检测的底层原理。

## 🌟 核心亮点
- **不使用任何现成检测框架**，纯手写 Dataset 编码（7x7x30 张量填充）、多任务 MSE Loss、解码与 NMS 逻辑。
- **Backbone** 采用预训练 ResNet18 替代原版 Darknet，解决显存溢出并加速收敛。
- **全流程工程实现**：包含 AMP 混合精度、梯度裁剪、AdamW + Cosine 调度、断点续训及基于 Torchmetrics 的 mAP 评估。
- **支持断点续训**：完美保存/恢复模型、优化器、调度器与 Scaler 状态，防止训练中断。

## 🚀 实验结果
在 PASCAL VOC 2007 数据集（合并 07+12 trainval）上，历经 135 个 epoch，最终 **mAP@0.5 达到 20.0% 左右**。

## 🛠️ 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 准备数据（下载 VOC 数据集至 `datasets/` 目录）
3. 开始训练: `python yolov1/train.py`
4. 测试推理: `python yolov1/predict.py`