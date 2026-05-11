# YOLOv11 运动生物力学分析工具

基于 YOLOv11 姿态估计的运动生物力学分析 WebUI，专为单人运动技术动作的科学研究设计。输入图像或视频，输出关节角度、质心轨迹、角速度/角加速度等生物力学参数，支持 Excel/CSV 格式化数据导出。

## 快速开始

```bash
# 双击启动（Windows）
启动应用.bat

# 或命令行启动
py311\python.exe app.py
```

浏览器访问 **http://127.0.0.1:7860**

首次使用会自动从 Ultralytics 下载姿态模型到 `models/pose/`。

## 功能

### Tab 1 — 姿态推理

| 输入 | 输出 |
|------|------|
| 单张或多张图像 | 骨架叠加图 + 自动 Excel 导出 |
| 视频文件 | 骨架标注视频 + 逐帧生物力学 Excel 导出 |

推理参数可调：置信度阈值、IOU、TTA 增强、**FP16 半精度**。视频模式支持发送至 Tab 3 进行深度分析。

### Tab 2 — 图像分析

- **单帧分析**：骨架绘制 + 关节角度弧线标注 + 质心 (COM) 位置 + 角度表
- **批量分析**：多图上传 → 图库展示 + 关节角度对照表 + 批量 Excel 导出

### Tab 3 — 视频分析 & 导出

- 逐帧姿态推理 → **Butterworth 低通滤波平滑** → 生物力学计算
- **时序图表**：上肢 / 下肢 / 躯干与颈部关节角度曲线
- **质心轨迹图**（帧编号着色）
- **统计摘要表**：均值、最大值、最小值、活动范围 ROM、标准差
- **灵活导出**：Excel (.xlsx) 或 CSV (.zip)，可按需勾选数据类型

## 性能优化

| 优化项 | 说明 |
|--------|------|
| **FP16 半精度推理** | GPU 推理速度 ~2x，CUDA 设备默认启用 |
| **模型预热** | 加载时自动跑 dummy 前向传播，消除首次推理延迟 |
| **帧批处理** | 视频推理批量处理（batch_size=8），GPU 吞吐量 ~2-4x |
| **ONNX Runtime** | 支持 ONNX 格式模型，自动优先加载 .onnx 文件 |
| **LRU 模型缓存** | 上限 3 个，自动淘汰最少使用的模型 |
| **关键点时序平滑** | 4 阶 Butterworth 零相位低通滤波（`filtfilt`），消除模型抖动对运动学数据的放大效应 |

## Butterworth 平滑滤波

姿态估计模型输出的关键点坐标存在帧间高频抖动，经过角度计算和二阶中心差分后，角速度/角加速度的噪声被剧烈放大。

本工具实现了生物力学领域的标准做法：

- **算法**：4 阶 Butterworth 低通滤波器 + `scipy.signal.filtfilt`（零相位前向-反向滤波）
- **作用位置**：对原始关键点 (x,y) 坐标时序进行滤波，一处处理，角度/质心/运动学全部受益
- **可调参数**：截止频率 1–20 Hz（默认 10 Hz），越低曲线越平滑
- **安全机制**：短 NaN 间隔自动插值、长缺失段保留、帧数不足自动跳过

| 截止频率 | 适用场景 |
|----------|----------|
| 5–8 Hz | 慢速运动（太极拳、康复训练） |
| 8–12 Hz | 一般运动（跑步、跳跃、球类） |
| 12–20 Hz | 快速运动（短跑冲刺、爆发力动作） |

## 架构

```
app.py                     — Gradio WebUI（3 个 Tab）及所有回调函数
├── inference/             — 推理层（GPU 推理）
│   ├── model_manager.py   — YOLO 模型加载、LRU 缓存、ONNX 优先加载
│   ├── predictor.py       — 单张图像姿态推理
│   ├── video_processor.py — 视频推理（帧批处理 + 进度回调）
│   └── exporter.py        — PT → ONNX 模型导出
├── biomechanics/          — 生物力学计算（纯 NumPy，不依赖 GPU）
│   ├── skeleton.py        — COCO 17 关键点骨架拓扑、关节定义
│   ├── anthropometry.py   — Dempster(1955) 人体段惯性参数
│   ├── angles.py          — 14 个关节角度计算
│   ├── com.py             — 段法全身质心计算
│   ├── kinematics.py      — 角速度/角加速度（中心差分 + NaN 插值）
│   └── smoothing.py       — Butterworth 低通滤波关键点平滑
├── visualization/         — 可视化层
│   ├── skeleton_viz.py    — OpenCV 骨架叠加 + 角度弧线 + COM 标注
│   └── time_series.py     — Matplotlib 时序图 + 统计摘要
├── export/                — 数据导出
│   ├── excel_writer.py    — 6-Sheet 格式化 Excel（openpyxl）
│   └── csv_writer.py      — CSV ZIP 导出（UTF-8 BOM，纯内存操作）
└── utils/
    └── config.py          — COCO 关键点字典、模型列表、路径、滤波参数
```

**数据流**：`GPU 推理 (YOLO tensor) → NumPy 关键点 → Butterworth 平滑（可选）→ 角度/质心/运动学计算 → 可视化 / Excel&CSV 导出`

GPU tensor 在推理后立即转为 NumPy 并释放，避免显存累积。

## 模型

| 模型 | 参数量 | 推理速度 | 精度 |
|------|--------|----------|------|
| yolo11n-pose | 2.6M | 最快 | 适合实时/预览 |
| yolo11s-pose | 9.4M | 快 | 平衡 |
| yolo11m-pose | 20.1M | 中等 | 高精度 |
| yolo11l-pose | 44.1M | 较慢 | 更高精度 |
| yolo11x-pose | 58.8M | 最慢 | 最高精度 |

模型存储在 `models/pose/`，PT 和 ONNX 格式均可。首次使用自动下载，最大缓存 3 个模型。

## 导出数据格式

### Excel (.xlsx)

6 个工作表：

| Sheet | 内容 |
|-------|------|
| 关键点坐标 | 17 个 COCO 关键点的 x, y 像素坐标及置信度 |
| 关节角度 | 14 个关节角度（°） |
| 质心数据 | COM 位置、速度、加速度 |
| 运动学数据 | 长格式：每关节每帧的角度、角速度（°/s）、角加速度（°/s²） |
| 统计摘要 | 各关节的均值、最大值、最小值、ROM、标准差 |
| 参考信息 | COCO 关键点索引对照表 |

### CSV (.zip)

与 Excel 相同的 6 个 CSV 文件打包为 ZIP，UTF-8 BOM 编码，Excel 可直接打开。

## 环境要求

- **Python**：3.10+（项目内置嵌入式 Python 3.11）
- **GPU**：CUDA 兼容 GPU（可选，CPU 推理也支持）
- **操作系统**：Windows 10/11

## 主要依赖

| 库 | 用途 |
|----|------|
| ultralytics 8.3+ | YOLO 姿态估计模型 |
| Gradio 4.44+ | Web UI |
| PyTorch 2.0+ | 深度学习推理 |
| OpenCV | 图像/视频 I/O、骨架绘制 |
| NumPy / SciPy | 生物力学计算、信号滤波 |
| Matplotlib | 时序图表 |
| openpyxl | Excel 导出 |

## 注意事项

- **单人分析**：本项目专为单人运动技术动作分析设计，多人场景会自动选择置信度最高的主体
- **FP16**：仅 CUDA 设备可用，CPU 模式下该选项自动隐藏
- **ONNX 模型**：如需使用，先在 Python 中调用 `inference/exporter.py` 导出，之后自动优先加载
- **平滑滤波**：少于 16 帧或 FPS ≤ 1 的视频会自动跳过滤波
- **Gradio 状态**：大数据行不经过 Gradio state 序列化（避免 HTTP Content-Length 溢出），改用内存缓存
