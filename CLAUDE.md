# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概要

YOLOv11 运动生物力学分析工具 — 基于 Gradio 的姿态估计 WebUI，专为单人运动技术动作分析设计。输入图像/视频，输出关节角度、质心轨迹、运动学参数等生物力学数据。

## 启动方式

双击 `启动应用.bat` 或命令行：
```
py311\python.exe app.py
```
浏览器访问 `http://127.0.0.1:7860`

`py311/` 是嵌入式 Python 3.11 发行版，第三方库在 `py311/Lib/site-packages/`。

## 嵌入式 Python 的路径陷阱

`py311/python311._pth` 完全控制 `sys.path`，**会抑制** Python 默认的"自动添加脚本所在目录到搜索路径"行为。如果在项目根目录下新增 Python 包（如 `inference/`），必须在 `app.py` 顶部显式插入：
```python
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
```
仅当新增顶层包时需要此操作。修改已有模块不需要。

## 架构

```
app.py                     — Gradio UI 定义（3 个 Tab），所有回调函数
├── inference/             — 推理层（仅姿态）
│   ├── model_manager.py   — 模型加载、LRU 缓存（上限 3 个）、下载
│   ├── predictor.py       — 单张图像推理 → 返回 (BGR图像, 关键点dict, 错误)
│   └── video_processor.py — 视频推理 → (输出视频, 帧级关键点列表, 错误)
├── biomechanics/          — 生物力学计算（纯 NumPy，不依赖 GPU）
│   ├── skeleton.py        — 骨架连接、关键点索引字典、关节定义
│   ├── anthropometry.py   — Dempster(1955) 人体段参数
│   ├── angles.py          — compute_all_angles() → 14 个关节角度
│   ├── com.py             — compute_center_of_mass() → 段法质心
│   └── kinematics.py      — 角速度/加速度（中心差分）、COM 运动学
├── visualization/         — 可视化
│   ├── skeleton_viz.py    — 骨架叠加 + 角度弧线 + COM 标注
│   └── time_series.py     — matplotlib 时序图（上肢/下肢/躯干角度 + COM 轨迹）
├── export/                — 数据导出
│   ├── excel_writer.py    — 6-Sheet 格式化 Excel
│   └── csv_writer.py      — CSV ZIP 导出（UTF-8 BOM）
└── utils/                 — 工具
    ├── config.py          — 全局常量（COCO 关键点名、模型列表、路径）
    └── translations.py    — COCO 标签中英翻译
```

**数据流**：`推理 (GPU tensor → numpy) → 生物力学 (纯 NumPy) → 可视化/导出`

关键点数据在推理后立即转为 NumPy 并释放 GPU tensor，避免显存累积。

## 颜色通道约定

- **YOLO / OpenCV / VideoWriter**：BGR
- **Gradio Image 组件**：RGB（输入和输出均为 RGB）
- `predict_image()` 返回 BGR 图像；`app.py` 中的回调负责在显示前 `cv2.cvtColor(..., BGR2RGB)`

## Gradio DataFrame 格式要求

新版 Gradio（带 Pydantic 验证）的 `gr.DataFrame` **只接受 `list[list]`**，必须在组件定义时声明 `headers=`。示例：
```python
gr.DataFrame(headers=["列1", "列2"], label="表名")
# 返回值格式: [["值1", "值2"], ...]
```
不能传 `list[dict]`，否则报 Pydantic `ValidationError`。

## 关键函数签名

- `predict_image(image, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment)` → `(annotated_bgr, kp_data_dict | None, error_str | None)`
- `process_video(video_path, model_name, device, ...)` → `(output_video_path, frames_keypoints_list, error_str | None)`
- `compute_all_angles(keypoints_xy, keypoints_conf)` → `{"joint_angles": {joint_name: {"value": deg|None, "cn_name": str, "status": "ok"|"low_confidence"|"missing"}, ...}}`
- `compute_center_of_mass(keypoints_xy)` → `{"com": (x,y)|None, "segments": {...}, "valid_mass_fraction": float}`
- `write_biomechanics_excel(kp_rows, angle_rows, com_rows, kinematics_rows, statistics_rows, prefix)` → `filepath`

关键点坐标 `keypoints_xy` 均为 `(17, 2)` 像素坐标 ndarray；`keypoints_conf` 为 `(17,)` 置信度 ndarray。

## Gradio Tab 结构

| Tab | 功能 |
|-----|------|
| 姿态推理 | 图像/视频推理 + 自动 Excel 导出 + 发送至分析 |
| 图像分析 | 子 Tab「单帧分析」(骨架+角度+COM标注) + 子 Tab「批量分析」(多图上传→图库+对照表+批量导出) |
| 视频分析 & 导出 | 视频推理 → 可视化(上肢/下肢/躯干时序图+COM轨迹+统计) → 灵活导出(Excel/CSV) |

## 模型

姿态模型存储在 `models/pose/`，支持 `yolo11n-pose` 到 `yolo11x-pose`。首次使用自动从 Ultralytics 下载。模型缓存上限 3 个，超出时 LRU 淘汰。
