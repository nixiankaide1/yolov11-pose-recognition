"""Model export utilities — YOLO pose models to ONNX for faster inference."""

import os
import logging
from ultralytics import YOLO
from utils.config import MODELS_DIR


def export_to_onnx(model_name, imgsz=640, half=False):
    """
    Export a YOLO pose model to ONNX format.

    Args:
        model_name: filename (e.g. 'yolo11n-pose.pt')
        imgsz: input image size
        half: use FP16 in the ONNX model

    Returns:
        path to the .onnx file
    """
    model_path = os.path.join(MODELS_DIR, model_name)
    onnx_path = model_path.replace('.pt', '.onnx')
    if os.path.exists(onnx_path):
        logging.info(f"ONNX 模型已存在: {onnx_path}")
        return onnx_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    logging.info(f"正在导出 ONNX 模型: {model_name} -> {onnx_path}")
    model = YOLO(model_path)
    model.export(format='onnx', imgsz=imgsz, half=half, simplify=True)
    logging.info(f"ONNX 导出完成: {onnx_path}")
    return onnx_path
