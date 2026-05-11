"""Model loading, caching, and download management — pose-only."""

import os
import shutil
import logging

from ultralytics import YOLO
from utils.config import MODELS_DIR, POSE_MODELS

models = {}


def get_available_models():
    """Return sorted list of available pose model filenames (pt and onnx)."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    local = set()
    for f in os.listdir(MODELS_DIR):
        if f.endswith('.pt') or f.endswith('.onnx'):
            local.add(f)
    return sorted(list(local.union(POSE_MODELS)))


def download_model(model_name):
    """Download a YOLO pose model and store it under models/pose/."""
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        return model_path

    logging.info(f"正在下载模型: {model_name}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    temp_model = YOLO(model_name)

    possible_sources = [
        model_name,
        os.path.join(os.getcwd(), model_name),
    ]
    try:
        from ultralytics.utils import SETTINGS
        weights_dir = SETTINGS.get('weights_dir', '')
        if weights_dir:
            possible_sources.append(os.path.join(weights_dir, model_name))
    except Exception:
        pass

    source_found = False
    for src in possible_sources:
        if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(model_path):
            shutil.copy2(src, model_path)
            cwd_dir = os.path.abspath(os.getcwd())
            if os.path.dirname(os.path.abspath(src)) == cwd_dir:
                try:
                    os.remove(src)
                except OSError:
                    pass
            source_found = True
            break

    if not source_found:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型下载后未找到文件: {model_name}")

    logging.info(f"模型已保存到: {model_path}")
    return model_path


def get_model(model_name, device, fp16=False):
    """Load a pose model with LRU cache (max 3) and optional FP16."""
    global models
    import torch

    cuda_available = torch.cuda.is_available()

    if model_name not in models and len(models) >= 3:
        oldest_key = next(iter(models))
        try:
            del models[oldest_key]
            if cuda_available:
                torch.cuda.empty_cache()
            logging.info(f"模型缓存已满，释放: {oldest_key}")
        except Exception:
            pass

    if model_name not in models:
        model_path = os.path.join(MODELS_DIR, model_name)
        # Prefer ONNX for faster CPU inference (YOLO constructor auto-detects format)
        onnx_path = model_path.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            model_path = onnx_path
            logging.info(f"使用 ONNX 模型: {onnx_path}")
        elif not os.path.exists(model_path):
            model_path = download_model(model_name)
        models[model_name] = YOLO(model_path)

        if device == "cuda" and cuda_available:
            models[model_name].to(device)
            if fp16:
                models[model_name].model.half()
                logging.info(f"已启用 FP16 半精度推理: {model_name}")
            # Warmup: eliminate first-inference lag
            try:
                dummy = torch.zeros(1, 3, 640, 640, device=device)
                if fp16:
                    dummy = dummy.half()
                models[model_name](dummy, verbose=False)
            except Exception:
                pass
    elif device == "cuda" and cuda_available:
        models[model_name].to(device)

    return models[model_name]


def release_model(model_name=None):
    """Release cached models to free GPU memory."""
    global models
    import torch

    cuda_available = torch.cuda.is_available()

    if model_name is None:
        models.clear()
    elif model_name in models:
        del models[model_name]

    if cuda_available:
        torch.cuda.empty_cache()
    logging.info(f"已释放模型缓存" if model_name is None else f"已释放模型: {model_name}")
