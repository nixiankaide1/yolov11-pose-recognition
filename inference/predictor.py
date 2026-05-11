"""Single-image pose inference with structured keypoint output."""

import os
import cv2
import numpy as np
import logging
from datetime import datetime

from inference.model_manager import get_model
from utils.config import EXPORTS_DIR, COCO_KEYPOINT_NAMES, CONFIDENCE_THRESHOLD


def select_primary_subject(keypoints_conf, boxes_xyxy):
    """
    Select the primary subject from multiple detections.

    Strategy: highest mean keypoint confidence, with bounding box area as tiebreaker.
    Returns the 0-based index of the selected person.
    """
    if keypoints_conf is None or len(keypoints_conf) == 0:
        return 0
    if len(keypoints_conf) == 1:
        return 0

    mean_confs = np.mean(keypoints_conf, axis=1)
    best_idx = int(np.argmax(mean_confs))

    # tiebreaker: larger bbox area when confs are very close (< 0.01 diff)
    ties = np.where(np.abs(mean_confs - mean_confs[best_idx]) < 0.01)[0]
    if len(ties) > 1:
        areas = (boxes_xyxy[ties, 2] - boxes_xyxy[ties, 0]) * (boxes_xyxy[ties, 3] - boxes_xyxy[ties, 1])
        best_idx = ties[int(np.argmax(areas))]

    return best_idx


def _extract_keypoints_data(results):
    """
    Extract structured keypoint data from ultralytics Results.

    Returns a dict:
        keypoints_xy:    ndarray (N, 17, 2) — pixel coordinates
        keypoints_conf:  ndarray (N, 17)    — confidence per keypoint
        boxes_xyxy:      ndarray (N, 4)     — bounding boxes
        boxes_conf:      ndarray (N,)       — detection confidence
    or None if no detections.
    """
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        kp_xy = []
        kp_conf = []
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        boxes_conf = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(r.boxes))

        if r.keypoints is not None:
            for k in r.keypoints:
                kp_xy.append(k.xy[0].cpu().numpy())       # (17, 2)
                kp_conf.append(k.conf[0].cpu().numpy())    # (17,)
        else:
            kp_xy = np.zeros((len(r.boxes), 17, 2))
            kp_conf = np.zeros((len(r.boxes), 17))

        return {
            "keypoints_xy": np.array(kp_xy) if isinstance(kp_xy, list) else kp_xy,
            "keypoints_conf": np.array(kp_conf) if isinstance(kp_conf, list) else kp_conf,
            "boxes_xyxy": boxes_xyxy,
            "boxes_conf": boxes_conf,
        }
    return None


def predict_image(image, model_name, device, fp16=False, conf=0.25, iou=0.7, max_det=300,
                  classes=None, agnostic_nms=False, augment=False):
    """
    Run pose inference on a single image.

    Args:
        image: numpy array (RGB, from Gradio)
        model_name, device, fp16, conf, iou, max_det, classes, agnostic_nms, augment

    Returns:
        (annotated_bgr, keypoints_data_dict_or_None, error_str_or_None)
    """
    model = get_model(model_name, device, fp16=fp16)

    # Gradio provides RGB; YOLO accepts BGR
    if len(image.shape) == 3 and image.shape[-1] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    try:
        results = model(image_bgr, device=device, conf=conf, iou=iou,
                        max_det=max_det, classes=classes,
                        agnostic_nms=agnostic_nms, augment=augment)

        kp_data = _extract_keypoints_data(results)
        annotated = results[0].plot()

        # auto-save to exports/
        try:
            os.makedirs(EXPORTS_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(EXPORTS_DIR, f"pose_{ts}.jpg")
            cv2.imwrite(save_path, annotated)
        except Exception as e:
            logging.warning(f"保存推理图片失败: {e}")

        return annotated, kp_data, None

    except Exception as e:
        logging.error(f"预测错误: {e}", exc_info=True)
        return None, None, str(e)
