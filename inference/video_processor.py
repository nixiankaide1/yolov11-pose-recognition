"""Video pose inference — frame-by-frame with keypoint extraction."""

import os
import cv2
import tempfile
import subprocess
import shutil
import logging
from datetime import datetime

from inference.model_manager import get_model
from inference.predictor import _extract_keypoints_data
from utils.config import EXPORTS_DIR


def _flush_batch(model, frame_buffer, frames_keypoints, out, device,
                 conf, iou, max_det, classes, agnostic_nms, augment,
                 total_frames, processed_offset, progress_callback):
    """Run batch inference on buffered frames and write annotated output."""
    batch_frames = [f for f, _ in frame_buffer]
    batch_indices = [i for _, i in frame_buffer]

    results_list = model(batch_frames, device=device, conf=conf, iou=iou,
                        max_det=max_det, classes=classes,
                        agnostic_nms=agnostic_nms, augment=augment)

    for batch_i, (results, frame_idx) in enumerate(zip(results_list, batch_indices)):
        kp_data = _extract_keypoints_data([results])
        if kp_data is not None:
            kp_data["frame_idx"] = frame_idx
            frames_keypoints.append(kp_data)

        annotated = results.plot()
        out.write(annotated)

        global_processed = processed_offset + batch_i + 1
        if global_processed % 10 == 0 or frame_idx == total_frames:
            pct = frame_idx / max(total_frames, 1)
            if progress_callback:
                try:
                    progress_callback(pct, f"处理帧 {frame_idx}/{total_frames}")
                except Exception:
                    pass
            logging.info(f"视频帧处理进度: {frame_idx}/{total_frames}")


def process_video(video_path, model_name, device, fp16=False, conf=0.25, iou=0.7, max_det=300,
                  classes=None, agnostic_nms=False, augment=False, sample_interval=1,
                  batch_size=1, progress_callback=None):
    """
    Run pose inference on every frame of a video.

    Args:
        video_path: str or Gradio dict with 'name' key
        model_name, device, fp16: model selection
        conf, iou, max_det, classes, agnostic_nms, augment: inference params
        sample_interval: process every Nth frame (1 = all frames)
        batch_size: number of frames to batch together for GPU inference
        progress_callback: callable(pct, message) for progress reporting

    Returns:
        (output_video_path, frames_keypoints_list, error_str_or_None)
        frames_keypoints_list is a list of dicts per processed frame, each dict:
            {"keypoints_xy": ndarray (N,17,2), "keypoints_conf": ndarray (N,17),
             "boxes_xyxy": ndarray (N,4), "boxes_conf": ndarray (N,),
             "frame_idx": int}
    """
    if isinstance(video_path, dict) and 'name' in video_path:
        video_path = video_path['name']
    elif not isinstance(video_path, str):
        return None, None, f"无效的视频路径类型: {type(video_path)}"

    model = get_model(model_name, device, fp16=fp16)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None, None, "无法打开视频文件"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        output_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_keypoints = []
    frame_count = 0
    processed_count = 0
    frame_buffer = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if (frame_count - 1) % sample_interval != 0:
            out.write(frame)
            continue

        frame_buffer.append((frame, frame_count))

        if len(frame_buffer) >= batch_size:
            _flush_batch(model, frame_buffer, frames_keypoints, out, device,
                        conf, iou, max_det, classes, agnostic_nms, augment,
                        total_frames, processed_count, progress_callback)
            processed_count += len(frame_buffer)
            frame_buffer = []

    if frame_buffer:
        _flush_batch(model, frame_buffer, frames_keypoints, out, device,
                    conf, iou, max_det, classes, agnostic_nms, augment,
                    total_frames, processed_count, progress_callback)
        processed_count += len(frame_buffer)

    cap.release()
    out.release()
    logging.info(f"视频推理完成，共 {processed_count} 帧，检测到 {len(frames_keypoints)} 帧有人体")

    # FFmpeg audio merge
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_ffmpeg = os.path.join(script_dir, "FFmpeg", "bin", "ffmpeg.exe")
    ffmpeg_path = local_ffmpeg if os.path.exists(local_ffmpeg) else shutil.which("ffmpeg")

    if not ffmpeg_path:
        logging.warning("FFmpeg 未找到，输出无音频")
        final_path = output_path
    else:
        final_path = output_path.replace('.mp4', '_final.mp4')
        cmd = [
            ffmpeg_path, "-y",
            "-i", output_path, "-i", video_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            final_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(output_path)
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg 转码失败: {e}")
            if os.path.exists(final_path):
                os.remove(final_path)
            final_path = output_path

    # auto-save to exports/
    try:
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.basename(video_path)
        saved = os.path.join(EXPORTS_DIR, f"video_pose_{os.path.splitext(video_name)[0]}_{ts}.mp4")
        shutil.copy2(final_path, saved)
    except Exception as e:
        logging.warning(f"保存推理视频失败: {e}")

    return final_path, frames_keypoints, None
