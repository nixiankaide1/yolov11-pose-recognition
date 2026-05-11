import os
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['GRADIO_OFFLINE'] = 'True'

import sys
# 嵌入式 Python 使用 ._pth 文件覆盖了默认路径，需手动添加项目根目录
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# 添加 py311 嵌入式 Python 目录
py311_path = os.path.join(_project_root, 'py311')
if py311_path not in sys.path:
    sys.path.append(py311_path)

import gradio as gr
gr.close_all()

import logging
import cv2
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from inference.model_manager import get_model, get_available_models
from inference.predictor import predict_image, select_primary_subject
from inference.video_processor import process_video

from biomechanics.angles import compute_all_angles
from biomechanics.com import compute_center_of_mass
from biomechanics.kinematics import (
    compute_joint_angular_kinematics,
    compute_com_kinematics,
)

from visualization.skeleton_viz import draw_biomechanics_overlay
from visualization.time_series import (
    generate_angle_time_series,
    generate_com_trajectory,
    generate_statistics_table,
)

from export.excel_writer import write_biomechanics_excel
from export.csv_writer import write_biomechanics_csv

from utils.config import (DEVICE_DEFAULT, COCO_KEYPOINT_NAMES,
                         SMOOTHING_DEFAULT_ENABLED, SMOOTHING_DEFAULT_CUTOFF,
                         SMOOTHING_MIN_FRAMES)
from biomechanics.smoothing import smooth_keypoints


def _safe_progress(progress, pct, msg):
    """Call gr.Progress safely, ignoring errors from uninitialized tracker."""
    try:
        progress(pct, desc=msg)
    except Exception:
        pass


# Module-level cache for large export data — avoids blowing up Gradio state JSON
_export_cache = {}


# ── Helper: build export data rows ──────────────────────────────────────────

def _build_export_rows(kp_data_list, fps):
    """Build all export data row formats from keypoint data + biomechanics analysis."""
    keypoint_rows = []
    angle_rows = []
    com_rows = []
    kinematics_rows = []
    all_angles = []

    for frame_data in kp_data_list:
        if frame_data is None:
            continue

        kp_xy = frame_data["keypoints_xy"]
        kp_conf = frame_data["keypoints_conf"]
        boxes_xyxy = frame_data["boxes_xyxy"]
        boxes_conf = frame_data["boxes_conf"]
        frame_idx = frame_data.get("frame_idx", 0)
        img_name = frame_data.get("image_name", f"frame_{frame_idx:06d}")
        time_sec = round(frame_idx / fps, 4) if fps > 0 else 0.0

        # select primary subject (respect pre-set best_idx if available)
        if "best_idx" in frame_data:
            best = frame_data["best_idx"]
        else:
            best = select_primary_subject(kp_conf, boxes_xyxy)
        best_kp_xy = kp_xy[best]
        best_kp_conf = kp_conf[best]
        x1, y1, x2, y2 = boxes_xyxy[best].tolist()
        det_conf = float(boxes_conf[best])

        # keypoints row
        kp_row = [img_name, frame_idx, time_sec, best + 1, x1, y1, x2, y2, det_conf]
        for k in range(len(COCO_KEYPOINT_NAMES)):
            if k < len(best_kp_xy):
                kp_row.extend([float(best_kp_xy[k][0]), float(best_kp_xy[k][1]),
                               float(best_kp_conf[k]) if k < len(best_kp_conf) else 0.0])
            else:
                kp_row.extend([0.0, 0.0, 0.0])
        keypoint_rows.append(kp_row)

        # biomechanics
        angles_data = compute_all_angles(best_kp_xy, best_kp_conf)
        all_angles.append(angles_data)

        com_data = compute_center_of_mass(best_kp_xy)
        com = com_data.get("com")

        # angle row
        angle_order = [
            "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "trunk_inclination", "neck_angle",
        ]
        a_row = [img_name, frame_idx, time_sec, best + 1]
        for jn in angle_order:
            info = angles_data["joint_angles"].get(jn, {})
            v = info.get("value")
            a_row.append(round(v, 1) if v is not None else None)
        angle_rows.append(a_row)

        # COM row
        if com is not None:
            com_rows.append([img_name, frame_idx, time_sec, float(com[0]), float(com[1]),
                             None, None, None, None, None])
        else:
            com_rows.append([img_name, frame_idx, time_sec, None, None, None, None, None, None, None])

    # kinematics (only meaningful for multi-frame video)
    if fps > 0 and len(all_angles) > 1:
        ang_kin = compute_joint_angular_kinematics(all_angles, fps)
        com_list = [r[3:5] if len(r) >= 5 and r[3] is not None else None for r in com_rows]
        com_kin = compute_com_kinematics(
            [(float(r[3]), float(r[4])) if r[3] is not None else None for r in com_rows], fps)

        # fill COM velocity/acceleration into com_rows
        if com_kin:
            for i, cr in enumerate(com_rows):
                if i < len(com_kin.get("com_vx", [])):
                    cr[5] = com_kin["com_vx"][i]
                    cr[6] = com_kin["com_vy"][i]
                    cr[7] = com_kin["com_speed"][i]
                    cr[8] = com_kin["com_ax"][i]
                    cr[9] = com_kin["com_ay"][i]

        # kinematics long-format rows
        for frame_i, frame_data in enumerate(kp_data_list):
            if frame_data is None:
                continue
            frame_idx = frame_data.get("frame_idx", 0)
            img_name = frame_data.get("image_name", f"frame_{frame_idx:06d}")
            time_sec = round(frame_idx / fps, 4) if fps > 0 else 0.0
            for jn in angle_order:
                if jn in ang_kin:
                    info = ang_kin[jn]
                    ang = info["angles"][frame_i] if frame_i < len(info["angles"]) else None
                    vel = info["angular_velocity"][frame_i] if frame_i < len(info["angular_velocity"]) else None
                    acc = info["angular_acceleration"][frame_i] if frame_i < len(info["angular_acceleration"]) else None
                    kinematics_rows.append([
                        img_name, frame_idx, time_sec, jn, info["cn_name"],
                        round(ang, 2) if ang is not None else None,
                        round(vel, 2) if vel is not None else None,
                        round(acc, 2) if acc is not None else None,
                    ])
    else:
        ang_kin = {}

    stats_rows = generate_statistics_table(all_angles, fps) if all_angles else []

    return keypoint_rows, angle_rows, com_rows, kinematics_rows, stats_rows, all_angles, ang_kin


# ── Gradio UI ───────────────────────────────────────────────────────────────

def _make_model_choices():
    models = get_available_models()
    return gr.Dropdown(choices=models, value=models[0] if models else None)


with gr.Blocks(title="YOLOv11 运动生物力学分析工具") as demo:
    gr.Markdown("""# YOLOv11 运动生物力学分析工具
    基于 YOLOv11 姿态估计的运动技术动作分析平台。支持图像批量分析、视频分析，计算关节角度、质心轨迹、运动学参数。
    """)

    device_options = ["cuda", "cpu"] if DEVICE_DEFAULT == "cuda" else ["cpu"]

    # ========================================================================
    # Tab 1: 姿态推理
    # ========================================================================
    with gr.Tab("姿态推理"):
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=get_available_models(),
                    label="选择姿态模型",
                    value=get_available_models()[0] if get_available_models() else None,
                )
                device_input = gr.Radio(device_options, label="设备选择", value=DEVICE_DEFAULT)

                source_type = gr.Radio(["图像", "视频"], label="输入类型", value="图像")

                image_input = gr.Image(type="numpy", label="上传图像", visible=True)
                video_input = gr.Video(label="上传视频", visible=False)

                with gr.Accordion("推理参数", open=False):
                    conf_slider = gr.Slider(0.1, 1.0, 0.25, label="置信度阈值")
                    iou_slider = gr.Slider(0.1, 1.0, 0.7, label="IOU阈值")
                    max_det_input = gr.Number(300, label="最大检测数", precision=0)
                    augment_checkbox = gr.Checkbox(False, label="使用TTA增强")
                    fp16_checkbox = gr.Checkbox(DEVICE_DEFAULT == "cuda", label="FP16半精度 (GPU加速~2x)", visible=(DEVICE_DEFAULT == "cuda"))
                    smooth_enable = gr.Checkbox(SMOOTHING_DEFAULT_ENABLED, label="启用关键点平滑 (Butterworth滤波)")
                    smooth_cutoff = gr.Slider(1, 20, SMOOTHING_DEFAULT_CUTOFF, step=0.5, label="滤波截止频率 (Hz, 越低越平滑)")

                infer_btn = gr.Button("开始推理", variant="primary")
                send_to_analysis_btn = gr.Button("发送至生物力学分析", visible=False)

            with gr.Column(scale=1):
                infer_image_output = gr.Image(type="numpy", label="推理结果", visible=True)
                infer_video_output = gr.Video(label="推理结果", visible=False)
                infer_status = gr.Textbox(label="状态")
                infer_excel_download = gr.File(label="Excel数据下载", visible=False)

        # store last inference data for "send to analysis"
        infer_state = gr.State({})

        def toggle_source(st):
            return [
                gr.update(visible=(st == "图像")),
                gr.update(visible=(st == "视频")),
                gr.update(visible=(st == "图像")),
                gr.update(visible=(st == "视频")),
            ]

        source_type.change(toggle_source, [source_type],
                          [image_input, video_input, infer_image_output, infer_video_output])

        def on_infer(source_type_val, image, video, model_name, device, conf, iou, max_det, augment, fp16,
                     smooth_enable, smooth_cutoff, progress=gr.Progress()):
            if source_type_val == "图像":
                if image is None:
                    return None, None, gr.update(visible=False), gr.update(visible=False), "请上传图像", gr.update(value={})
                annotated, kp_data, err = predict_image(
                    image, model_name, device, fp16=fp16, conf=conf, iou=iou, max_det=int(max_det),
                    classes=None, agnostic_nms=False, augment=augment,
                )
                if err:
                    return None, None, gr.update(visible=False), gr.update(visible=False), err, gr.update(value={})
                # build minimal kp_data_list for export
                if kp_data is not None:
                    kp_data["image_name"] = "image"
                    kp_data["frame_idx"] = 0
                    kp_list = [kp_data]
                    kp_rows, a_rows, c_rows, k_rows, s_rows, _, _ = _build_export_rows(kp_list, fps=0)
                    excel_path = write_biomechanics_excel(kp_rows, a_rows, c_rows, k_rows, s_rows, "image")
                    state = {"kp_data_list": kp_list, "fps": 0, "source_type": "图像"}
                    # convert BGR to RGB for display
                    if len(annotated.shape) == 3 and annotated.shape[-1] == 3:
                        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    return annotated, None, gr.update(visible=True, value=excel_path), gr.update(visible=True), "推理完成", gr.update(value=state)
                return annotated, None, gr.update(visible=False), gr.update(visible=True), "未检测到人体", gr.update(value={})
            else:
                if video is None:
                    return None, None, gr.update(visible=False), gr.update(visible=False), "请上传视频", gr.update(value={})
                progress(0, desc="开始处理视频")
                out_path, kp_list, err = process_video(
                    video, model_name, device, fp16=fp16, conf=conf, iou=iou, max_det=int(max_det),
                    classes=None, agnostic_nms=False, augment=augment, sample_interval=1,
                    progress_callback=lambda pct, msg: _safe_progress(progress, pct, msg),
                )
                if err:
                    return None, None, gr.update(visible=False), gr.update(visible=False), err, gr.update(value={})
                # get fps from video
                if isinstance(video, dict):
                    vpath = video.get('name', video)
                else:
                    vpath = video
                cap = cv2.VideoCapture(vpath)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()

                for i, fd in enumerate(kp_list):
                    fd["image_name"] = f"frame_{fd.get('frame_idx', i+1):06d}"
                    # Assign best_idx for smoothing compatibility
                    if "best_idx" not in fd:
                        fd["best_idx"] = select_primary_subject(
                            fd["keypoints_conf"], fd["boxes_xyxy"])
                if smooth_enable and fps > 1 and len(kp_list) >= SMOOTHING_MIN_FRAMES:
                    smooth_keypoints(kp_list, fps, cutoff=smooth_cutoff)

                kp_rows, a_rows, c_rows, k_rows, s_rows, _, _ = _build_export_rows(kp_list, fps)
                excel_path = write_biomechanics_excel(kp_rows, a_rows, c_rows, k_rows, s_rows, "video")
                state = {"kp_data_list": kp_list, "fps": fps, "source_type": "视频"}
                return None, out_path, gr.update(visible=True, value=excel_path), gr.update(visible=True), f"处理完成，共 {len(kp_list)} 帧有检测结果", gr.update(value=state)

        infer_btn.click(
            on_infer,
            [source_type, image_input, video_input, model_selector, device_input,
             conf_slider, iou_slider, max_det_input, augment_checkbox, fp16_checkbox,
             smooth_enable, smooth_cutoff],
            [infer_image_output, infer_video_output, infer_excel_download, send_to_analysis_btn, infer_status, infer_state],
        )

    # ========================================================================
    # Tab 2: 图像分析
    # ========================================================================
    with gr.Tab("图像分析"):
        with gr.Tabs():
            # ── Sub-tab 2A: 单帧分析 ──
            with gr.Tab("单帧分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sf_image = gr.Image(type="numpy", label="上传运动图像")
                        sf_model = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择姿态模型",
                            value=get_available_models()[0] if get_available_models() else None,
                        )
                        sf_device = gr.Radio(device_options, label="设备选择", value=DEVICE_DEFAULT)
                        sf_target = gr.Number(0, label="分析目标编号 (0=自动选择)", precision=0)
                        sf_show_angles = gr.Checkbox(True, label="显示关节角度标注")
                        sf_show_com = gr.Checkbox(True, label="显示质心位置")
                        sf_fp16 = gr.Checkbox(DEVICE_DEFAULT == "cuda", label="FP16半精度", visible=(DEVICE_DEFAULT == "cuda"))
                        sf_btn = gr.Button("开始分析", variant="primary")

                    with gr.Column(scale=1):
                        sf_overlay = gr.Image(type="numpy", label="生物力学分析图")
                        sf_angle_table = gr.DataFrame(
                            headers=["关节名称", "角度 (°)", "状态"],
                            label="关节角度表",
                        )
                        sf_com_info = gr.Textbox(label="质心信息", lines=2)

                def on_single_frame_analyze(image, model_name, device, target, show_angles, show_com, fp16):
                    if image is None:
                        return None, None, "请上传图像"
                    annotated, kp_data, err = predict_image(
                        image, model_name, device, fp16=fp16, conf=0.25, iou=0.7, max_det=300,
                        classes=None, agnostic_nms=False, augment=False,
                    )
                    if err or kp_data is None:
                        return None, None, f"推理失败: {err or '未检测到人体'}"
                    kp_xy = kp_data["keypoints_xy"]
                    kp_conf = kp_data["keypoints_conf"]
                    # select subject
                    if target and target > 0:
                        best = min(int(target) - 1, len(kp_xy) - 1)
                    else:
                        best = select_primary_subject(kp_conf, kp_data["boxes_xyxy"])
                    best_kp_xy = kp_xy[best]
                    best_kp_conf = kp_conf[best]
                    # biomechanics
                    angles_data = compute_all_angles(best_kp_xy, best_kp_conf)
                    com_data = compute_center_of_mass(best_kp_xy)
                    # overlay
                    overlay = draw_biomechanics_overlay(annotated, best_kp_xy, best_kp_conf, show_angles, show_com)
                    if len(overlay.shape) == 3 and overlay.shape[-1] == 3:
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    # angle table — list of lists for gr.DataFrame
                    table = []
                    for jn, info in angles_data["joint_angles"].items():
                        v = info["value"]
                        table.append([
                            info["cn_name"],
                            f"{v:.1f}" if v is not None else "N/A",
                            info["status"],
                        ])
                    com = com_data.get("com")
                    com_str = f"COM: ({com[0]:.1f}, {com[1]:.1f}) px  |  有效质量比例: {com_data.get('valid_mass_fraction', 0):.1%}" if com else "COM: 无法计算"
                    return overlay, table, com_str

                sf_btn.click(
                    on_single_frame_analyze,
                    [sf_image, sf_model, sf_device, sf_target, sf_show_angles, sf_show_com, sf_fp16],
                    [sf_overlay, sf_angle_table, sf_com_info],
                )

            # ── Sub-tab 2B: 批量分析 ──
            with gr.Tab("批量分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            file_count="multiple", file_types=["image"],
                            label="上传多张运动图像",
                        )
                        batch_model = gr.Dropdown(
                            choices=get_available_models(),
                            label="选择姿态模型",
                            value=get_available_models()[0] if get_available_models() else None,
                        )
                        batch_device = gr.Radio(device_options, label="设备选择", value=DEVICE_DEFAULT)
                        batch_target = gr.Number(0, label="分析目标编号 (0=自动选择)", precision=0)
                        batch_fp16 = gr.Checkbox(DEVICE_DEFAULT == "cuda", label="FP16半精度", visible=(DEVICE_DEFAULT == "cuda"))
                        batch_btn = gr.Button("开始批量分析", variant="primary")

                    with gr.Column(scale=1):
                        batch_gallery = gr.Gallery(label="生物力学分析结果", columns=2, height="500px")
                        batch_table = gr.DataFrame(
                            headers=["图片名称", "左肘", "右肘", "左肩", "右肩", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "躯干倾角", "颈部角度"],
                            label="关节角度对照表 (°)",
                        )
                        batch_download = gr.File(label="批量导出Excel", visible=False)

                def on_batch_analyze(files, model_name, device, target, fp16):
                    if not files:
                        return None, None, gr.update(visible=False)
                    gallery_imgs = []
                    all_angle_data = {}
                    kp_list_all = []
                    angle_order = [
                        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
                        "left_hip", "right_hip", "left_knee", "right_knee",
                        "left_ankle", "right_ankle", "trunk_inclination", "neck_angle",
                    ]
                    cn_names = ["左肘", "右肘", "左肩", "右肩", "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "躯干倾角", "颈部角度"]

                    for f in files:
                        # gr.File returns file paths, NamedStrings, or dicts
                        if isinstance(f, str):
                            fpath = f
                        elif isinstance(f, dict):
                            fpath = f.get('name', '')
                        elif hasattr(f, 'name'):
                            fpath = f.name
                        else:
                            fpath = str(f)
                        fname = os.path.basename(fpath)

                        img = cv2.imread(fpath)
                        if img is None:
                            continue
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        annotated, kp_data, err = predict_image(
                            img_rgb, model_name, device, fp16=fp16, conf=0.25, iou=0.7, max_det=300,
                            classes=None, agnostic_nms=False, augment=False,
                        )
                        if err or kp_data is None:
                            continue

                        kp_xy = kp_data["keypoints_xy"]
                        kp_conf = kp_data["keypoints_conf"]
                        if target and target > 0:
                            best = min(int(target) - 1, len(kp_xy) - 1)
                        else:
                            best = select_primary_subject(kp_conf, kp_data["boxes_xyxy"])
                        best_kp_xy = kp_xy[best]
                        best_kp_conf = kp_conf[best]

                        angles_data = compute_all_angles(best_kp_xy, best_kp_conf)
                        overlay = draw_biomechanics_overlay(annotated, best_kp_xy, best_kp_conf, True, True)
                        if len(overlay.shape) == 3 and overlay.shape[-1] == 3:
                            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        gallery_imgs.append((overlay, fname))

                        # collect angles
                        row_angles = []
                        for jn in angle_order:
                            info = angles_data["joint_angles"].get(jn, {})
                            v = info.get("value")
                            row_angles.append(f"{v:.1f}" if v is not None else "N/A")
                        all_angle_data[fname] = row_angles

                        # collect for export
                        kp_data["image_name"] = fname
                        kp_data["frame_idx"] = 0
                        kp_list_all.append(kp_data)

                    # build comparison table
                    tbl_headers = ["图片名称"] + cn_names
                    tbl = []
                    for img_name in all_angle_data:
                        tbl.append([img_name] + all_angle_data[img_name])

                    # export Excel
                    excel_path = None
                    if kp_list_all:
                        kp_rows, a_rows, c_rows, k_rows, s_rows, _, _ = _build_export_rows(kp_list_all, fps=0)
                        excel_path = write_biomechanics_excel(kp_rows, a_rows, c_rows, k_rows, s_rows, "batch")

                    return (
                        gallery_imgs if gallery_imgs else None,
                        tbl if tbl else [],
                        gr.update(visible=True, value=excel_path) if excel_path else gr.update(visible=False),
                    )

                batch_btn.click(
                    on_batch_analyze,
                    [batch_files, batch_model, batch_device, batch_target, batch_fp16],
                    [batch_gallery, batch_table, batch_download],
                )

    # ========================================================================
    # Tab 3: 视频分析 & 导出
    # ========================================================================
    with gr.Tab("视频分析 & 导出"):
        gr.Markdown("""### 视频运动生物力学分析
        上传运动视频 → 逐帧推理 → 关节角度时序图 + COM轨迹 + 统计摘要 → 导出完整数据。
        """)
        with gr.Row():
            with gr.Column(scale=1):
                ve_video = gr.Video(label="上传运动视频")
                ve_model = gr.Dropdown(
                    choices=get_available_models(),
                    label="选择姿态模型",
                    value=get_available_models()[0] if get_available_models() else None,
                )
                ve_device = gr.Radio(device_options, label="设备选择", value=DEVICE_DEFAULT)
                ve_target = gr.Number(0, label="分析目标编号 (0=自动选择)", precision=0)
                ve_sample = gr.Number(1, label="帧采样间隔 (1=全部分析)", precision=0)
                ve_fp16 = gr.Checkbox(DEVICE_DEFAULT == "cuda", label="FP16半精度", visible=(DEVICE_DEFAULT == "cuda"))
                ve_smooth_enable = gr.Checkbox(SMOOTHING_DEFAULT_ENABLED, label="启用关键点平滑 (Butterworth滤波)")
                ve_smooth_cutoff = gr.Slider(1, 20, SMOOTHING_DEFAULT_CUTOFF, step=0.5, label="滤波截止频率 (Hz, 越低越平滑)")

                with gr.Accordion("导出内容", open=True):
                    ve_do_kp = gr.Checkbox(True, label="关键点坐标")
                    ve_do_ang = gr.Checkbox(True, label="关节角度")
                    ve_do_com = gr.Checkbox(True, label="质心数据")
                    ve_do_kin = gr.Checkbox(True, label="运动学数据")
                    ve_do_stats = gr.Checkbox(True, label="统计摘要")
                ve_format = gr.Radio(["Excel (.xlsx)", "CSV (.zip)"], label="导出格式", value="Excel (.xlsx)")
                ve_btn = gr.Button("开始分析", variant="primary")

            with gr.Column(scale=1):
                ve_status = gr.Textbox(label="分析状态")
                with gr.Tabs():
                    with gr.Tab("上肢角度"):
                        ve_upper_plot = gr.Plot(label="上肢关节角度")
                    with gr.Tab("下肢角度"):
                        ve_lower_plot = gr.Plot(label="下肢关节角度")
                    with gr.Tab("躯干与颈部"):
                        ve_trunk_plot = gr.Plot(label="躯干与颈部角度")
                ve_com_plot = gr.Plot(label="质心轨迹")
                ve_stats = gr.DataFrame(
                    headers=["关节名称", "平均值 (°)", "最大值 (°)", "最小值 (°)", "活动范围 ROM (°)", "标准差 (°)", "有效帧数"],
                    label="统计摘要",
                )
                with gr.Row():
                    ve_export_btn = gr.Button("导出数据文件")
                    ve_download = gr.File(label="下载文件", visible=True)

        ve_state = gr.State({})

        def on_video_export_analyze(video, model_name, device, target, sample_interval,
                                     do_kp, do_ang, do_com, do_kin, do_stats, exp_fmt, fp16,
                                     ve_smooth_enable, ve_smooth_cutoff, progress=gr.Progress()):
            if video is None:
                return [None]*5 + [gr.update(value={})] + ["请上传视频"] + [gr.update(value=None)]
            plt.close('all')  # clean up matplotlib figures from previous runs
            progress(0, desc="开始处理视频")
            out_path, kp_list, err = process_video(
                video, model_name, device, fp16=fp16, conf=0.25, iou=0.7, max_det=300,
                classes=None, agnostic_nms=False, augment=False,
                sample_interval=int(sample_interval), batch_size=8,
                progress_callback=lambda pct, msg: _safe_progress(progress, pct, msg),
            )
            if err or not kp_list:
                return [None]*5 + [gr.update(value={})] + [f"处理失败: {err or '无检测结果'}"] + [gr.update(value=None)]

            if isinstance(video, dict):
                vpath = video.get('name', video)
            else:
                vpath = video
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()

            # select primary subject per frame (biomechanics computed once in _build_export_rows)
            filtered_kp = []
            for fd in kp_list:
                kp_xy = fd["keypoints_xy"]
                kp_conf = fd["keypoints_conf"]
                if target and target > 0:
                    best = min(int(target) - 1, len(kp_xy) - 1)
                else:
                    best = select_primary_subject(kp_conf, fd["boxes_xyxy"])
                fd["best_idx"] = best
                fd["image_name"] = f"frame_{fd.get('frame_idx', 0):06d}"
                filtered_kp.append(fd)

            # Apply Butterworth smoothing to keypoint trajectories
            if ve_smooth_enable and fps > 1 and len(filtered_kp) >= SMOOTHING_MIN_FRAMES:
                smooth_keypoints(filtered_kp, fps, cutoff=ve_smooth_cutoff)

            # build export data + compute biomechanics (all_angles, ang_kin, com_rows)
            kp_rows, a_rows, c_rows, k_rows, s_rows, all_angles, ang_kin = _build_export_rows(filtered_kp, fps)

            # extract com_positions from com_rows for charting
            com_positions = [(r[3], r[4]) if len(r) >= 5 and r[3] is not None else None for r in c_rows]

            # charts (using pre-computed data, no recomputation)
            upper_fig = generate_angle_time_series(ang_kin, fps, "upper")
            lower_fig = generate_angle_time_series(ang_kin, fps, "lower")
            trunk_fig = generate_angle_time_series(ang_kin, fps, "trunk")
            com_fig = generate_com_trajectory(com_positions)
            stats_rows = list(s_rows)

            # filter per checkboxes
            if not do_kp: kp_rows = []
            if not do_ang: a_rows = []
            if not do_com: c_rows = []
            if not do_kin: k_rows = []
            if not do_stats: s_rows = []

            prefix = "video"
            if exp_fmt == "Excel (.xlsx)":
                file_path = write_biomechanics_excel(kp_rows, a_rows, c_rows, k_rows, s_rows, prefix)
            else:
                file_path = write_biomechanics_csv(kp_rows, a_rows, c_rows, k_rows, s_rows, prefix)

            # convert stats for DataFrame
            stats_headers = ["关节名称", "平均值 (°)", "最大值 (°)", "最小值 (°)", "活动范围 ROM (°)", "标准差 (°)", "有效帧数"]
            stats_list = [[r.get(h, "N/A") for h in stats_headers] for r in stats_rows] if stats_rows else []

            # Store large row data in module cache (not Gradio state) to avoid
            # JSON serialization blowing up HTTP Content-Length.
            import uuid
            cache_key = str(uuid.uuid4())
            _export_cache[cache_key] = {
                "kp_rows": kp_rows, "a_rows": a_rows, "c_rows": c_rows,
                "k_rows": k_rows, "s_rows": s_rows,
                "exp_fmt": exp_fmt, "prefix": prefix,
            }
            state = {"cache_key": cache_key, "file_path": file_path}

            return (
                upper_fig, lower_fig, trunk_fig, com_fig, stats_list,
                gr.update(value=state),
                f"分析完成: {len(filtered_kp)} 帧有检测 ({len(all_angles)} 帧有姿态数据) | 已导出: {file_path}",
                gr.update(value=file_path),
            )

        ve_btn.click(
            on_video_export_analyze,
            [ve_video, ve_model, ve_device, ve_target, ve_sample,
             ve_do_kp, ve_do_ang, ve_do_com, ve_do_kin, ve_do_stats, ve_format, ve_fp16,
             ve_smooth_enable, ve_smooth_cutoff],
            [ve_upper_plot, ve_lower_plot, ve_trunk_plot, ve_com_plot, ve_stats, ve_state, ve_status, ve_download],
        )

        def on_re_export(state):
            if not state:
                return gr.update(value=None)
            cache_key = state.get("cache_key")
            data = _export_cache.pop(cache_key, {}) if cache_key else {}
            if not data:
                return gr.update(value=None)
            if data.get("exp_fmt") == "Excel (.xlsx)":
                path = write_biomechanics_excel(
                    data["kp_rows"], data["a_rows"], data["c_rows"],
                    data["k_rows"], data["s_rows"], data["prefix"],
                )
            else:
                path = write_biomechanics_csv(
                    data["kp_rows"], data["a_rows"], data["c_rows"],
                    data["k_rows"], data["s_rows"], data["prefix"],
                )
            return gr.update(value=path)

        ve_export_btn.click(on_re_export, [ve_state], [ve_download])


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=3)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
    )
