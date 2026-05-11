"""CSV export for biomechanics data. Outputs a ZIP of multiple CSV files."""

import os
import csv
import io
import zipfile
import logging
from datetime import datetime
from utils.config import EXPORTS_DIR, COCO_KEYPOINT_NAMES


def _csv_string(headers, rows):
    """Write CSV rows to an in-memory string with UTF-8 BOM."""
    buf = io.StringIO()
    buf.write('﻿')  # UTF-8 BOM for Excel
    writer = csv.writer(buf)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue()


def write_biomechanics_csv(keypoint_rows, angle_rows, com_rows, kinematics_rows, statistics_rows, prefix="analysis"):
    """
    Write biomechanics data as a ZIP of CSV files (all in-memory, no temp files).

    Args:
        keypoint_rows, angle_rows, com_rows, kinematics_rows, statistics_rows: same format as excel_writer

    Returns:
        path to the generated ZIP file
    """
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    zip_path = os.path.join(EXPORTS_DIR, f"biomechanics_{prefix}_{timestamp}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if keypoint_rows:
            headers = ["图像名称", "帧编号", "时间(s)", "人体ID", "x1", "y1", "x2", "y2", "检测置信度"]
            for kp in COCO_KEYPOINT_NAMES:
                headers.extend([f"{kp}_x", f"{kp}_y", f"{kp}_conf"])
            zf.writestr("keypoints.csv", _csv_string(headers, keypoint_rows))

        if angle_rows:
            headers = ["图像名称", "帧编号", "时间(s)", "人体ID",
                       "左肘(°)", "右肘(°)", "左肩(°)", "右肩(°)",
                       "左髋(°)", "右髋(°)", "左膝(°)", "右膝(°)",
                       "左踝(°)", "右踝(°)", "躯干倾角(°)", "颈部角度(°)"]
            zf.writestr("joint_angles.csv", _csv_string(headers, angle_rows))

        if com_rows:
            headers = ["图像名称", "帧编号", "时间(s)", "COM_x(px)", "COM_y(px)",
                       "COM_vx(px/s)", "COM_vy(px/s)", "COM_speed(px/s)",
                       "COM_ax(px/s²)", "COM_ay(px/s²)"]
            zf.writestr("com_data.csv", _csv_string(headers, com_rows))

        if kinematics_rows:
            headers = ["图像名称", "帧编号", "时间(s)", "关节名称", "关节中文名", "角度(°)", "角速度(°/s)", "角加速度(°/s²)"]
            zf.writestr("kinematics.csv", _csv_string(headers, kinematics_rows))

        if statistics_rows:
            headers = list(statistics_rows[0].keys())
            zf.writestr("statistics.csv", _csv_string(headers, [list(r.values()) for r in statistics_rows]))

        # reference
        from utils.config import COCO_KEYPOINT_NAMES_CN
        ref_headers = ["关键点编号", "英文名称", "中文名称"]
        ref_rows = [[i, name, COCO_KEYPOINT_NAMES_CN.get(name, name)] for i, name in enumerate(COCO_KEYPOINT_NAMES)]
        zf.writestr("keypoint_reference.csv", _csv_string(ref_headers, ref_rows))

    logging.info(f"生物力学CSV已导出: {zip_path}")
    return zip_path
