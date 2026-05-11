"""Time-series chart generation for biomechanics video analysis."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io


# Chinese font configuration
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _time_axis(n_frames, fps):
    """Generate time axis in seconds."""
    return np.arange(n_frames) / fps


def generate_angle_time_series(kinematics_data, fps, group="upper"):
    """
    Generate a matplotlib Figure for joint angle time series.

    Args:
        kinematics_data: dict from compute_joint_angular_kinematics()
        fps: frames per second
        group: "upper" | "lower" | "trunk"

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    t = _time_axis(len(next(iter(kinematics_data.values()))["angles"]), fps)

    groups = {
        "upper": ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
        "lower": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
        "trunk": ["trunk_inclination", "neck_angle"],
    }

    colors = plt.cm.tab10.colors
    color_idx = 0
    for jn in groups.get(group, []):
        if jn not in kinematics_data:
            continue
        data = kinematics_data[jn]
        angles = [a if a is not None else np.nan for a in data["angles"]]
        label = data["cn_name"]
        ax.plot(t, angles, label=label, color=colors[color_idx % len(colors)], linewidth=1.5)
        color_idx += 1

    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("角度 (°)")
    titles = {"upper": "上肢关节角度", "lower": "下肢关节角度", "trunk": "躯干与颈部角度"}
    ax.set_title(titles.get(group, ""))
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 190)

    fig.tight_layout()
    return fig


def generate_com_trajectory(com_positions):
    """
    Generate COM trajectory scatter plot.

    Args:
        com_positions: list of (com_x, com_y) tuples or None

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = [(c[0], c[1], i) for i, c in enumerate(com_positions) if c is not None]
    if not valid:
        ax.text(0.5, 0.5, "无COM数据", ha='center', va='center', transform=ax.transAxes)
        return fig

    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]
    frame_nums = [v[2] for v in valid]

    sc = ax.scatter(xs, ys, c=frame_nums, cmap='viridis', s=15, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("帧编号")

    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("质心轨迹 (COM Trajectory)")
    ax.invert_yaxis()  # image coordinates: y increases downward
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')

    fig.tight_layout()
    return fig


def fig_to_bytes(fig):
    """Convert matplotlib Figure to PNG bytes for Gradio."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def generate_statistics_table(angles_over_time, fps):
    """
    Generate a statistics summary table for all joint angles.

    Args:
        angles_over_time: list of dicts from compute_all_angles()
        fps: frames per second

    Returns:
        list of dicts suitable for gr.DataFrame:
        [{"关节名称": cn_name, "平均值": ..., "最大值": ..., "最小值": ..., "活动范围(ROM)": ..., "标准差": ...}, ...]
    """
    if not angles_over_time:
        return []

    joint_names = list(angles_over_time[0]["joint_angles"].keys())
    rows = []
    for jn in joint_names:
        values = []
        for frame in angles_over_time:
            v = frame["joint_angles"][jn]["value"]
            if v is not None:
                values.append(v)

        cn_name = angles_over_time[0]["joint_angles"][jn]["cn_name"]
        if values:
            arr = np.array(values)
            rows.append({
                "关节名称": cn_name,
                "平均值 (°)": round(float(np.mean(arr)), 1),
                "最大值 (°)": round(float(np.max(arr)), 1),
                "最小值 (°)": round(float(np.min(arr)), 1),
                "活动范围 ROM (°)": round(float(np.max(arr) - np.min(arr)), 1),
                "标准差 (°)": round(float(np.std(arr)), 1),
                "有效帧数": len(values),
            })
        else:
            rows.append({
                "关节名称": cn_name,
                "平均值 (°)": "N/A",
                "最大值 (°)": "N/A",
                "最小值 (°)": "N/A",
                "活动范围 ROM (°)": "N/A",
                "标准差 (°)": "N/A",
                "有效帧数": 0,
            })

    return rows
