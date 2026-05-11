"""Kinematic calculations: angular velocity/acceleration, joint displacement/velocity."""

import numpy as np


def _interpolate_nan(arr, max_gap=5):
    """Linearly interpolate NaN values, leaving gaps > max_gap as NaN."""
    arr = np.asarray(arr, dtype=np.float64)
    nans = np.isnan(arr)
    if not nans.any():
        return arr

    x = np.arange(len(arr))
    result = arr.copy()
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])

    # Re-mask gaps larger than max_gap
    gap_start = None
    for i in range(len(arr)):
        if nans[i] and gap_start is None:
            gap_start = i
        elif not nans[i] and gap_start is not None:
            if i - gap_start > max_gap:
                result[gap_start:i] = np.nan
            gap_start = None
    if gap_start is not None and len(arr) - gap_start > max_gap:
        result[gap_start:] = np.nan

    return result


def _central_diff(values, dt):
    """Central difference for interior points, forward/backward for endpoints."""
    n = len(values)
    if n < 2:
        return np.zeros_like(values)
    deriv = np.zeros_like(values)
    deriv[0] = (values[1] - values[0]) / dt
    deriv[-1] = (values[-1] - values[-2]) / dt
    if n > 2:
        deriv[1:-1] = (values[2:] - values[:-2]) / (2 * dt)
    return deriv


def compute_joint_angular_kinematics(angles_over_time, fps):
    """
    Compute angular velocity and acceleration for each joint over time.

    Args:
        angles_over_time: list of dicts from compute_all_angles(), one per frame
        fps: frames per second

    Returns:
        dict: {
            joint_name: {
                "cn_name": str,
                "angles": [float|None],
                "angular_velocity": [float|None],  # deg/s
                "angular_acceleration": [float|None],  # deg/s^2
            }, ...
        }
    """
    if not angles_over_time or fps <= 0:
        return {}

    dt = 1.0 / fps
    n_frames = len(angles_over_time)

    # collect joint names and initialize
    joint_names = list(angles_over_time[0]["joint_angles"].keys())
    result = {}
    for jn in joint_names:
        cn_name = angles_over_time[0]["joint_angles"][jn]["cn_name"]
        result[jn] = {
            "cn_name": cn_name,
            "angles": [],
            "angular_velocity": [],
            "angular_acceleration": [],
        }

    # extract angle time series
    for frame_data in angles_over_time:
        for jn in joint_names:
            v = frame_data["joint_angles"][jn]["value"]
            result[jn]["angles"].append(v)

    # compute velocity & acceleration
    for jn in joint_names:
        angles = result[jn]["angles"]
        arr = np.array([a if a is not None else np.nan for a in angles], dtype=np.float64)
        arr = _interpolate_nan(arr, max_gap=5)

        vel = _central_diff(arr, dt)
        acc = _central_diff(vel, dt)

        result[jn]["angular_velocity"] = [float(v) if not np.isnan(v) else None for v in vel]
        result[jn]["angular_acceleration"] = [float(v) if not np.isnan(v) else None for v in acc]

    return result


def compute_joint_kinematics(keypoints_over_time, fps):
    """
    Compute joint displacement, velocity, and acceleration for each keypoint over time.

    Args:
        keypoints_over_time: list of (17, 2) ndarrays, one per frame
        fps: frames per second

    Returns:
        dict: {
            kp_idx (0-16): {
                "displacement": [(dx, dy), ...],
                "velocity": [(vx, vy), ...],      # px/s
                "speed": [float, ...],
            }, ...
        }
    """
    if not keypoints_over_time or fps <= 0:
        return {}

    dt = 1.0 / fps
    n_frames = len(keypoints_over_time)
    n_kp = keypoints_over_time[0].shape[0]

    result = {}
    for k in range(n_kp):
        positions = np.array([frame[k] for frame in keypoints_over_time])  # (T, 2)
        vel = _central_diff(positions, dt)  # (T, 2)
        acc = _central_diff(vel, dt)  # (T, 2)

        disp = [None] * n_frames
        vel_list = [None] * n_frames
        acc_list = [None] * n_frames
        speed_list = [None] * n_frames

        for t in range(n_frames):
            if t > 0:
                dp = positions[t] - positions[t - 1]
                disp[t] = (float(dp[0]), float(dp[1]))
            vel_list[t] = (float(vel[t][0]), float(vel[t][1]))
            acc_list[t] = (float(acc[t][0]), float(acc[t][1]))
            speed_list[t] = float(np.linalg.norm(vel[t]))

        result[k] = {
            "displacement": disp,
            "velocity": vel_list,
            "acceleration": acc_list,
            "speed": speed_list,
        }

    return result


def compute_com_kinematics(com_positions, fps):
    """
    Compute COM velocity and acceleration over time.

    Args:
        com_positions: list of (com_x, com_y) tuples or None per frame
        fps: frames per second

    Returns:
        dict: {
            "com_x": [float|None], "com_y": [float|None],
            "com_vx": [float|None], "com_vy": [float|None],
            "com_speed": [float|None],
            "com_ax": [float|None], "com_ay": [float|None],
        }
    """
    n = len(com_positions)
    if n == 0 or fps <= 0:
        return {}

    dt = 1.0 / fps
    xs = np.array([p[0] if p is not None else np.nan for p in com_positions], dtype=np.float64)
    ys = np.array([p[1] if p is not None else np.nan for p in com_positions], dtype=np.float64)

    xs = _interpolate_nan(xs, max_gap=5)
    ys = _interpolate_nan(ys, max_gap=5)

    vx = _central_diff(xs, dt)
    vy = _central_diff(ys, dt)
    ax = _central_diff(vx, dt)
    ay = _central_diff(vy, dt)

    def safe(val):
        return float(val) if not np.isnan(val) else None

    return {
        "com_x": [safe(x) for x in xs],
        "com_y": [safe(y) for y in ys],
        "com_vx": [safe(v) for v in vx],
        "com_vy": [safe(v) for v in vy],
        "com_speed": [safe(np.sqrt(vx[i]**2 + vy[i]**2)) for i in range(n)],
        "com_ax": [safe(a) for a in ax],
        "com_ay": [safe(a) for a in ay],
    }
