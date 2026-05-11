"""Temporal smoothing for keypoint trajectories — Butterworth low-pass filter."""

import logging
import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_lowpass(data, fps, cutoff=10.0, order=4):
    """
    Zero-phase Butterworth low-pass filter (filtfilt).

    Args:
        data: 1D numpy array (must be finite — no NaN, length > 3*order)
        fps: sampling rate in Hz
        cutoff: -3 dB cutoff frequency in Hz (default 10 Hz)
        order: filter order (default 4)

    Returns:
        filtered 1D array of same shape
    """
    n = len(data)
    min_padlen = 3 * order
    if n <= min_padlen:
        return data  # not enough data for reliable filtfilt
    nyquist = 0.5 * fps
    if cutoff >= nyquist:
        return data  # cutoff too high for sampling rate, skip
    b, a = butter(order, cutoff / nyquist, btype='low')
    padlen = min(3 * order, n - 1)
    return filtfilt(b, a, data, padlen=padlen)


def _interp_short_nan(arr, max_gap=5):
    """Linearly interpolate short NaN runs; leave long gaps as NaN."""
    arr = np.asarray(arr, dtype=np.float64)
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    x = np.arange(len(arr))
    result = arr.copy()
    result[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    # Re-mask gaps > max_gap
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


def smooth_keypoints(filtered_kp, fps, cutoff=10.0, order=4):
    """
    Apply Butterworth low-pass filter to selected-subject keypoints across frames.

    Modifies filtered_kp in place — writes smoothed coordinates back into
    keypoints_xy[best_idx] for each frame.

    Args:
        filtered_kp: list of frame data dicts, each must have:
            - keypoints_xy: (N, 17, 2) ndarray
            - keypoints_conf: (N, 17) ndarray
            - best_idx: int (index of selected person)
        fps: frames per second
        cutoff: Butterworth cutoff frequency in Hz
        order: filter order

    Returns:
        filtered_kp (same object, modified in place)
    """
    from utils.config import SMOOTHING_MIN_FRAMES

    T = len(filtered_kp)
    if T < SMOOTHING_MIN_FRAMES or fps <= 1:
        logging.warning(f"帧数不足({T})或FPS过低({fps})，跳过滤波")
        return filtered_kp

    n_kp = 17  # COCO keypoints

    # Extract selected-subject coordinates: (T, 17, 2)
    trajectory = np.full((T, n_kp, 2), np.nan, dtype=np.float64)
    conf_mask = np.zeros((T, n_kp), dtype=bool)

    for t, fd in enumerate(filtered_kp):
        kp_xy = fd["keypoints_xy"]
        kp_conf = fd["keypoints_conf"]
        best = fd.get("best_idx", 0)
        if best >= len(kp_xy):
            continue
        trajectory[t] = kp_xy[best]
        conf_mask[t] = kp_conf[best] >= 0.3

    smoothed_count = 0
    for k in range(n_kp):
        valid = conf_mask[:, k]
        valid_count = valid.sum()
        if valid_count < SMOOTHING_MIN_FRAMES:
            continue  # too few valid frames for this keypoint

        for dim in range(2):
            signal = trajectory[:, k, dim].copy()
            # Interpolate short gaps before filtering
            signal = _interp_short_nan(signal, max_gap=5)
            finite = ~np.isnan(signal)
            if finite.sum() < SMOOTHING_MIN_FRAMES:
                continue
            # Extract contiguous segments, filter each
            filtered = signal.copy()
            # Simple approach: interpolate all NaN, filter, then re-mask original NaN
            all_finite = signal.copy()
            nans = np.isnan(all_finite)
            if nans.any():
                x = np.arange(len(all_finite))
                all_finite[nans] = np.interp(x[nans], x[~nans], all_finite[~nans])
            all_finite = butterworth_lowpass(all_finite, fps, cutoff, order)
            # Restore original NaN positions
            all_finite[nans] = np.nan
            trajectory[:, k, dim] = all_finite
            smoothed_count += 1

    # Write smoothed coordinates back
    for t, fd in enumerate(filtered_kp):
        best = fd.get("best_idx", 0)
        if best < len(fd["keypoints_xy"]):
            fd["keypoints_xy"][best] = trajectory[t]

    logging.info(f"已平滑 {smoothed_count // 2} 个关键点轨迹 (Butterworth {order}阶, cutoff={cutoff}Hz)")
    return filtered_kp
