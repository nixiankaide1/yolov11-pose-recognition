"""Center of mass estimation using Dempster segmental analysis."""

import numpy as np
from biomechanics.skeleton import KP, get_toe_approx, get_hand_end, get_midpoint
from biomechanics.anthropometry import SEGMENT_PARAMS, SEGMENT_ENDPOINTS


def _get_point(name, keypoints_xy):
    """Resolve a named point from keypoints data.

    Handles real keypoints (e.g. 'left_shoulder'), midpoints ('shoulder_midpoint', 'hip_midpoint'),
    and approximated endpoints ('left_toe_approx', 'left_hand_end').
    """
    # midpoints
    if name == "shoulder_midpoint":
        l = _get_point("left_shoulder", keypoints_xy)
        r = _get_point("right_shoulder", keypoints_xy)
        return get_midpoint(l, r) if l is not None and r is not None else None
    if name == "hip_midpoint":
        l = _get_point("left_hip", keypoints_xy)
        r = _get_point("right_hip", keypoints_xy)
        return get_midpoint(l, r) if l is not None and r is not None else None

    # approximated endpoints
    if name == "left_toe_approx":
        ankle = _get_point("left_ankle", keypoints_xy)
        knee = _get_point("left_knee", keypoints_xy)
        return get_toe_approx(ankle, knee) if ankle is not None and knee is not None else None
    if name == "right_toe_approx":
        ankle = _get_point("right_ankle", keypoints_xy)
        knee = _get_point("right_knee", keypoints_xy)
        return get_toe_approx(ankle, knee) if ankle is not None and knee is not None else None
    if name == "left_hand_end":
        wrist = _get_point("left_wrist", keypoints_xy)
        elbow = _get_point("left_elbow", keypoints_xy)
        return get_hand_end(wrist, elbow) if wrist is not None and elbow is not None else None
    if name == "right_hand_end":
        wrist = _get_point("right_wrist", keypoints_xy)
        elbow = _get_point("right_elbow", keypoints_xy)
        return get_hand_end(wrist, elbow) if wrist is not None and elbow is not None else None

    # real keypoint
    idx = KP.get(name)
    if idx is not None:
        return keypoints_xy[idx].astype(np.float64)
    return None


def compute_center_of_mass(keypoints_xy):
    """
    Compute whole-body center of mass using Dempster segment parameters.

    Args:
        keypoints_xy: (17, 2) ndarray — pixel coordinates for one person

    Returns:
        dict: {
            "com": (com_x, com_y) or None,
            "segments": {seg_name: {"com": (x, y), "proximal": (x, y), "distal": (x, y), "mass": float, "cn_name": str}, ...},
            "valid_mass_fraction": float (0.0-1.0)
        }
    """
    total_com = np.array([0.0, 0.0])
    total_mass = 0.0
    segments_detail = {}
    max_mass = sum(s["mass"] for s in SEGMENT_PARAMS.values())

    for seg_name, endpoints in SEGMENT_ENDPOINTS.items():
        params = SEGMENT_PARAMS[seg_name]
        prox_name, dist_name, _ = endpoints

        p_prox = _get_point(prox_name, keypoints_xy)
        p_dist = _get_point(dist_name, keypoints_xy)

        if p_prox is None or p_dist is None:
            continue

        seg_com = p_prox + params["com_ratio"] * (p_dist - p_prox)
        total_com += params["mass"] * seg_com
        total_mass += params["mass"]

        segments_detail[seg_name] = {
            "com": tuple(seg_com),
            "proximal": tuple(p_prox),
            "distal": tuple(p_dist),
            "mass": params["mass"],
            "cn_name": params["cn_name"],
        }

    if total_mass < 1e-6:
        return {"com": None, "segments": segments_detail, "valid_mass_fraction": 0.0}

    # normalize if some segments are missing
    scale = max_mass / total_mass
    total_com *= scale

    return {
        "com": (float(total_com[0]), float(total_com[1])),
        "segments": segments_detail,
        "valid_mass_fraction": total_mass / max_mass,
    }
