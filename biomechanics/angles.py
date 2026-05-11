"""Joint angle calculations from COCO 17-keypoint data."""

import numpy as np
from biomechanics.skeleton import KP, JOINT_DEFS, SPECIAL_ANGLES, get_toe_approx, get_midpoint
from utils.config import CONFIDENCE_THRESHOLD


def _calc_angle(p1, p2, p3):
    """
    Angle at p2 formed by vectors p1->p2 and p2->p3, in degrees [0, 180].
    """
    v1 = np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)
    v2 = np.array(p3, dtype=np.float64) - np.array(p2, dtype=np.float64)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _kp_xy(kp_name, keypoints_xy):
    """Get (x, y) for a named keypoint from the (17, 2) array."""
    idx = KP.get(kp_name)
    if idx is None:
        return None
    return keypoints_xy[idx].astype(np.float64)


def _kp_conf(kp_name, keypoints_conf):
    """Get confidence for a named keypoint from the (17,) array."""
    idx = KP.get(kp_name)
    if idx is None:
        return 0.0
    return float(keypoints_conf[idx])


def compute_all_angles(keypoints_xy, keypoints_conf):
    """
    Compute all joint angles for one person.

    Args:
        keypoints_xy: (17, 2) ndarray — pixel coordinates
        keypoints_conf: (17,) ndarray — confidence per keypoint

    Returns:
        dict: {"joint_angles": {joint_name: {"value": deg, "cn_name": str, "confidence": str}}, ...}
        Angles with low-confidence keypoints are marked as "low_confidence".
    """
    result = {}

    for joint_name, (prox, center, dist, cn_name) in JOINT_DEFS.items():
        # resolve special distal points
        if dist == "left_toe_approx":
            ankle = _kp_xy("left_ankle", keypoints_xy)
            knee = _kp_xy("left_knee", keypoints_xy)
            p3 = get_toe_approx(ankle, knee) if ankle is not None and knee is not None else None
        elif dist == "right_toe_approx":
            ankle = _kp_xy("right_ankle", keypoints_xy)
            knee = _kp_xy("right_knee", keypoints_xy)
            p3 = get_toe_approx(ankle, knee) if ankle is not None and knee is not None else None
        else:
            p3 = _kp_xy(dist, keypoints_xy)

        p1 = _kp_xy(prox, keypoints_xy)
        p2 = _kp_xy(center, keypoints_xy)

        if p1 is None or p2 is None or p3 is None:
            value = None
            status = "missing"
        else:
            value = _calc_angle(p1, p2, p3)
            c1 = _kp_conf(prox, keypoints_conf)
            c2 = _kp_conf(center, keypoints_conf)
            c3_name = dist.replace("left_toe_approx", "left_ankle").replace("right_toe_approx", "right_ankle")
            c3 = _kp_conf(c3_name, keypoints_conf)
            min_conf = min(c1, c2, c3)
            status = "ok" if min_conf >= CONFIDENCE_THRESHOLD else "low_confidence"

        result[joint_name] = {
            "value": value,
            "cn_name": cn_name,
            "status": status,
        }

    # trunk inclination: angle between (shoulder_mid -> hip_mid) and vertical (0, -1)
    sh_mid = get_midpoint(
        _kp_xy("left_shoulder", keypoints_xy),
        _kp_xy("right_shoulder", keypoints_xy),
    )
    hip_mid = get_midpoint(
        _kp_xy("left_hip", keypoints_xy),
        _kp_xy("right_hip", keypoints_xy),
    )
    if sh_mid is not None and hip_mid is not None:
        trunk = sh_mid - hip_mid
        vertical = np.array([0.0, -1.0])
        norm_t = np.linalg.norm(trunk)
        if norm_t > 1e-6:
            cos_t = np.clip(np.dot(trunk, vertical) / norm_t, -1.0, 1.0)
            trunk_angle = float(np.degrees(np.arccos(cos_t)))
        else:
            trunk_angle = None
        sh_conf = min(_kp_conf("left_shoulder", keypoints_conf), _kp_conf("right_shoulder", keypoints_conf))
        hip_c = min(_kp_conf("left_hip", keypoints_conf), _kp_conf("right_hip", keypoints_conf))
        trunk_status = "ok" if min(sh_conf, hip_c) >= CONFIDENCE_THRESHOLD else "low_confidence"
    else:
        trunk_angle = None
        trunk_status = "missing"

    result["trunk_inclination"] = {
        "value": trunk_angle,
        "cn_name": SPECIAL_ANGLES["trunk_inclination"],
        "status": trunk_status,
    }

    # neck angle: nose -> shoulder_mid vs vertical
    nose = _kp_xy("nose", keypoints_xy)
    if nose is not None and sh_mid is not None:
        neck_vec = nose - sh_mid
        norm_n = np.linalg.norm(neck_vec)
        vertical = np.array([0.0, -1.0])
        if norm_n > 1e-6:
            cos_n = np.clip(np.dot(neck_vec, vertical) / norm_n, -1.0, 1.0)
            neck_angle = float(np.degrees(np.arccos(cos_n)))
        else:
            neck_angle = None
        neck_status = "ok" if _kp_conf("nose", keypoints_conf) >= CONFIDENCE_THRESHOLD else "low_confidence"
    else:
        neck_angle = None
        neck_status = "missing"

    result["neck_angle"] = {
        "value": neck_angle,
        "cn_name": SPECIAL_ANGLES["neck_angle"],
        "status": neck_status,
    }

    return {"joint_angles": result}
