"""Skeleton topology, keypoint indices, joint angle definitions."""

import numpy as np

# 0-based keypoint index constants
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Skeleton connections (0-based) for drawing
SKELETON_CONNECTIONS = [
    # left leg (COCO left = image left side)
    (15, 13), (13, 11),
    # right leg
    (16, 14), (14, 12),
    # hips cross
    (11, 12),
    # trunk sides
    (5, 11), (6, 12),
    # shoulders cross
    (5, 6),
    # left arm
    (5, 7), (7, 9),
    # right arm
    (6, 8), (8, 10),
    # face
    (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
]

# Color for each connection group
CONNECTION_COLORS = {
    "left_leg": (255, 107, 53),      # orange
    "right_leg": (78, 205, 196),     # cyan
    "left_arm": (255, 107, 53),
    "right_arm": (78, 205, 196),
    "trunk": (149, 165, 166),        # gray
    "face": (247, 220, 111),         # yellow
}

def get_connection_group(idx_pair):
    a, b = idx_pair
    a, b = min(a, b), max(a, b)
    if (a, b) in [(15, 13), (13, 11)]:
        return "left_leg"
    if (a, b) in [(16, 14), (14, 12)]:
        return "right_leg"
    if (a, b) in [(5, 7), (7, 9)]:
        return "left_arm"
    if (a, b) in [(6, 8), (8, 10)]:
        return "right_arm"
    if (a, b) in [(5, 11), (6, 12), (5, 6), (11, 12)]:
        return "trunk"
    return "face"


# Joint definitions: (proximal_kp, center_kp, distal_kp, cn_name)
JOINT_DEFS = {
    "left_elbow":   ("left_shoulder", "left_elbow", "left_wrist", "左肘"),
    "right_elbow":  ("right_shoulder", "right_elbow", "right_wrist", "右肘"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip", "左肩"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip", "右肩"),
    "left_hip":     ("left_shoulder", "left_hip", "left_knee", "左髋"),
    "right_hip":    ("right_shoulder", "right_hip", "right_knee", "右髋"),
    "left_knee":    ("left_hip", "left_knee", "left_ankle", "左膝"),
    "right_knee":   ("right_hip", "right_knee", "right_ankle", "右膝"),
    "left_ankle":   ("left_knee", "left_ankle", "left_toe_approx", "左踝"),
    "right_ankle":  ("right_knee", "right_ankle", "right_toe_approx", "右踝"),
}

# Special angles
SPECIAL_ANGLES = {
    "trunk_inclination": "躯干倾角",
    "neck_angle": "颈部角度",
}


def get_toe_approx(ankle_xy, knee_xy):
    """Approximate toe position from ankle and knee."""
    vec = ankle_xy - knee_xy
    toe = ankle_xy + vec * 0.4
    toe[1] += 10  # slight downward offset in image coords
    return toe


def get_hand_end(wrist_xy, elbow_xy):
    """Approximate hand endpoint from wrist and elbow."""
    return wrist_xy + (wrist_xy - elbow_xy) * 0.3


def get_midpoint(kp1_xy, kp2_xy):
    """Midpoint between two keypoints."""
    return (kp1_xy + kp2_xy) / 2.0
