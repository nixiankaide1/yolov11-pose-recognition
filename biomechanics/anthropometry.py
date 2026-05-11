"""Dempster (1955) body segment parameters for center-of-mass estimation."""

# Each segment: (mass_fraction, com_fraction_from_proximal, cn_name)
# Mass fractions sum to 1.0
SEGMENT_PARAMS = {
    "head_neck":         {"mass": 0.081,  "com_ratio": 0.433, "cn_name": "头颈"},
    "trunk":             {"mass": 0.497,  "com_ratio": 0.450, "cn_name": "躯干"},
    "left_upper_arm":    {"mass": 0.028,  "com_ratio": 0.436, "cn_name": "左上臂"},
    "right_upper_arm":   {"mass": 0.028,  "com_ratio": 0.436, "cn_name": "右上臂"},
    "left_forearm":      {"mass": 0.016,  "com_ratio": 0.430, "cn_name": "左前臂"},
    "right_forearm":     {"mass": 0.016,  "com_ratio": 0.430, "cn_name": "右前臂"},
    "left_hand":         {"mass": 0.006,  "com_ratio": 0.506, "cn_name": "左手"},
    "right_hand":        {"mass": 0.006,  "com_ratio": 0.506, "cn_name": "右手"},
    "left_thigh":        {"mass": 0.100,  "com_ratio": 0.433, "cn_name": "左大腿"},
    "right_thigh":       {"mass": 0.100,  "com_ratio": 0.433, "cn_name": "右大腿"},
    "left_shank":        {"mass": 0.0465, "com_ratio": 0.433, "cn_name": "左小腿"},
    "right_shank":       {"mass": 0.0465, "com_ratio": 0.433, "cn_name": "右小腿"},
    "left_foot":         {"mass": 0.0145, "com_ratio": 0.429, "cn_name": "左脚"},
    "right_foot":        {"mass": 0.0145, "com_ratio": 0.429, "cn_name": "右脚"},
}

# Segment endpoint definitions: (proximal_kp_name, distal_kp_name, distal_is_approx)
# "approx" means the distal endpoint requires a helper function
SEGMENT_ENDPOINTS = {
    "head_neck":         ("shoulder_midpoint", "nose", False),
    "trunk":             ("hip_midpoint", "shoulder_midpoint", False),
    "left_upper_arm":    ("left_shoulder", "left_elbow", False),
    "right_upper_arm":   ("right_shoulder", "right_elbow", False),
    "left_forearm":      ("left_elbow", "left_wrist", False),
    "right_forearm":     ("right_elbow", "right_wrist", False),
    "left_hand":         ("left_wrist", "left_hand_end", True),
    "right_hand":        ("right_wrist", "right_hand_end", True),
    "left_thigh":        ("left_hip", "left_knee", False),
    "right_thigh":       ("right_hip", "right_knee", False),
    "left_shank":        ("left_knee", "left_ankle", False),
    "right_shank":       ("right_knee", "right_ankle", False),
    "left_foot":         ("left_ankle", "left_toe_approx", True),
    "right_foot":        ("right_ankle", "right_toe_approx", True),
}
