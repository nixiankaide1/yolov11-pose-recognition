"""Biomechanics overlay: skeleton, joint angle arcs, COM marker on image."""

import cv2
import numpy as np
import math
from biomechanics.skeleton import SKELETON_CONNECTIONS, KP, get_connection_group, CONNECTION_COLORS
from biomechanics.angles import compute_all_angles
from biomechanics.com import compute_center_of_mass


def _draw_angle_arc(img, p1, p2, p3, angle_value, color=(255, 255, 0)):
    """Draw a small arc at p2 showing the joint angle, with the value in degrees."""
    if angle_value is None:
        return

    v1 = np.array(p1, dtype=np.float64) - np.array(p2, dtype=np.float64)
    v2 = np.array(p3, dtype=np.float64) - np.array(p2, dtype=np.float64)

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return

    v1u = v1 / n1
    v2u = v2 / n2

    radius = 25
    start_angle = math.atan2(-v1u[1], v1u[0])
    end_angle = math.atan2(-v2u[1], v2u[0])

    # arcs direction
    if end_angle < start_angle:
        end_angle += 2 * math.pi

    center = tuple(p2.astype(int))
    cv2.ellipse(img, center, (radius, radius), 0,
                math.degrees(start_angle), math.degrees(end_angle),
                color, 2)
    # angle label midpoint
    mid_angle = (start_angle + end_angle) / 2
    label_pos = (
        int(p2[0] + (radius + 18) * math.cos(mid_angle)),
        int(p2[1] - (radius + 18) * math.sin(mid_angle)),
    )
    cv2.putText(img, f"{angle_value:.0f}", label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_biomechanics_overlay(image, keypoints_xy, keypoints_conf, show_angles=True, show_com=True):
    """
    Draw skeleton, angle arcs, and COM on a BGR image.

    Args:
        image: BGR numpy array
        keypoints_xy: (17, 2) ndarray
        keypoints_conf: (17,) ndarray
        show_angles: draw angle arcs with values
        show_com: draw center of mass marker

    Returns:
        annotated BGR image
    """
    img = image.copy()

    # draw skeleton
    for conn in SKELETON_CONNECTIONS:
        i1, i2 = conn
        if i1 >= len(keypoints_xy) or i2 >= len(keypoints_xy):
            continue
        pt1 = tuple(keypoints_xy[i1].astype(int))
        pt2 = tuple(keypoints_xy[i2].astype(int))
        group = get_connection_group(conn)
        color = CONNECTION_COLORS.get(group, (200, 200, 200))
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

    # draw keypoints
    for i, kp_xy in enumerate(keypoints_xy):
        pt = tuple(kp_xy.astype(int))
        conf = keypoints_conf[i] if i < len(keypoints_conf) else 1.0
        color = (0, 255, 0) if conf >= 0.3 else (0, 0, 255)
        cv2.circle(img, pt, 4, color, -1, cv2.LINE_AA)

    # draw angles
    if show_angles:
        angles_data = compute_all_angles(keypoints_xy, keypoints_conf)
        for joint_name, info in angles_data["joint_angles"].items():
            if info["status"] != "ok" or info["value"] is None:
                continue
            # resolve positions from JOINT_DEFS
            from biomechanics.skeleton import JOINT_DEFS
            if joint_name in JOINT_DEFS:
                prox, center, dist, _ = JOINT_DEFS[joint_name]
                from biomechanics.com import _get_point
                p1 = _get_point(prox, keypoints_xy)
                p2 = _get_point(center, keypoints_xy)
                if dist in ("left_toe_approx", "right_toe_approx"):
                    from biomechanics.skeleton import get_toe_approx
                    ankle_name = "left_ankle" if "left" in joint_name else "right_ankle"
                    knee_name = "left_knee" if "left" in joint_name else "right_knee"
                    ankle = _get_point(ankle_name, keypoints_xy)
                    knee = _get_point(knee_name, keypoints_xy)
                    p3 = get_toe_approx(ankle, knee) if ankle is not None and knee is not None else None
                else:
                    p3 = _get_point(dist, keypoints_xy)
                if p1 is not None and p2 is not None and p3 is not None:
                    _draw_angle_arc(img, p1, p2, p3, info["value"])

        # trunk inclination label
        for sn in ["trunk_inclination", "neck_angle"]:
            info = angles_data["joint_angles"].get(sn)
            if info and info["status"] == "ok" and info["value"] is not None:
                # place label near shoulder midpoint
                from biomechanics.com import _get_point
                sh_mid = _get_point("shoulder_midpoint", keypoints_xy)
                if sh_mid is not None:
                    offset = -40 if sn == "neck_angle" else 40
                    pos = (int(sh_mid[0]) + 60, int(sh_mid[1]) + offset)
                    cv2.putText(img, f"{info['cn_name']}:{info['value']:.1f}", pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # draw COM
    if show_com:
        com_data = compute_center_of_mass(keypoints_xy)
        com = com_data.get("com")
        if com is not None:
            com_pt = (int(com[0]), int(com[1]))
            cv2.drawMarker(img, com_pt, (0, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
            cv2.circle(img, com_pt, 8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, "COM", (com_pt[0] + 12, com_pt[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return img
