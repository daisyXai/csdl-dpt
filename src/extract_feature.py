from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import psycopg2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    from .postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB
except ImportError:
    from postgres import DB_HOST, DB_PASSWORD, DB_PORT, DB_USER, TARGET_DB


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract multi-group handcrafted face features and save to PostgreSQL."
    )
    parser.add_argument(
        "--images-dir",
        default="src/images_v2_face_only",
        help="Folder containing input face images.",
    )
    parser.add_argument("--face-size", type=int, default=128, help="Aligned face size.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images to process (0 means all).",
    )
    parser.add_argument(
        "--landmarker-model",
        default="src/models/face_landmarker.task",
        help="Path to MediaPipe FaceLandmarker .task model file.",
    )
    return parser.parse_args()


def list_child_images(images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for file in sorted(images_dir.rglob("*")):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
            paths.append(file)
    return paths


def clamp01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def face_detect_bgr(image_bgr: np.ndarray, landmarker) -> Optional[Tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    lms = result.face_landmarks[0]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(w, int(max(xs)))
    y2 = min(h, int(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def estimate_eye_centers_from_landmarker(
    face_rgb: np.ndarray, landmarker
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]]:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
    res = landmarker.detect(mp_image)
    if not res.face_landmarks:
        return None

    lm = res.face_landmarks[0]
    h, w = face_rgb.shape[:2]

    # FaceMesh landmark indices around eyes + nose tip + mouth corners.
    left_eye_idx = [33, 133, 159, 145]
    right_eye_idx = [362, 263, 386, 374]
    nose_idx = 1
    mouth_left_idx = 61
    mouth_right_idx = 291

    def to_xy(i: int) -> np.ndarray:
        return np.array([float(lm[i].x) * w, float(lm[i].y) * h], dtype=np.float32)

    left_eye_pts = np.stack([to_xy(i) for i in left_eye_idx], axis=0)
    right_eye_pts = np.stack([to_xy(i) for i in right_eye_idx], axis=0)
    left_eye = left_eye_pts.mean(axis=0)
    right_eye = right_eye_pts.mean(axis=0)

    keypoints = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "nose": to_xy(nose_idx),
        "mouth_left": to_xy(mouth_left_idx),
        "mouth_right": to_xy(mouth_right_idx),
    }
    return left_eye, right_eye, keypoints


def rotate_image_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def resize_with_padding(image_bgr: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    scale = min(size / w, size / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def safe_patch(
    gray: np.ndarray,
    center: np.ndarray | Tuple[float, float],
    patch_w: float,
    patch_h: float,
) -> np.ndarray:
    h, w = gray.shape[:2]
    pw = max(1, int(round(patch_w)))
    ph = max(1, int(round(patch_h)))
    cx, cy = float(center[0]), float(center[1])

    x1 = int(round(cx - pw / 2.0))
    y1 = int(round(cy - ph / 2.0))
    x2 = x1 + pw
    y2 = y1 + ph

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w, x2)
    src_y2 = min(h, y2)

    patch = np.zeros((ph, pw), dtype=gray.dtype)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return patch

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    patch[
        dst_y1 : dst_y1 + (src_y2 - src_y1),
        dst_x1 : dst_x1 + (src_x2 - src_x1),
    ] = gray[src_y1:src_y2, src_x1:src_x2]
    return patch


def patch_entropy(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    hist = cv2.calcHist([patch.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        return 0.0
    p = hist[hist > 0.0] / total
    entropy = -float(np.sum(p * np.log2(p)))
    return clamp01(entropy / 8.0)


def edge_density(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    edges = cv2.Canny(patch.astype(np.uint8), 50, 150)
    return clamp01(float(np.mean(edges > 0)))


def gradient_mean(patch: np.ndarray) -> float:
    if patch.size == 0:
        return 0.0
    patch_f = patch.astype(np.float32)
    sobel_x = cv2.Sobel(patch_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(patch_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    return clamp01(float(np.mean(mag)) / 255.0)


def symmetry_score(left: np.ndarray, right: Optional[np.ndarray] = None) -> float:
    if left.size == 0:
        return 0.0
    if right is None:
        mid = left.shape[1] // 2
        left_half = left[:, :mid]
        right_half = left[:, left.shape[1] - mid :]
    else:
        left_half = left
        right_half = right

    if left_half.size == 0 or right_half.size == 0:
        return 0.0

    right_half = np.fliplr(right_half)
    min_h = min(left_half.shape[0], right_half.shape[0])
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_h == 0 or min_w == 0:
        return 0.0

    l = left_half[:min_h, :min_w].astype(np.float32)
    r = right_half[:min_h, :min_w].astype(np.float32)
    diff = np.abs(l - r)
    return clamp01(1.0 - float(np.mean(diff)) / 255.0)


def _triangle_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a.astype(np.float32) - b.astype(np.float32)
    bc = c.astype(np.float32) - b.astype(np.float32)
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return 0.0
    cos_v = float(np.dot(ba, bc) / denom)
    return math.degrees(math.acos(float(np.clip(cos_v, -1.0, 1.0))))


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def _compute_face_geometry(
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    pts: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
) -> Tuple[List[float], Dict[str, float]]:
    h, w = image_shape
    landmarks = np.stack(
        [pts["left_eye"], pts["right_eye"], pts["nose"], pts["mouth_left"], pts["mouth_right"]],
        axis=0,
    ).astype(np.float32)
    lx1, ly1 = np.min(landmarks, axis=0)
    lx2, ly2 = np.max(landmarks, axis=0)

    fx1 = max(0.0, float(lx1) - 0.26 * w)
    fx2 = min(float(w), float(lx2) + 0.26 * w)
    fy1 = max(0.0, float(ly1) - 0.34 * h)
    fy2 = min(float(h), float(ly2) + 0.34 * h)
    face_w = max(1.0, fx2 - fx1)
    face_h = max(1.0, fy2 - fy1)
    face_center_x = 0.5 * (fx1 + fx2)
    face_area = face_w * face_h

    eye_line_y = 0.5 * (float(left_eye[1]) + float(right_eye[1]))
    eye_dist = float(np.linalg.norm(left_eye - right_eye))
    left_eye_nose = float(np.linalg.norm(left_eye - pts["nose"]))
    right_eye_nose = float(np.linalg.norm(right_eye - pts["nose"]))
    eye_nose_mean = 0.5 * (left_eye_nose + right_eye_nose)
    mouth_center = 0.5 * (pts["mouth_left"] + pts["mouth_right"])
    nose_mouth = float(np.linalg.norm(pts["nose"] - mouth_center))
    mouth_width = float(np.linalg.norm(pts["mouth_left"] - pts["mouth_right"]))

    forehead_ratio = (eye_line_y - fy1) / face_h
    jaw_ratio = mouth_width / max(eye_dist, 1e-6)
    chin_ratio = (fy2 - float(pts["nose"][1])) / face_h
    eye_vertical_offset = abs(float(left_eye[1]) - float(right_eye[1])) / face_h
    nose_horizontal_offset = abs(float(pts["nose"][0]) - face_center_x) / face_w
    mouth_horizontal_offset = abs(float(mouth_center[0]) - face_center_x) / face_w
    landmark_asymmetry = float(
        np.mean(
            [
                abs(float(left_eye[1]) - float(right_eye[1])) / face_h,
                abs(float(pts["nose"][0]) - face_center_x) / face_w,
                abs(float(mouth_center[0]) - face_center_x) / face_w,
                abs(left_eye_nose - right_eye_nose) / face_h,
            ]
        )
    )

    features = [
        clamp01(eye_dist / face_w),
        clamp01(left_eye_nose / face_h),
        clamp01(right_eye_nose / face_h),
        clamp01(nose_mouth / face_h),
        clamp01(mouth_width / face_w),
        clamp01((face_h / face_w) / 2.0),
        clamp01(forehead_ratio),
        clamp01(jaw_ratio / 2.0),
        clamp01(chin_ratio),
        clamp01(eye_vertical_offset),
        clamp01(nose_horizontal_offset),
        clamp01(mouth_horizontal_offset),
        clamp01(_triangle_angle(left_eye, pts["nose"], right_eye) / 180.0),
        clamp01(_triangle_angle(pts["mouth_left"], pts["nose"], pts["mouth_right"]) / 180.0),
    ]
    return features, {
        "fx1": float(fx1),
        "fy1": float(fy1),
        "fx2": float(fx2),
        "fy2": float(fy2),
        "face_w": float(face_w),
        "face_h": float(face_h),
        "face_area": float(face_area),
        "eye_dist": float(eye_dist),
        "mouth_width": float(mouth_width),
        "nose_mouth": float(nose_mouth),
        "mouth_center_x": float(mouth_center[0]),
        "mouth_center_y": float(mouth_center[1]),
        "face_center_x": float(face_center_x),
        "eye_line_y": float(eye_line_y),
        "jaw_ratio": float(jaw_ratio),
        "landmark_asymmetry": float(landmark_asymmetry),
    }


def _extract_texture_features(face_gray: np.ndarray, geom_ctx: Dict[str, float], pts: Dict[str, np.ndarray]) -> List[float]:
    face_blur = cv2.GaussianBlur(face_gray, (3, 3), 0)
    local_mean = cv2.blur(face_blur.astype(np.float32), (5, 5))
    local_sq_mean = cv2.blur(face_blur.astype(np.float32) ** 2, (5, 5))
    local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)
    local_std = np.sqrt(local_var).astype(np.float32)

    face_w = geom_ctx["face_w"]
    face_h = geom_ctx["face_h"]
    mouth_center = np.array([geom_ctx["mouth_center_x"], geom_ctx["mouth_center_y"]], dtype=np.float32)
    eye_center = 0.5 * (pts["left_eye"] + pts["right_eye"])
    eye_patch = safe_patch(face_blur, eye_center, 0.62 * face_w, 0.22 * face_h)
    nose_patch = safe_patch(face_blur, pts["nose"], 0.28 * face_w, 0.24 * face_h)
    mouth_patch = safe_patch(face_blur, mouth_center, 0.56 * face_w, 0.22 * face_h)

    lap = cv2.Laplacian(face_blur.astype(np.float32), cv2.CV_32F)
    return [
        patch_entropy(eye_patch),                                 # 1
        patch_entropy(nose_patch),                                # 2
        patch_entropy(mouth_patch),                               # 3
        patch_entropy(face_blur),                                 # 4
        clamp01(float(np.var(eye_patch)) / (255.0 * 255.0)),      # 5
        clamp01(float(np.var(nose_patch)) / (255.0 * 255.0)),     # 6
        clamp01(float(np.var(mouth_patch)) / (255.0 * 255.0)),    # 7
        clamp01(float(np.mean(local_std)) / 255.0),               # 8
        clamp01(float(np.std(local_std)) / 255.0),                # 9
        clamp01(float(np.mean(np.abs(lap))) / 255.0),             # 10
    ]


def _extract_color_features(face_bgr: np.ndarray) -> List[float]:
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    hist_h = cv2.calcHist([h_channel], [0], None, [3], [0, 180]).ravel().astype(np.float32)
    hist_s = cv2.calcHist([s_channel], [0], None, [3], [0, 256]).ravel().astype(np.float32)
    hist_v = cv2.calcHist([v_channel], [0], None, [3], [0, 256]).ravel().astype(np.float32)
    hist_h = hist_h / (float(hist_h.sum()) + 1e-6)
    hist_s = hist_s / (float(hist_s.sum()) + 1e-6)
    hist_v = hist_v / (float(hist_v.sum()) + 1e-6)
    mean_h = clamp01(float(np.mean(h_channel)) / 179.0)
    mean_s = clamp01(float(np.mean(s_channel)) / 255.0)
    mean_v = clamp01(float(np.mean(v_channel)) / 255.0)
    return hist_h.tolist() + hist_s.tolist() + hist_v.tolist() + [mean_h, mean_s, mean_v]


def _extract_structural_features(face_gray: np.ndarray, geom_ctx: Dict[str, float], pts: Dict[str, np.ndarray]) -> List[float]:
    face_w = geom_ctx["face_w"]
    face_h = geom_ctx["face_h"]
    mouth_center = np.array([geom_ctx["mouth_center_x"], geom_ctx["mouth_center_y"]], dtype=np.float32)
    eye_center = 0.5 * (pts["left_eye"] + pts["right_eye"])
    eye_patch = safe_patch(face_gray, eye_center, 0.62 * face_w, 0.22 * face_h)
    nose_patch = safe_patch(face_gray, pts["nose"], 0.28 * face_w, 0.24 * face_h)
    mouth_patch = safe_patch(face_gray, mouth_center, 0.56 * face_w, 0.22 * face_h)
    face_patch = safe_patch(face_gray, (geom_ctx["face_center_x"], 0.5 * (geom_ctx["fy1"] + geom_ctx["fy2"])), face_w, face_h)

    sobel_x = cv2.Sobel(face_patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(face_patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_features = [
        clamp01(float(np.mean(mag)) / 255.0),
        clamp01(float(np.std(mag)) / 255.0),
        clamp01(float(np.mean(np.abs(sobel_x))) / 255.0),
        clamp01(float(np.mean(np.abs(sobel_y))) / 255.0),
    ]

    canny_features = [
        edge_density(eye_patch),
        edge_density(nose_patch),
        edge_density(mouth_patch),
        edge_density(face_patch),
    ]

    edges = cv2.Canny(face_patch.astype(np.uint8), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contour_features = [0.0, 0.0]
    else:
        contour_lengths = [float(cv2.arcLength(c, True)) for c in contours]
        contour_areas = [float(cv2.contourArea(c)) for c in contours]
        contour_features = [
            clamp01(np.mean(contour_lengths) / 256.0),
            clamp01(np.mean(contour_areas) / float(face_patch.shape[0] * face_patch.shape[1] + 1e-6)),
        ]
    return sobel_features + canny_features + contour_features


def _extract_symmetry_statistics_features(
    gray: np.ndarray,
    left_eye: np.ndarray,
    right_eye: np.ndarray,
    pts: Dict[str, np.ndarray],
    geom_ctx: Dict[str, float],
) -> List[float]:
    face_w = geom_ctx["face_w"]
    face_h = geom_ctx["face_h"]
    fx1 = geom_ctx["fx1"]
    fy1 = geom_ctx["fy1"]
    fx2 = geom_ctx["fx2"]
    fy2 = geom_ctx["fy2"]
    mouth_center = np.array([geom_ctx["mouth_center_x"], geom_ctx["mouth_center_y"]], dtype=np.float32)
    face_center_x = geom_ctx["face_center_x"]
    eye_line_y = geom_ctx["eye_line_y"]

    face_patch = safe_patch(gray, (face_center_x, 0.5 * (fy1 + fy2)), face_w, face_h)
    eye_patch = safe_patch(gray, 0.5 * (left_eye + right_eye), 0.62 * face_w, 0.22 * face_h)
    mouth_patch = safe_patch(gray, mouth_center, 0.56 * face_w, 0.22 * face_h)
    left_eye_patch = safe_patch(gray, left_eye, 0.24 * face_w, 0.18 * face_h)
    right_eye_patch = safe_patch(gray, right_eye, 0.24 * face_w, 0.18 * face_h)
    left_cheek_c = np.array(
        [face_center_x - 0.24 * face_w, 0.5 * (float(pts["nose"][1]) + float(mouth_center[1]))],
        dtype=np.float32,
    )
    right_cheek_c = np.array(
        [face_center_x + 0.24 * face_w, 0.5 * (float(pts["nose"][1]) + float(mouth_center[1]))],
        dtype=np.float32,
    )
    left_cheek_patch = safe_patch(gray, left_cheek_c, 0.22 * face_w, 0.22 * face_h)
    right_cheek_patch = safe_patch(gray, right_cheek_c, 0.22 * face_w, 0.22 * face_h)
    forehead_c = np.array([face_center_x, fy1 + 0.5 * max(1.0, eye_line_y - fy1)], dtype=np.float32)
    forehead_patch = safe_patch(gray, forehead_c, 0.38 * face_w, 0.18 * face_h)
    del fx1, fx2  # kept for readability and future debug/expansion

    patch_vars = np.array(
        [
            float(np.var(eye_patch)) / (255.0 * 255.0),
            float(np.var(mouth_patch)) / (255.0 * 255.0),
            float(np.var(left_cheek_patch)) / (255.0 * 255.0),
            float(np.var(right_cheek_patch)) / (255.0 * 255.0),
            float(np.var(forehead_patch)) / (255.0 * 255.0),
        ],
        dtype=np.float32,
    )
    texture_consistency = 1.0 / (1.0 + float(np.std(patch_vars)) / (float(np.mean(patch_vars)) + 1e-6))

    intensity_mean = clamp01(float(np.mean(face_patch)) / 255.0)
    intensity_var = clamp01(float(np.var(face_patch)) / (255.0 * 255.0))
    intensity_contrast = clamp01(float(np.std(face_patch)) / 128.0)

    return [
        symmetry_score(face_patch),
        symmetry_score(left_eye_patch, right_eye_patch),
        symmetry_score(mouth_patch),
        intensity_mean,
        intensity_var,
        intensity_contrast,
    ]


def extract_face_features_from_aligned_face(
    face_bgr_128: np.ndarray, landmarker, orientation_angle_deg: float
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    face_rgb = cv2.cvtColor(face_bgr_128, cv2.COLOR_BGR2RGB)
    eye_info = estimate_eye_centers_from_landmarker(face_rgb, landmarker)
    if eye_info is None:
        return None, None
    left_eye, right_eye, pts = eye_info

    h, w = face_bgr_128.shape[:2]
    gray = cv2.cvtColor(face_bgr_128, cv2.COLOR_BGR2GRAY)

    geometry_features, geom_ctx = _compute_face_geometry(
        left_eye=left_eye,
        right_eye=right_eye,
        pts=pts,
        image_shape=(h, w),
    )
    texture_features = _extract_texture_features(gray, geom_ctx=geom_ctx, pts=pts)
    color_features = _extract_color_features(face_bgr_128)
    structural_features = _extract_structural_features(gray, geom_ctx=geom_ctx, pts=pts)
    symmetry_features = _extract_symmetry_statistics_features(
        gray=gray,
        left_eye=left_eye,
        right_eye=right_eye,
        pts=pts,
        geom_ctx=geom_ctx,
    )

    features = np.array(
        geometry_features
        + texture_features
        + color_features
        + structural_features
        + symmetry_features,
        dtype=np.float32,
    )
    features = np.clip(features, 0.0, 1.0).astype(np.float32)
    assert features.shape == (52,)

    metadata = {
        "feature_version": "v3_geometry_texture_color_structure_symmetry_no_hog",
        "feature_group_weights": {
            "geometry": 0.35,
            "texture": 0.20,
            "color": 0.20,
            "structural": 0.20,
            "symmetry_statistics": 0.05,
        },
        "group_dimensions": {
            "geometry": len(geometry_features),
            "texture": len(texture_features),
            "color": len(color_features),
            "structural": len(structural_features),
            "symmetry": len(symmetry_features),
            "total": int(features.shape[0]),
        },
        "face_bbox": (
            float(geom_ctx["fx1"]),
            float(geom_ctx["fy1"]),
            float(geom_ctx["fx2"]),
            float(geom_ctx["fy2"]),
        ),
        "face_w": float(geom_ctx["face_w"]),
        "face_h": float(geom_ctx["face_h"]),
        "eye_dist": float(geom_ctx["eye_dist"]),
        "mouth_width": float(geom_ctx["mouth_width"]),
        "nose_mouth": float(geom_ctx["nose_mouth"]),
        "orientation": float(orientation_angle_deg),
        "jaw_ratio": float(geom_ctx["jaw_ratio"]),
        "landmark_asymmetry": float(geom_ctx["landmark_asymmetry"]),
    }
    return features, metadata


def extract_30_features_from_aligned_face(
    face_bgr_128: np.ndarray, landmarker, orientation_angle_deg: float
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    # Backward-compatible wrapper name for old call sites.
    return extract_face_features_from_aligned_face(face_bgr_128, landmarker, orientation_angle_deg)


def align_face_and_extract(
    image_path: Path, landmarker, face_size: int
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return None

    face_box = face_detect_bgr(image_bgr, landmarker)
    if face_box is None:
        return None
    x1, y1, x2, y2 = face_box
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    eye_info = estimate_eye_centers_from_landmarker(face_rgb, landmarker)
    if eye_info is None:
        return None
    left_eye, right_eye, _ = eye_info

    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = math.degrees(math.atan2(dy, dx))
    aligned_crop = rotate_image_keep_size(face_crop, -angle)
    aligned_128 = resize_with_padding(aligned_crop, size=face_size)

    features, feature_metadata = extract_30_features_from_aligned_face(
        aligned_128, landmarker, orientation_angle_deg=angle
    )
    if features is None:
        return None

    metadata = {
        "detector": "mediapipe_tasks_face_landmarker",
        "aligned_size": face_size,
        "orientation_angle_deg": float(angle),
        "face_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "feature_debug": feature_metadata,
    }
    return features, metadata


def insert_feature_row(conn, image_name: str, image_location: str, metadata: Dict, features: np.ndarray) -> None:
    vector_literal = "[" + ",".join(f"{float(v):.8f}" for v in features.tolist()) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO images (image_name, image_metadata, vector, image_location)
            VALUES (%s, %s::jsonb, %s::vector, %s);
            """,
            (image_name, json.dumps(metadata), vector_literal, image_location),
        )


def ensure_landmarker_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    print(f"[INFO] Downloading model to {model_path} ...")
    urllib.request.urlretrieve(url, str(model_path))
    return model_path


def create_landmarker(model_path: Path, num_faces: int = 1):
    model_path = ensure_landmarker_model(model_path)
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_single_face_features_from_bgr(
    image_bgr: np.ndarray, landmarker, face_size: int = 128
) -> Tuple[Optional[np.ndarray], Optional[Dict], str, int]:
    """
    Extract features from one uploaded image.
    Returns: (features, metadata, status, face_count)
      - status in {"ok", "no_face", "multiple_faces", "landmark_failed"}
    """
    if image_bgr is None or image_bgr.size == 0:
        return None, None, "no_face", 0

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detect_result = landmarker.detect(mp_image)
    face_count = len(detect_result.face_landmarks)

    if face_count == 0:
        return None, None, "no_face", 0
    if face_count > 1:
        return None, None, "multiple_faces", face_count

    lms = detect_result.face_landmarks[0]
    h, w = image_bgr.shape[:2]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(w, int(max(xs)))
    y2 = min(h, int(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None, None, "landmark_failed", 1

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None, "landmark_failed", 1

    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    eye_info = estimate_eye_centers_from_landmarker(face_rgb, landmarker)
    if eye_info is None:
        return None, None, "landmark_failed", 1
    left_eye, right_eye, _ = eye_info

    dx = float(right_eye[0] - left_eye[0])
    dy = float(right_eye[1] - left_eye[1])
    angle = math.degrees(math.atan2(dy, dx))
    aligned_crop = rotate_image_keep_size(face_crop, -angle)
    aligned_128 = resize_with_padding(aligned_crop, size=face_size)
    features, feature_metadata = extract_30_features_from_aligned_face(
        aligned_128, landmarker, orientation_angle_deg=angle
    )
    if features is None:
        return None, None, "landmark_failed", 1

    metadata = {
        "detector": "mediapipe_tasks_face_landmarker",
        "aligned_size": face_size,
        "orientation_angle_deg": float(angle),
        "face_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "feature_debug": feature_metadata,
    }
    return features, metadata, "ok", 1


def run_pipeline(images_dir: Path, face_size: int, model_path: Path, limit: int = 0) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")

    image_paths = list_child_images(images_dir)
    if not image_paths:
        print(f"No images found in: {images_dir}")
        return
    if limit > 0:
        image_paths = image_paths[:limit]
        print(f"[INFO] limit={limit}, processing {len(image_paths)} images.")

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=TARGET_DB,
    )
    conn.autocommit = False

    success = 0
    skipped = 0
    try:
        with create_landmarker(model_path) as landmarker:
            for image_path in image_paths:
                result = align_face_and_extract(
                    image_path, landmarker, face_size=face_size
                )
                if result is None:
                    skipped += 1
                    print(f"[SKIP] {image_path}")
                    continue

                features, metadata = result
                insert_feature_row(
                    conn=conn,
                    image_name=image_path.name,
                    image_location=str(image_path.resolve()),
                    metadata=metadata,
                    features=features,
                )
                success += 1
                if success % 50 == 0:
                    conn.commit()
                    print(f"[INFO] committed {success} rows...")

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"[DONE] inserted: {success}, skipped: {skipped}, total: {len(image_paths)}")


def main() -> None:
    args = parse_args()
    run_pipeline(
        images_dir=Path(args.images_dir),
        face_size=args.face_size,
        model_path=Path(args.landmarker_model),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
