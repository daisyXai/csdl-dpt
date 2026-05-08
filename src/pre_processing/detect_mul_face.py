from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and crop faces from images using MTCNN."
    )
    parser.add_argument(
        "--input-dir",
        default="src/images",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--prefix",
        default="child",
        help="Prefix for output folders and files.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Target output size (size x size).",
    )
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def keep_ratio_with_padding(image: np.ndarray, size: int = 128) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def clip_bbox(
    bbox: Iterable[float], image_w: int, image_h: int
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(image_w, int(x + w))
    y2 = min(image_h, int(y + h))
    return x1, y1, x2, y2


def process_images(input_dir: Path, prefix: str, size: int) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    detector = MTCNN()
    images = list_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    for j, image_path in enumerate(images, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Cannot read image: {image_path.name}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)
        if not detections:
            print(f"[INFO] No face found: {image_path.name}")
            continue

        face_count = 0
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2 = clip_bbox(det["box"], image_rgb.shape[1], image_rgb.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop_rgb = image_rgb[y1:y2, x1:x2]
            crop_rgb = keep_ratio_with_padding(crop_rgb, size=size)
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

            out_dir = input_dir / f"{prefix}_{j}_{i}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{prefix}_{j}_{i}.jpg"
            cv2.imwrite(str(out_dir / out_name), crop_bgr)
            face_count += 1

        print(f"[DONE] {image_path.name}: {face_count} face(s) saved")


def main() -> None:
    args = parse_args()
    process_images(Path(args.input_dir), args.prefix, args.size)


if __name__ == "__main__":
    main()
