from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from mtcnn import MTCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect single-face images and crop face regions using MTCNN."
    )
    parser.add_argument(
        "--input-dir",
        default="src/images_v2",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        default="src/images_v2_pre_processing",
        help="Folder to save cropped face images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (for quick testing).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.95,
        help="Minimum confidence score for a face detection.",
    )
    parser.add_argument(
        "--expand-x",
        type=float,
        default=0.25,
        help="Horizontal expansion ratio for face crop (include both ears).",
    )
    parser.add_argument(
        "--expand-y-top",
        type=float,
        default=0.20,
        help="Top expansion ratio for face crop (include more forehead).",
    )
    parser.add_argument(
        "--expand-y-bottom",
        type=float,
        default=0.20,
        help="Bottom expansion ratio for face crop (include more chin/jaw).",
    )
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    )


def keep_ratio_with_padding(image: np.ndarray, size: int = 256) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), 255, dtype=np.uint8)

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x_off = (size - new_w) // 2
    y_off = (size - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def expand_and_clip_bbox(
    bbox: Iterable[float],
    image_w: int,
    image_h: int,
    expand_x: float,
    expand_y_top: float,
    expand_y_bottom: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox

    x_pad = w * expand_x
    y_pad_top = h * expand_y_top
    y_pad_bottom = h * expand_y_bottom

    x1 = max(0, int(round(x - x_pad)))
    y1 = max(0, int(round(y - y_pad_top)))
    x2 = min(image_w, int(round(x + w + x_pad)))
    y2 = min(image_h, int(round(y + h + y_pad_bottom)))
    return x1, y1, x2, y2


def filter_valid_faces(
    detections: List[dict], score_threshold: float
) -> List[dict]:
    return [det for det in detections if det.get("confidence", 0.0) >= score_threshold]


def process_images(
    input_dir: Path,
    output_dir: Path,
    limit: Optional[int],
    score_threshold: float,
    expand_x: float,
    expand_y_top: float,
    expand_y_bottom: float,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = MTCNN()
    images = list_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    if limit is not None:
        if limit <= 0:
            print("[WARN] --limit must be > 0, no image processed.")
            return
        images = images[:limit]

    processed = 0
    skipped_no_face = 0
    skipped_multi_face = 0
    skipped_invalid = 0

    for image_path in images:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[WARN] Cannot read image: {image_path.name}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)
        valid_faces = filter_valid_faces(detections, score_threshold)

        if len(valid_faces) == 0:
            print(f"[INFO] No face found: {image_path.name}")
            skipped_no_face += 1
            continue

        if len(valid_faces) > 1:
            print(f"[SKIP] Multiple faces: {image_path.name}")
            skipped_multi_face += 1
            continue

        det = valid_faces[0]
        x1, y1, x2, y2 = expand_and_clip_bbox(
            det["box"],
            image_rgb.shape[1],
            image_rgb.shape[0],
            expand_x,
            expand_y_top,
            expand_y_bottom,
        )
        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Invalid crop box: {image_path.name}")
            skipped_invalid += 1
            continue

        crop_rgb = image_rgb[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        crop_bgr = keep_ratio_with_padding(crop_bgr, size=256)

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), crop_bgr)
        processed += 1
        print(f"[DONE] Saved: {out_path.name}")

    print("\n=== Summary ===")
    print(f"Input images checked: {len(images)}")
    print(f"Saved (single face): {processed}")
    print(f"Skipped (no face): {skipped_no_face}")
    print(f"Skipped (multiple faces): {skipped_multi_face}")
    print(f"Skipped (invalid crop): {skipped_invalid}")
    print(f"Output folder: {output_dir}")


def main() -> None:
    args = parse_args()
    process_images(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        score_threshold=args.score_threshold,
        expand_x=args.expand_x,
        expand_y_top=args.expand_y_top,
        expand_y_bottom=args.expand_y_bottom,
    )


if __name__ == "__main__":
    main()
