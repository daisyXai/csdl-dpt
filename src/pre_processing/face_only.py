from __future__ import annotations

"""
Face foreground extraction using MTCNN.

Install dependencies:
    pip install mtcnn opencv-python numpy tqdm
"""

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm


def list_images(input_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def best_face(detections: List[dict], score_threshold: float) -> Optional[dict]:
    valid = [det for det in detections if det.get("confidence", 0.0) >= score_threshold]
    if not valid:
        return None
    return max(valid, key=lambda det: det.get("confidence", 0.0))


def create_face_mask(
    image_shape: tuple[int, int, int],
    bbox: list[float],
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
    blur_ksize: int,
) -> np.ndarray:
    height, width = image_shape[:2]
    x, y, w, h = bbox

    pad_x = w * expand_x
    pad_top = h * expand_top
    pad_bottom = h * expand_bottom

    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_top)))
    x2 = min(width, int(round(x + w + pad_x)))
    y2 = min(height, int(round(y + h + pad_bottom)))

    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask

    # Ellipse mask cho chuyển tiếp mềm, giảm viền cứng.
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    axis_x = max(1, (x2 - x1) // 2)
    axis_y = max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

    if blur_ksize > 1:
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def apply_white_background(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(image_bgr, 255)
    blended = image_bgr.astype(np.float32) * alpha + white.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    limit: Optional[int],
    score_threshold: float,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
    blur_ksize: int,
    output_size: int,
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    images = list_images(input_dir)
    if not images:
        print(f"[INFO] No images found in: {input_dir}")
        return

    if limit is not None:
        images = images[: max(0, limit)]

    detector = MTCNN()
    failed_files: List[str] = []
    success_count = 0

    for image_path in tqdm(images, desc="Removing background", unit="img"):
        try:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                raise ValueError("Cannot read image")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(image_rgb)
            det = best_face(detections, score_threshold=score_threshold)
            if det is None:
                raise ValueError("No valid face found")

            mask = create_face_mask(
                image_shape=image_bgr.shape,
                bbox=det["box"],
                expand_x=expand_x,
                expand_top=expand_top,
                expand_bottom=expand_bottom,
                blur_ksize=blur_ksize,
            )
            if np.count_nonzero(mask) == 0:
                raise ValueError("Generated mask is empty")

            output_bgr = apply_white_background(image_bgr=image_bgr, mask=mask)
            output_bgr = cv2.resize(output_bgr, (output_size, output_size), interpolation=cv2.INTER_AREA)

            save_path = output_dir / image_path.name
            ok = cv2.imwrite(str(save_path), output_bgr)
            if not ok:
                raise RuntimeError("cv2.imwrite failed")
            success_count += 1
        except Exception as exc:
            failed_files.append(f"{image_path.name}: {exc}")

    print("\n=== Done ===")
    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Total images: {len(images)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed_files)}")
    if failed_files:
        print("\nFailed files:")
        for item in failed_files:
            print(f" - {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove background using MTCNN face detection.")
    parser.add_argument("--input-dir", default="src/images_v2_pre_processing", help="Input image folder.")
    parser.add_argument("--output-dir", default="src/images_v2_face_only", help="Output image folder.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images.")
    parser.add_argument("--score-threshold", type=float, default=0.90, help="Minimum face confidence.")
    parser.add_argument("--expand-x", type=float, default=0.35, help="Horizontal bbox expansion ratio.")
    parser.add_argument("--expand-top", type=float, default=0.35, help="Top bbox expansion ratio.")
    parser.add_argument("--expand-bottom", type=float, default=0.45, help="Bottom bbox expansion ratio.")
    parser.add_argument("--blur-ksize", type=int, default=41, help="Gaussian blur kernel for soft edge.")
    parser.add_argument("--output-size", type=int, default=256, help="Final output square size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_folder(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        score_threshold=args.score_threshold,
        expand_x=args.expand_x,
        expand_top=args.expand_top,
        expand_bottom=args.expand_bottom,
        blur_ksize=args.blur_ksize,
        output_size=args.output_size,
    )


if __name__ == "__main__":
    main()
