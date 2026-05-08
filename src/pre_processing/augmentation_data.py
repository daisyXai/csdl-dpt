from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 4 augmentations for each image in child_* folders."
    )
    parser.add_argument(
        "--images-dir",
        default="src/images",
        help="Root images directory containing child_* folders.",
    )
    parser.add_argument(
        "--translate-px",
        type=int,
        default=12,
        help="Translation pixels for left/right shift (recommended <= 15).",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.1,
        help="Zoom factor in range [0.9, 1.1].",
    )
    return parser.parse_args()


def list_child_dirs(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.is_dir() and p.name.startswith("child")])


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def translate_image(image: np.ndarray, dx: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def zoom_image_keep_size(image: np.ndarray, factor: float) -> np.ndarray:
    h, w = image.shape[:2]
    if factor <= 0:
        return image.copy()

    if abs(factor - 1.0) < 1e-6:
        return image.copy()

    if factor > 1.0:
        # Zoom in: crop center then resize back
        crop_w = max(1, int(w / factor))
        crop_h = max(1, int(h / factor))
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Zoom out: resize smaller and pad black
    new_w = max(1, int(w * factor))
    new_h = max(1, int(h * factor))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x_off = (w - new_w) // 2
    y_off = (h - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def build_output_name(image_path: Path, aug_tag: str) -> str:
    return f"{image_path.stem}_{aug_tag}{image_path.suffix.lower()}"


def is_augmented_file(file_name: str) -> bool:
    tags = ("rot_p10", "rot_m10", "trans", "zoom")
    return any(tag in file_name for tag in tags)


def augment_one_image(image_path: Path, translate_px: int, zoom_factor: float) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Skip unreadable file: {image_path}")
        return 0

    # Alternate left/right translation by file name hash to diversify.
    dx = translate_px if (hash(image_path.name) % 2 == 0) else -translate_px

    augmented = {
        "rot_p10": rotate_image(image, 10),
        "rot_m10": rotate_image(image, -10),
        "trans": translate_image(image, dx),
        "zoom": zoom_image_keep_size(image, zoom_factor),
    }

    count = 0
    for tag, aug_img in augmented.items():
        out_name = build_output_name(image_path, tag)
        out_path = image_path.parent / out_name
        if cv2.imwrite(str(out_path), aug_img):
            count += 1
    return count


def run_augmentation(images_dir: Path, translate_px: int, zoom_factor: float) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if translate_px < 0 or translate_px > 15:
        raise ValueError("translate-px must be in range [0, 15].")
    if zoom_factor < 0.9 or zoom_factor > 1.1:
        raise ValueError("zoom-factor must be in range [0.9, 1.1].")

    child_dirs = list_child_dirs(images_dir)
    if not child_dirs:
        print(f"No child_* folders found in: {images_dir}")
        return

    total_source = 0
    total_generated = 0

    for child_dir in child_dirs:
        images = list_images(child_dir)
        source_images = [img for img in images if not is_augmented_file(img.stem)]
        if not source_images:
            print(f"[INFO] No source image in {child_dir.name}")
            continue

        generated_in_dir = 0
        for image_path in source_images:
            generated_in_dir += augment_one_image(image_path, translate_px, zoom_factor)
            total_source += 1

        total_generated += generated_in_dir
        print(
            f"[DONE] {child_dir.name}: {len(source_images)} source -> {generated_in_dir} augmented"
        )

    print(
        f"[SUMMARY] Processed {total_source} source image(s), generated {total_generated} image(s)."
    )


def main() -> None:
    args = parse_args()
    run_augmentation(
        images_dir=Path(args.images_dir),
        translate_px=args.translate_px,
        zoom_factor=args.zoom_factor,
    )


if __name__ == "__main__":
    main()
