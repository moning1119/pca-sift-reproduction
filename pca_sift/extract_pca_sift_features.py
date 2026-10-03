

import os
import cv2
import gc
import numpy as np
from multiprocessing import Pool, cpu_count
from pca_sift import pca_sift
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================
# Configuration
# =====================
IMAGE_DIR = "./data/train_pca"
OUTPUT_DIR = "./data/pca_sift_features"

MAX_KEYPOINTS_PER_IMAGE = 500
RANDOM_SEED = 42

# ⚠️ 建議從 2 或 3 開始，不要直接用 cpu_count()
NUM_WORKERS = os.cpu_count() // 2

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# Utility
# =====================
def select_random_keypoints(keypoints, max_kp, seed):
    np.random.seed(seed)
    n = len(keypoints)
    if n <= max_kp:
        return keypoints

    indices = np.random.choice(n, max_kp, replace=False)
    indices = np.sort(indices)
    return [keypoints[i] for i in indices]


# =====================
# Worker function
# =====================
def process_single_image(args):
    idx, fname = args

    img_path = os.path.join(IMAGE_DIR, fname)
    out_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(fname)[0] + ".npy"
    )

    logger.info(f"[Worker] Processing {fname}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Failed to load {fname}")
        return

    # -----------------------------
    # Compute keypoints + Gaussian
    # -----------------------------
    keypoints, gaussian_images = pca_sift.computeKeypoints(img)

    if keypoints is None or len(keypoints) == 0:
        logger.warning(f"No keypoints in {fname}")
        return

    # -----------------------------
    # Random selection
    # -----------------------------
    keypoints = select_random_keypoints(
        keypoints,
        MAX_KEYPOINTS_PER_IMAGE,
        seed=RANDOM_SEED + idx
    )

    # -----------------------------
    # Compute PCA-SIFT descriptors
    # -----------------------------
    descriptors = pca_sift.compute_pca_sift_descriptors(
        keypoints, gaussian_images
    )

    if descriptors is None or len(descriptors) == 0:
        logger.warning(f"No descriptors in {fname}")
        return

    descriptors = descriptors.astype(np.float32)

    # Gaussian pyramid no longer needed
    del gaussian_images
    gc.collect()

    # -----------------------------
    # Save
    # -----------------------------
    np.save(out_path, descriptors)
    logger.info(
        f"[Worker] Saved {fname}: "
        f"{descriptors.shape[0]} × {descriptors.shape[1]}"
    )

    # -----------------------------
    # Cleanup
    # -----------------------------
    del img, keypoints, descriptors
    gc.collect()


# =====================
# Main
# =====================
def main():
    # image_files = sorted([
    #     f for f in os.listdir(IMAGE_DIR)
    #     if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
    # ])
    image_files = [
        "12.jpg","14.jpg","32.jpg"
    ]

    logger.info(f"Found {len(image_files)} images")
    logger.info(f"Using {NUM_WORKERS} worker processes")

    tasks = list(enumerate(image_files))

    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(process_single_image, tasks)

    logger.info("Multiprocessing extraction completed.")


if __name__ == "__main__":
    main()
