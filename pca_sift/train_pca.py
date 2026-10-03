import os
import glob
import numpy as np
import joblib
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_DIR = "./data/pca_sift_features"
MODEL_DIR = "./data/models"

DTYPE = np.float32

os.makedirs(MODEL_DIR, exist_ok=True)


def load_all_descriptors(feature_dir):
    files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")))
    all_desc = []

    for f in files:
        desc = np.load(f)
        if desc is None or len(desc) == 0:
            continue
        all_desc.append(desc.astype(DTYPE, copy=False))

    X = np.vstack(all_desc)
    logger.info(f"X shape = {X.shape}")
    logger.info(f"Total memory = {X.nbytes / 1024**2:.1f} MB")
    return X


def compute_full_PCA():
    X = load_all_descriptors(FEATURE_DIR)

    logger.info("Training full PCA (n_components = full)")
    pca = PCA(
        n_components=None,   # full PCA
        svd_solver="full",
        whiten=False
    )
    pca.fit(X)

    model_path = os.path.join(MODEL_DIR, "pca_sift_full.pkl")
    joblib.dump(pca, model_path)

    logger.info("Saved full PCA model")
    logger.info(
        f"Total explained variance = "
        f"{np.sum(pca.explained_variance_ratio_):.4f}"
    )

    # 額外：列出前幾個維度的能量
    for k in [20, 36, 64]:
        ev = np.sum(pca.explained_variance_ratio_[:k])
        logger.info(f"Explained variance @ {k}D = {ev:.4f}")


def export_truncated_pca(full_pca_path, n, output_path):
    """
    Load a full PCA model and export a truncated PCA model with n components.

    Parameters
    ----------
    full_pca_path : str
        Path to pca_sift_full.pkl
    n : int
        Number of principal components to keep
    output_path : str
        Output path for truncated PCA model
    """
    logger.info(f"Loading full PCA model from {full_pca_path}")
    full_pca = joblib.load(full_pca_path)

    if n > full_pca.components_.shape[0]:
        raise ValueError(
            f"Requested n={n} exceeds PCA dimension "
            f"{full_pca.components_.shape[0]}"
        )

    logger.info(f"Exporting PCA-SIFT model with {n} dimensions")

    pca_trunc = PCA(n_components=n)

    # ===== 核心：直接裁切 PCA 參數 =====
    pca_trunc.components_ = full_pca.components_[:n].copy()
    pca_trunc.mean_ = full_pca.mean_.copy()
    pca_trunc.explained_variance_ = (
        full_pca.explained_variance_[:n].copy()
    )
    pca_trunc.explained_variance_ratio_ = (
        full_pca.explained_variance_ratio_[:n].copy()
    )

    # sklearn 有時會用到這個（保險）
    pca_trunc.n_features_in_ = full_pca.n_features_in_

    # singular_values_ 可選，但補上最完整
    if hasattr(full_pca, "singular_values_"):
        pca_trunc.singular_values_ = (
            full_pca.singular_values_[:n].copy()
        )

    joblib.dump(pca_trunc, output_path)
    logger.info(f"Saved truncated PCA model → {output_path}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR+"pca_sift_full.pkl"):
        compute_full_PCA()
    
    # =====================
    pca_dims=[6,8,10,12,16,20,28,36,3042]
    for k in pca_dims:
        output_path = os.path.join(MODEL_DIR, f"pca_sift_{k}d.pkl")
        export_truncated_pca(
            full_pca_path=os.path.join(MODEL_DIR, "pca_sift_full.pkl"),
            n=k,
            output_path=output_path
        )
