import numpy as np
import cv2
from pca_sift import pca_sift
from matplotlib import pyplot as plt
import logging
import pickle
import joblib



logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 10

img1 = cv2.imread('./data/raw/box.png', 0)           # queryImage
img2 = cv2.imread('./data/raw/box_in_scene.png', 0)  # trainImage

# ===============================
# Load PCA model (trained before)
# ===============================
# with open('./data/models/pca_sift_20d.pkl', 'rb') as f:
#     pca_model = pickle.load(f)

pca_model = joblib.load("./data/models/pca_sift_36d.pkl")

# PCA_MEAN = pca_model['mean']          # shape: (3042,)
# PCA_COMPONENTS = pca_model['components']  # shape: (D, 3042)

# def project_pca(vec):
#     """
#     Project 3042-D PCA-SIFT vector to low-dimensional PCA space.
#     """
#     v = vec - PCA_MEAN
#     proj = np.dot(PCA_COMPONENTS, v)
#     n = np.linalg.norm(proj)
#     if n > 1e-7:
#         proj /= n
#     return proj.astype(np.float32)

# ==================================
# Compute PCA-SIFT keypoints + raw descriptors
# ==================================
kp1, raw_des1 = pca_sift.computeKeypointsAndDescriptors(img1)
kp2, raw_des2 = pca_sift.computeKeypointsAndDescriptors(img2)

# ===============================
# PCA projection (THIS IS THE KEY)
# ===============================
# des1 = np.array([project_pca(v) for v in raw_des1], dtype=np.float32)
# des2 = np.array([project_pca(v) for v in raw_des2], dtype=np.float32)

des1 = pca_model.transform(raw_des1)
des2 = pca_model.transform(raw_des2)

# L2 normalization (PCA-SIFT 標準步驟)
des1 /= np.linalg.norm(des1, axis=1, keepdims=True)
des2 /= np.linalg.norm(des2, axis=1, keepdims=True)


# Initialize and use FLANN
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)

    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw PCA-SIFT keypoint matches
    for m in good:
        pt1 = (
            int(kp1[m.queryIdx].pt[0]),
            int(kp1[m.queryIdx].pt[1] + hdif)
        )
        pt2 = (
            int(kp2[m.trainIdx].pt[0] + w1),
            int(kp2[m.trainIdx].pt[1])
        )
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

    plt.imshow(newimg)
    plt.title("PCA-SIFT Matching (with PCA Projection)")
    plt.axis("off")
    plt.show()

else:
    print(
        "Not enough matches are found - %d/%d"
        % (len(good), MIN_MATCH_COUNT)
    )
