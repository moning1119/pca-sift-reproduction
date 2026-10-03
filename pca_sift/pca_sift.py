import numpy as np
import cv2
from numpy.linalg import norm
from numpy import sqrt
from sift import pysift


def compute_support_region(keypoint, gaussian_images,
                           window_width=4, scale_multiplier=3):
    """
    Compute SIFT support region parameters consistent with OpenCV SIFT.
    """
    octave, layer, scale = pysift.unpackOctave(keypoint)
    gaussian_image = gaussian_images[octave + 1, layer]

    hist_width = scale_multiplier * 0.5 * scale * keypoint.size
    num_rows, num_cols = gaussian_image.shape

    half_width = int(round(
        hist_width * sqrt(2) * (window_width + 1) * 0.5
    ))
    half_width = int(min(
        half_width,
        sqrt(num_rows ** 2 + num_cols ** 2)
    ))

    return gaussian_image, hist_width, half_width, scale

def extract_pca_sift_patch(keypoint, gaussian_image,
                           hist_width, half_width, scale,
                           patch_size=41):
    """
    Extract rotation-aligned PCA-SIFT patch (41x41) from SIFT support region.
    """
    angle = np.deg2rad(360. - keypoint.angle)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    point = np.round(scale * np.array(keypoint.pt)).astype(int)

    patch = np.zeros((patch_size, patch_size), dtype=np.float32)
    half = patch_size // 2

    for i in range(patch_size):
        for j in range(patch_size):
            # Normalized PCA-SIFT coordinate → SIFT support region
            x = (j - half) * (hist_width / (patch_size // 2))
            y = (i - half) * (hist_width / (patch_size // 2))

            # Rotate back to image coordinates
            col = x * cos_angle - y * sin_angle
            row = x * sin_angle + y * cos_angle

            img_row = int(round(point[1] + row))
            img_col = int(round(point[0] + col))

            if (img_row <= 0 or img_row >= gaussian_image.shape[0] - 1 or
                img_col <= 0 or img_col >= gaussian_image.shape[1] - 1):
                return None

            patch[i, j] = gaussian_image[img_row, img_col]

    return patch

def compute_gradient_vector(patch):
    """
    Convert 41x41 patch to 3042-D gradient vector.
    """
    # Compute gradients
    dx = patch[1:-1, 2:] - patch[1:-1, :-2]
    dy = patch[:-2, 1:-1] - patch[2:, 1:-1]

    # Stack gradients: (39 x 39 x 2)
    grad = np.stack([dx, dy], axis=-1)

    # Flatten
    vec = grad.reshape(-1).astype(np.float32)

    # L2 normalization
    n = norm(vec)
    if n > 1e-7:
        vec /= n

    return vec

def compute_pca_sift_descriptors(keypoints, gaussian_images):
    """
    Compute PCA-SIFT gradient vectors (before PCA).
    """
    descriptors = []

    for kp in keypoints:
        result = compute_support_region(kp, gaussian_images)
        gaussian_image, hist_width, half_width, scale = result

        patch = extract_pca_sift_patch(
            kp, gaussian_image,
            hist_width, half_width, scale
        )

        if patch is None:
            continue

        vec = compute_gradient_vector(patch)
        descriptors.append(vec)

    return np.array(descriptors, dtype=np.float32)

# ONLY Compute SIFT keypoints for an input image. for training purpose
def computeKeypoints(image,
                                   sigma=1.6,
                                   num_intervals=3,
                                   assumed_blur=0.5,
                                   image_border_width=5,
                                   max_size=800):
    """
    ONLY Compute SIFT keypoints for an input image.
    """
    h,w=image.shape
    scale = min(1.0,max_size/max(h,w))
    if scale<1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale,interpolation=cv2.INTER_AREA)
    
    
    image = image.astype('float32')
    
    base_image = pysift.generateBaseImage(image, sigma, assumed_blur)
    num_octaves = pysift.computeNumberOfOctaves(base_image.shape)
    gaussian_kernels = pysift.generateGaussianKernels(sigma, num_intervals)
    gaussian_images = pysift.generateGaussianImages(
        base_image, num_octaves, gaussian_kernels
    )
    dog_images = pysift.generateDoGImages(gaussian_images)

    keypoints = pysift.findScaleSpaceExtrema(
        gaussian_images, dog_images,
        num_intervals, sigma, image_border_width
    )
    del dog_images
    keypoints = pysift.removeDuplicateKeypoints(keypoints)
    keypoints = pysift.convertKeypointsToInputImageSize(keypoints)

    return keypoints,gaussian_images

def computeKeypointsAndDescriptors(image,
                                   sigma=1.6,
                                   num_intervals=3,
                                   assumed_blur=0.5,
                                   image_border_width=5):
    """
    Compute SIFT keypoints and PCA-SIFT descriptors for an input image.
    """
    # image = image.astype('float32')
    # base_image = pysift.generateBaseImage(image, sigma, assumed_blur)
    # num_octaves = pysift.computeNumberOfOctaves(base_image.shape)
    # gaussian_kernels = pysift.generateGaussianKernels(sigma, num_intervals)
    # gaussian_images = pysift.generateGaussianImages(
    #     base_image, num_octaves, gaussian_kernels
    # )
    # dog_images = pysift.generateDoGImages(gaussian_images)

    # keypoints = pysift.findScaleSpaceExtrema(
    #     gaussian_images, dog_images,
    #     num_intervals, sigma, image_border_width
    # )
    # keypoints = pysift.removeDuplicateKeypoints(keypoints)
    # keypoints = pysift.convertKeypointsToInputImageSize(keypoints)
    keypoints, gaussian_images = computeKeypoints(
        image, sigma, num_intervals, assumed_blur, image_border_width
    )

    descriptors = compute_pca_sift_descriptors(keypoints, gaussian_images)

    return keypoints, descriptors


