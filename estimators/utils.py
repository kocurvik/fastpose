"""Shared helpers for the full estimation pipelines."""

import math

import numpy as np


def normalize_points(x1, x2):
    # single isotropic shift + scale shared by both images so that the
    # Sampson error (and therefore the inlier threshold) scales uniformly
    points = np.concatenate([x1, x2])
    centroid = points.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
    scale = math.sqrt(2.0) / mean_dist if mean_dist > 0.0 else 1.0
    T = np.array([[scale, 0.0, -scale * centroid[0]],
                  [0.0, scale, -scale * centroid[1]],
                  [0.0, 0.0, 1.0]])
    return scale * (x1 - centroid), scale * (x2 - centroid), T, scale


def point_columns(x1, x2):
    # the `data` tuple layout shared by the epipolar problems: four
    # contiguous float64 coordinate columns (SIMD-friendly)
    return (np.ascontiguousarray(x1[:, 0]), np.ascontiguousarray(x1[:, 1]),
            np.ascontiguousarray(x2[:, 0]), np.ascontiguousarray(x2[:, 1]))
