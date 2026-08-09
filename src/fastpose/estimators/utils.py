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


def camera_focal(camera):
    # average focal length of a camera given either as a 3x3 intrinsic
    # matrix K or a poselib.Camera (which exposes .focal())
    if hasattr(camera, 'focal'):
        return float(camera.focal())
    K = np.asarray(camera, dtype=np.float64)
    return float((K[0, 0] + K[1, 1]) / 2.0)


def unproject_points(camera, x):
    # camera is None (points already calibrated, returned unchanged), a 3x3
    # intrinsic matrix K, or a poselib.Camera; returns calibrated (x, y)
    # points with any homogeneous scale divided out
    if camera is None:
        return x
    if hasattr(camera, 'unproject'):
        xu = camera.unproject(x)
    else:
        K = np.asarray(camera, dtype=np.float64)
        xh = np.column_stack([x, np.ones(len(x))])
        xu = xh @ np.linalg.inv(K).T
    return xu[:, :2] / xu[:, 2, np.newaxis]


def unproject_pair(camera1, camera2, x1, x2, max_error, *extra_errors):
    # unprojects x1/x2 through camera1/camera2 (each None, a 3x3 K, or a
    # poselib.Camera) and rescales max_error (plus any extra thresholds,
    # e.g. a reprojection threshold) by the average focal length of the
    # cameras that were supplied; a no-op when both cameras are None
    if camera1 is None and camera2 is None:
        return (x1, x2, max_error) + extra_errors
    x1u = unproject_points(camera1, x1)
    x2u = unproject_points(camera2, x2)
    focals = [camera_focal(c) for c in (camera1, camera2) if c is not None]
    avg_focal = float(np.mean(focals))
    scaled = tuple(err / avg_focal if err is not None else None
                   for err in (max_error,) + extra_errors)
    return (x1u, x2u) + scaled
