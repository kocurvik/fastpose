import numpy as np
import pytest

from fastpose.estimators.utils import (camera_focal, unproject_pair,
                                       unproject_points)


def _K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# camera_focal
# ---------------------------------------------------------------------------

def test_camera_focal_from_matrix_averages_fx_fy():
    K = _K(900.0, 1100.0, 320.0, 240.0)
    assert camera_focal(K) == pytest.approx(1000.0)


def test_camera_focal_from_square_pixel_matrix():
    K = _K(1000.0, 1000.0, 320.0, 240.0)
    assert camera_focal(K) == pytest.approx(1000.0)


def test_camera_focal_from_poselib_camera():
    poselib = pytest.importorskip('poselib')
    camera = poselib.Camera('PINHOLE', [900.0, 1100.0, 320.0, 240.0], 640, 480)
    assert camera_focal(camera) == pytest.approx(camera.focal())


# ---------------------------------------------------------------------------
# unproject_points
# ---------------------------------------------------------------------------

def test_unproject_points_with_none_camera_is_identity():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert unproject_points(None, x) is x


def test_unproject_points_matrix_matches_manual_formula():
    rng = np.random.default_rng(0)
    fx, fy, cx, cy = 900.0, 1100.0, 320.0, 240.0
    K = _K(fx, fy, cx, cy)
    x = rng.uniform(0.0, 640.0, size=(20, 2))

    xu = unproject_points(K, x)

    expected = np.column_stack([(x[:, 0] - cx) / fx, (x[:, 1] - cy) / fy])
    np.testing.assert_allclose(xu, expected, atol=1e-12)


def test_unproject_points_poselib_camera_matches_matrix():
    poselib = pytest.importorskip('poselib')
    rng = np.random.default_rng(1)
    fx, fy, cx, cy = 900.0, 1100.0, 320.0, 240.0
    x = rng.uniform(0.0, 640.0, size=(20, 2))

    xu_matrix = unproject_points(_K(fx, fy, cx, cy), x)
    camera = poselib.Camera('PINHOLE', [fx, fy, cx, cy], 640, 480)
    xu_camera = unproject_points(camera, x)

    np.testing.assert_allclose(xu_camera, xu_matrix, atol=1e-9)


# ---------------------------------------------------------------------------
# unproject_pair
# ---------------------------------------------------------------------------

def test_unproject_pair_noop_when_both_cameras_none():
    x1 = np.array([[1.0, 2.0]])
    x2 = np.array([[3.0, 4.0]])
    x1u, x2u, max_error = unproject_pair(None, None, x1, x2, 0.002)
    assert x1u is x1
    assert x2u is x2
    assert max_error == 0.002


def test_unproject_pair_scales_threshold_by_average_focal():
    K1 = _K(900.0, 900.0, 320.0, 240.0)
    K2 = _K(1100.0, 1100.0, 330.0, 250.0)
    x1 = np.array([[320.0, 240.0], [400.0, 300.0]])
    x2 = np.array([[330.0, 250.0], [410.0, 310.0]])

    x1u, x2u, max_error = unproject_pair(K1, K2, x1, x2, 2.0)

    np.testing.assert_allclose(x1u, unproject_points(K1, x1))
    np.testing.assert_allclose(x2u, unproject_points(K2, x2))
    assert max_error == pytest.approx(2.0 / 1000.0)  # average of 900, 1100


def test_unproject_pair_scales_extra_thresholds_and_passes_none_through():
    K = _K(1000.0, 1000.0, 320.0, 240.0)
    x1 = np.array([[320.0, 240.0]])
    x2 = np.array([[330.0, 250.0]])

    _, _, max_error, max_reproj_error, extra = unproject_pair(
        K, K, x1, x2, 2.0, 16.0, None)

    assert max_error == pytest.approx(2.0 / 1000.0)
    assert max_reproj_error == pytest.approx(16.0 / 1000.0)
    assert extra is None


def test_unproject_pair_uses_focal_of_whichever_camera_is_supplied():
    # only camera1 given: average focal falls back to that single camera
    K1 = _K(500.0, 500.0, 320.0, 240.0)
    x1 = np.array([[320.0, 240.0]])
    x2 = np.array([[0.1, 0.2]])

    x1u, x2u, max_error = unproject_pair(K1, None, x1, x2, 1.0)

    np.testing.assert_allclose(x1u, unproject_points(K1, x1))
    assert x2u is x2
    assert max_error == pytest.approx(1.0 / 500.0)
