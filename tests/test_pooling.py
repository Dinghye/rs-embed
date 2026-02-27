import numpy as np
import pytest

from rs_embed.ops.pooling import pool_chw_to_vec


# ══════════════════════════════════════════════════════════════════════
# basic correctness
# ══════════════════════════════════════════════════════════════════════

def test_pool_chw_to_vec_mean():
    x = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )
    out = pool_chw_to_vec(x, method="mean")
    np.testing.assert_allclose(out, np.array([4.0, 5.0], dtype=np.float32))


def test_pool_chw_to_vec_max():
    x = np.array(
        [
            [[1.0, 3.0], [5.0, 7.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ],
        dtype=np.float32,
    )
    out = pool_chw_to_vec(x, method="max")
    np.testing.assert_allclose(out, np.array([7.0, 8.0], dtype=np.float32))


def test_pool_chw_to_vec_unknown_method():
    x = np.zeros((1, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown pooling"):
        pool_chw_to_vec(x, method="median")


# ══════════════════════════════════════════════════════════════════════
# output dtype is always float32
# ══════════════════════════════════════════════════════════════════════

def test_pool_output_dtype_float64_input():
    x = np.ones((2, 3, 3), dtype=np.float64)
    out = pool_chw_to_vec(x, method="mean")
    assert out.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════
# single pixel (1×1 spatial)
# ══════════════════════════════════════════════════════════════════════

def test_pool_single_pixel():
    x = np.array([[[5.0]], [[9.0]]], dtype=np.float32)
    np.testing.assert_allclose(pool_chw_to_vec(x, method="mean"), [5.0, 9.0])
    np.testing.assert_allclose(pool_chw_to_vec(x, method="max"), [5.0, 9.0])


# ══════════════════════════════════════════════════════════════════════
# single channel
# ══════════════════════════════════════════════════════════════════════

def test_pool_single_channel():
    x = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    out = pool_chw_to_vec(x, method="mean")
    assert out.shape == (1,)
    np.testing.assert_allclose(out, [1.5])


# ══════════════════════════════════════════════════════════════════════
# NaN handling (nanmean / nanmax)
# ══════════════════════════════════════════════════════════════════════

def test_pool_with_nans():
    x = np.array([[[1.0, np.nan], [3.0, 4.0]]], dtype=np.float32)
    out_mean = pool_chw_to_vec(x, method="mean")
    np.testing.assert_allclose(out_mean, [np.nanmean([1.0, np.nan, 3.0, 4.0])])
    out_max = pool_chw_to_vec(x, method="max")
    np.testing.assert_allclose(out_max, [4.0])


# ══════════════════════════════════════════════════════════════════════
# wrong ndim raises
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("shape", [(4,), (2, 3), (1, 2, 3, 4)])
def test_pool_wrong_ndim(shape):
    x = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected"):
        pool_chw_to_vec(x)


# ══════════════════════════════════════════════════════════════════════
# large array (regression / smoke)
# ══════════════════════════════════════════════════════════════════════

def test_pool_large_array():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((64, 128, 128)).astype(np.float32)
    out = pool_chw_to_vec(x, method="mean")
    assert out.shape == (64,)
    assert out.dtype == np.float32
