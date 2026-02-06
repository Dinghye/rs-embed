import numpy as np
import pytest

from rs_embed.ops.pooling import pool_chw_to_vec


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
