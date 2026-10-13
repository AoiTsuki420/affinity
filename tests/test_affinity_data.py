import shutil
from pathlib import Path

import numpy as np
import pytest

from data_affinity import AffinityScaler


@pytest.fixture()
def scratch_dir():
    """仓库内的一次性目录，替代 pytest 的 tmp_path（见 R12）。

    必须新建一个顶层目录：沙箱只对仓库根下新造出来的顶层目录放行，
    往里层嵌套（或复用 pytest 自己建的 basetemp）会被拒绝 scan dir。
    """
    root = Path("pytest-cache-files-affinity-scaler")
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_fit_transform_standardizes_to_zero_mean_unit_std():
    y = np.array([-2.0, 0.0, 4.0, 6.0, 8.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(y)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std()) - 1.0) < 1e-4


def test_inverse_transform_round_trips():
    y = np.array([-1.5, 2.25, 7.0, 11.5], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    restored = scaler.inverse_transform(scaler.transform(y))
    np.testing.assert_allclose(restored, y, atol=1e-4)


def test_deterministic_scale_for_constant_input():
    """常量输入不能让 scale 变成 0，否则后续除零。"""
    y = np.full(5, 5.0, dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(y)
    assert np.isfinite(z).all()


def test_fit_ignores_non_finite_values():
    y = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    assert np.isfinite(scaler.mean_)
    assert np.isfinite(scaler.scale_)


def test_fit_raises_when_no_finite_values():
    y = np.array([np.nan, np.inf], dtype=np.float32)
    with pytest.raises(ValueError):
        AffinityScaler().fit(y)


def test_transform_preserves_nan_for_invalid_rows():
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    z = scaler.transform(np.array([1.0, np.nan, 3.0], dtype=np.float32))
    assert np.isnan(z[1])
    assert np.isfinite(z[0]) and np.isfinite(z[2])


def test_save_and_load_round_trip(scratch_dir):
    y = np.array([-1.0, 0.5, 3.0, 9.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    path = scratch_dir / "scaler_affinity.json"
    scaler.save(str(path))

    reloaded = AffinityScaler.load(str(path))
    np.testing.assert_allclose(reloaded.transform(y), scaler.transform(y), atol=1e-6)
