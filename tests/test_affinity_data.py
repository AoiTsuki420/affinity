import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data_affinity import AffinityDataset, AffinityScaler


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


def test_transform_masks_infinities_not_just_nan():
    """±inf 与 NaN 一样必须映射成 NaN——只有有限值掩码能做到这点。

    NaN 自己会顺着算术传播，所以「掩码写」与「整体写」在 NaN 上无法区分；
    inf 则不同：不掩码时 (inf - mean) / scale 仍是 inf，会原样漏进输出。
    """
    fit_y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    scaler = AffinityScaler().fit(fit_y)

    z = scaler.transform(np.array([1.0, np.inf, np.nan, -np.inf, 3.0], dtype=np.float32))

    # 非有限位置（inf / -inf / nan）必须全部是 NaN，而不是 ±inf 或 0。
    assert np.isnan(z[1]), f"np.inf 未掩码，得到 {z[1]!r}"
    assert np.isnan(z[2]), f"np.nan 未掩码，得到 {z[2]!r}"
    assert np.isnan(z[3]), f"-np.inf 未掩码，得到 {z[3]!r}"
    # 有限位置照常标准化：fit 用的 [1, 2, 3] 标准化后约 [-1.2247, 0, 1.2247]。
    assert np.isfinite(z[[0, 4]]).all()
    np.testing.assert_allclose(z[[0, 4]], [-1.224745, 1.224745], atol=1e-5)


def test_save_and_load_round_trip(scratch_dir):
    y = np.array([-1.0, 0.5, 3.0, 9.0], dtype=np.float32)
    scaler = AffinityScaler().fit(y)
    path = scratch_dir / "scaler_affinity.json"
    scaler.save(str(path))

    reloaded = AffinityScaler.load(str(path))
    np.testing.assert_allclose(reloaded.transform(y), scaler.transform(y), atol=1e-6)


def _frame(**overrides):
    data = {
        "pdb_id": ["1a", "1b", "1c"],
        "Sequence": ["MKTAYIAK", "MKTAYIAL", "MKTAYIAM"],
        "Smiles": ["CCO", "CCC", "CCN"],
        "pKaffinity": [4.0, 6.0, 8.0],
        "affinity_type": ["Kd", "Ki", "IC50"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_dataset_returns_three_tuple():
    dataset = AffinityDataset(_frame())
    item = dataset[0]
    assert len(item) == 3
    seq, smi, y = item
    assert seq == "MKTAYIAK"
    assert smi == "CCO"
    assert isinstance(y, torch.Tensor)


def test_dataset_drops_rows_with_non_finite_labels():
    frame = _frame()
    frame.loc[1, "pKaffinity"] = float("nan")
    dataset = AffinityDataset(frame)
    assert len(dataset) == 2
    assert list(dataset.df["pdb_id"]) == ["1a", "1c"]


def test_dataset_raises_when_label_column_missing():
    with pytest.raises(ValueError) as excinfo:
        AffinityDataset(_frame().drop(columns=["pKaffinity"]))
    assert "pKaffinity" in str(excinfo.value)


def test_dataset_raises_when_no_valid_rows():
    frame = _frame()
    frame["pKaffinity"] = float("nan")
    with pytest.raises(ValueError):
        AffinityDataset(frame)


def test_dataset_accepts_sequence_and_smiles_column_aliases():
    frame = _frame().rename(columns={"Sequence": "sequence", "Smiles": "SMILES"})
    dataset = AffinityDataset(frame)
    seq, smi, _ = dataset[0]
    assert seq == "MKTAYIAK"
    assert smi == "CCO"


def test_dataset_applies_scaler():
    frame = _frame()
    scaler = AffinityScaler().fit(frame["pKaffinity"].to_numpy())
    dataset = AffinityDataset(frame, scaler=scaler)
    _, _, y = dataset[0]
    expected = scaler.transform(frame["pKaffinity"].to_numpy())[0]
    assert abs(float(y) - float(expected)) < 1e-5


def test_dataset_without_scaler_returns_raw_label():
    dataset = AffinityDataset(_frame())
    _, _, y = dataset[2]
    assert abs(float(y) - 8.0) < 1e-5


def test_dataset_return_source_flag():
    dataset = AffinityDataset(_frame(), return_source=True)
    item = dataset[0]
    assert len(item) == 4
    assert item[3] == "Kd"


def test_dataset_respects_custom_label_column():
    frame = _frame().rename(columns={"pKaffinity": "pKd"})
    dataset = AffinityDataset(frame, label_col="pKd")
    _, _, y = dataset[1]
    assert abs(float(y) - 6.0) < 1e-5
