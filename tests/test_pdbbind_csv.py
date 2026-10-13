import math

import pandas as pd
import pytest

from tools.build_pdbbind_affinity_csv import (
    AFFINITY_PRIORITY,
    EXCLUDED_PDB_IDS,
    build_affinity_table,
    pick_affinity,
)


def test_priority_is_kd_then_ki_then_ic50():
    assert AFFINITY_PRIORITY == ("Kd", "Ki", "IC50")


def test_pick_affinity_prefers_kd():
    value, kind = pick_affinity({"Kd": 1e-9, "Ki": 1e-6, "IC50": 1e-5})
    assert kind == "Kd"
    assert abs(value - 9.0) < 1e-6


def test_pick_affinity_falls_back_to_ki():
    value, kind = pick_affinity({"Kd": None, "Ki": 1e-6, "IC50": 1e-5})
    assert kind == "Ki"
    assert abs(value - 6.0) < 1e-6


def test_pick_affinity_falls_back_to_ic50():
    value, kind = pick_affinity({"Kd": None, "Ki": float("nan"), "IC50": 1e-5})
    assert kind == "IC50"
    assert abs(value - 5.0) < 1e-6


def test_pick_affinity_returns_none_when_nothing_usable():
    assert pick_affinity({"Kd": None, "Ki": 0.0, "IC50": float("nan")}) is None


def test_excluded_ids_are_the_documented_set():
    assert EXCLUDED_PDB_IDS == frozenset(
        {"2pll", "2vgc", "3d45", "3pjg", "4b70", "6ajz", "6e5b", "6fmp"}
    )


def _index_frame():
    return pd.DataFrame(
        {
            "pdb_id": ["1abc", "1abc", "1def", "2pll"],
            "Kd": [1e-9, 1e-9, None, 1e-9],
            "Ki": [None, None, 1e-7, None],
            "IC50": [None, None, None, None],
            "Sequence": ["MKT", "MKT", "MKA", "MKB"],
            "Smiles": ["CCO", "CCO", "CCC", "CCN"],
        }
    )


def test_build_table_collapses_duplicate_pdb_ids():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert list(out["pdb_id"]) == ["1abc", "1def"]


def test_build_table_dedup_keeps_the_highest_priority_label():
    """同一 pdb_id 多行给出不同类型标签时，折叠必须按 priority 而不是行序取值。"""
    frame = pd.DataFrame(
        {
            "pdb_id": ["1abc", "1abc"],
            "Kd": [None, 1e-9],
            "Ki": [1e-6, 1e-6],
            "IC50": [None, None],
            "Sequence": ["MKT", "MKT"],
            "Smiles": ["CCO", "CCO"],
        }
    )
    out = build_affinity_table(
        frame, id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert list(out["pdb_id"]) == ["1abc"]
    row = out.iloc[0]
    assert row["affinity_type"] == "Kd"
    assert abs(float(row["pKaffinity"]) - 9.0) < 1e-6


def test_build_table_excludes_known_bad_ids():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert "2pll" not in set(out["pdb_id"])


def test_build_table_emits_expected_columns():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    for column in ("pdb_id", "Sequence", "Smiles", "pKaffinity", "affinity_type"):
        assert column in out.columns


def test_build_table_label_is_negative_log10():
    out = build_affinity_table(
        _index_frame(), id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    row = out[out["pdb_id"] == "1def"].iloc[0]
    assert row["affinity_type"] == "Ki"
    assert abs(float(row["pKaffinity"]) - 7.0) < 1e-6


def test_build_table_drops_rows_without_usable_affinity():
    frame = _index_frame()
    frame.loc[2, "Ki"] = None
    out = build_affinity_table(
        frame, id_col="pdb_id",
        affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
    )
    assert "1def" not in set(out["pdb_id"])


def test_build_table_raises_when_column_missing():
    frame = _index_frame().drop(columns=["IC50"])
    with pytest.raises(ValueError) as excinfo:
        build_affinity_table(
            frame, id_col="pdb_id",
            affinity_columns={"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"},
        )
    assert "IC50" in str(excinfo.value)
