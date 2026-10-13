"""把 PDBbind v2020 索引表转换成单标签亲和力训练 CSV。

标签口径：-log10(Kd/Ki/IC50)，按 Kd > Ki > IC50 的优先级取第一个可用值。
同一 pdb_id 的多行（链/界面级索引）折叠为一行。
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Mapping

import pandas as pd

AFFINITY_PRIORITY = ("Kd", "Ki", "IC50")

# 8 个在 xtrimofold3 的 MoLFormer 缓存里缺失的样本，见
# docs/HANDOFF_molformer_esmc.md:74-81。这里一并排除，保证两边样本集可比。
EXCLUDED_PDB_IDS = frozenset(
    {"2pll", "2vgc", "3d45", "3pjg", "4b70", "6ajz", "6e5b", "6fmp"}
)

# 由 Step 1 在集群上确认后填写真实列名。
DEFAULT_AFFINITY_COLUMNS = {"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"}


def _as_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def pick_affinity(
    record: Mapping[str, Any],
    *,
    priority: tuple[str, ...] = AFFINITY_PRIORITY,
) -> tuple[float, str] | None:
    """返回 (-log10(value), 类型)；全都不可用时返回 None。"""
    for kind in priority:
        value = _as_positive_float(record.get(kind))
        if value is not None:
            return -math.log10(value), kind
    return None


def build_affinity_table(
    df: pd.DataFrame,
    *,
    id_col: str,
    affinity_columns: Mapping[str, str],
    priority: tuple[str, ...] = AFFINITY_PRIORITY,
    excluded: frozenset[str] = EXCLUDED_PDB_IDS,
    sequence_col: str = "Sequence",
    smiles_col: str = "Smiles",
) -> pd.DataFrame:
    missing = [name for name in affinity_columns.values() if name not in df.columns]
    if missing:
        raise ValueError(
            f"索引表缺少亲和力列 {missing}；实际列: {sorted(df.columns)}"
        )
    if id_col not in df.columns:
        raise ValueError(f"索引表缺少 id 列 {id_col!r}")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        pdb_id = str(row[id_col]).strip().lower()
        if not pdb_id or pdb_id in excluded:
            continue

        source = {kind: row.get(column) for kind, column in affinity_columns.items()}
        picked = pick_affinity(source, priority=priority)
        if picked is None:
            continue
        label, affinity_type = picked

        records.append(
            {
                "pdb_id": pdb_id,
                "Sequence": row.get(sequence_col),
                "Smiles": row.get(smiles_col),
                "pKaffinity": float(label),
                "affinity_type": affinity_type,
            }
        )

    table = pd.DataFrame.from_records(records)
    if table.empty:
        return pd.DataFrame(columns=["pdb_id", "Sequence", "Smiles", "pKaffinity", "affinity_type"])

    table["_priority"] = table["affinity_type"].map(
        {kind: rank for rank, kind in enumerate(priority)}
    )
    table = (
        table.sort_values(["pdb_id", "_priority"])
        .drop_duplicates(subset="pdb_id", keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-csv", required=True,
                        help="PDBbind v2020 索引表，例如 pdbbind_2020_feats.csv")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--id-col", default="pdb_id")
    parser.add_argument("--sequence-col", default="Sequence")
    parser.add_argument("--smiles-col", default="Smiles")
    parser.add_argument("--kd-col", default=DEFAULT_AFFINITY_COLUMNS["Kd"])
    parser.add_argument("--ki-col", default=DEFAULT_AFFINITY_COLUMNS["Ki"])
    parser.add_argument("--ic50-col", default=DEFAULT_AFFINITY_COLUMNS["IC50"])
    parser.add_argument("--split-csv", default=None,
                        help="可选：既有划分文件，仅用于统计覆盖情况")
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    table = build_affinity_table(
        df,
        id_col=args.id_col,
        affinity_columns={"Kd": args.kd_col, "Ki": args.ki_col, "IC50": args.ic50_col},
        sequence_col=args.sequence_col,
        smiles_col=args.smiles_col,
    )
    table.to_csv(args.out_csv, index=False)

    print(f"[PDBbind] 输入行数={len(df)} 输出复合物数={len(table)}")
    print(f"[PDBbind] 标签类型分布={table['affinity_type'].value_counts().to_dict()}")
    print(f"[PDBbind] pKaffinity 统计: min={table['pKaffinity'].min():.3f} "
          f"max={table['pKaffinity'].max():.3f} mean={table['pKaffinity'].mean():.3f} "
          f"std={table['pKaffinity'].std():.3f}")
    if args.split_csv:
        split = pd.read_csv(args.split_csv)
        split_ids = set(split.iloc[:, 0].astype(str).str.strip().str.lower())
        covered = len(split_ids & set(table["pdb_id"]))
        print(f"[PDBbind] 与 {args.split_csv} 的交集={covered} / {len(split_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
