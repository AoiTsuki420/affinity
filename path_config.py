from pathlib import Path
import os
from typing import Any


PATH_PROFILES = {
    "autodl": {
        "project_root": "/root/autodl-tmp/gptsrc",
        "runs_dir": "/root/autodl-fs/runs",
        "weights_dir": "/root/autodl-fs/runs",
        "train_data_dir": "/root/autodl-fs/data",
        "train_csv_path": "/root/autodl-fs/data/kcat-over-Km-data_0.4simi-10fold.csv",
        "kcat_struct_csv_path": "/root/autodl-fs/data/kcat_dataset_cleaned_after_drop.csv",
        "km_struct_csv_path": "/root/autodl-fs/data/Km_dataset_cleaned_after_drop.csv",
        "unified_train_csv_path": "/root/autodl-fs/data/unified_train_mixed.csv",
        "unified_struct_npz_path": "/root/autodl-fs/data/unified_train_mixed_struct.npz",
        "skid_kcat_archive_dir": "/root/autodl-fs/data/SKiD_kcat_archive",
        "skid_km_archive_dir": "/root/autodl-fs/data/SKiD_Km_archive",
        "val_csv_path": "/root/autodl-tmp/gptsrc/val_set.csv",
        "scaler_json_path": "/root/autodl-tmp/gptsrc/scaler_fold.json",
        "metrics_json_path": "/root/autodl-tmp/gptsrc/metrics.json",
        "esm_model_path": "/root/autodl-tmp/models/esm2_t33_650M_UR50D",
        "molt5_model_path": "/root/autodl-tmp/models/molt5-base-smiles2caption",
    },
    "windows": {
        "project_root": "./",
        "runs_dir": "./runs",
        "weights_dir": "./weights",
        "train_data_dir": "./train_data",
        "train_csv_path": "./kcat-over-Km-data_0.4simi-10fold.csv",
        "kcat_struct_csv_path": "./structdata/kcat_dataset_cleaned_after_drop.csv",
        "val_csv_path": "./val_set.csv",
        "scaler_json_path": "./scaler_fold.json",
        "metrics_json_path": "./metrics.json",
        "esm_model_path": "./backbone/esm2_t6_8M_UR50D",
        "molt5_model_path": "./backbone/molt5-base-smiles2caption",
    },
    # ${USER} 不是 shell 展开：它由 resolve_runtime_paths 在解析时按环境替换
    # （见 _expand_user_placeholder）。留着字面量就会写出名为 "${USER}" 的目录。
    "k8s": {
        "project_root": "/workspace/kineager",
        "runs_dir": "/pfs_beijing/${USER}/kineager_runs",
        "weights_dir": "/pfs_beijing/${USER}/kineager_runs",
        # train_data_dir 是**写入**目标：train_epoch 往里 np.save 每个 epoch 的
        # preds/labels .npy。所以它指向用户自己的目录，而不是 PDBbind 数据集目录
        # （那是别人共读的数据：写进去要么 PermissionError，要么污染共享数据集）。
        "train_data_dir": "/pfs_beijing/${USER}/kineager_runs",
        "train_csv_path": "/pfs_beijing/${USER}/kineager_data/pdbbindv2020_affinity.csv",
        "val_csv_path": "/pfs_beijing/${USER}/kineager_data/pdbbindv2020_affinity_val.csv",
        "esm_model_path": "/pfs_beijing/xinran_2026/models/biohub--ESMC-6B",
        "molt5_model_path": "/pfs_beijing/xinran_2026/models/molformer/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314",
    },
}


ENV_OVERRIDES = {
    "project_root": "CATA_PROJECT_ROOT",
    "runs_dir": "CATA_RUNS_DIR",
    "weights_dir": "CATA_WEIGHTS_DIR",
    "train_data_dir": "CATA_TRAIN_DATA_DIR",
    "train_csv_path": "CATA_TRAIN_CSV_PATH",
    "kcat_struct_csv_path": "CATA_KCAT_STRUCT_CSV_PATH",
    "km_struct_csv_path": "CATA_KM_STRUCT_CSV_PATH",
    "unified_train_csv_path": "CATA_UNIFIED_TRAIN_CSV_PATH",
    "unified_struct_npz_path": "CATA_UNIFIED_STRUCT_NPZ_PATH",
    "skid_kcat_archive_dir": "CATA_SKID_KCAT_ARCHIVE_DIR",
    "skid_km_archive_dir": "CATA_SKID_KM_ARCHIVE_DIR",
    "val_csv_path": "CATA_VAL_CSV_PATH",
    "scaler_json_path": "CATA_SCALER_JSON_PATH",
    "metrics_json_path": "CATA_METRICS_JSON_PATH",
    "esm_model_path": "CATA_ESM_MODEL_PATH",
    "molt5_model_path": "CATA_MOLT5_MODEL_PATH",
}


FILESYSTEM_PATH_KEYS = {
    "project_root",
    "runs_dir",
    "weights_dir",
    "train_data_dir",
    "train_csv_path",
    "kcat_struct_csv_path",
    "km_struct_csv_path",
    "unified_train_csv_path",
    "unified_struct_npz_path",
    "skid_kcat_archive_dir",
    "skid_km_archive_dir",
    "val_csv_path",
    "scaler_json_path",
    "metrics_json_path",
    "esm_model_path",
    "molt5_model_path",
}


# profile 里可以写 ${USER}；path_config 不做 shell 展开，只在解析时替换一次。
_USER_PLACEHOLDER = "${USER}"
_DEFAULT_USER_NAME = "kineager"


def _current_user_name() -> str:
    """当前用户名：USER → USERNAME（Windows）→ 固定回退。

    容器里两个变量常常都没有，所以回退值是必需的：宁可落到一个共享但可覆盖的
    目录（CATA_RUNS_DIR 等），也不能把字面量 ${USER} 写进路径。
    """
    return os.getenv("USER") or os.getenv("USERNAME") or _DEFAULT_USER_NAME


def _expand_user_placeholder(value):
    if isinstance(value, str) and (_USER_PLACEHOLDER in value):
        return value.replace(_USER_PLACEHOLDER, _current_user_name())
    return value


def resolve_runtime_paths(profile=None):
    selected = profile or os.getenv("CATA_PROFILE")
    if not selected:
        selected = "windows" if os.name == "nt" else "autodl"
    if selected not in PATH_PROFILES:
        supported = ", ".join(sorted(PATH_PROFILES.keys()))
        raise ValueError(f"Unknown path profile '{selected}'. Supported profiles: {supported}")

    raw = dict(PATH_PROFILES[selected])
    for key, env_name in ENV_OVERRIDES.items():
        env_value = os.getenv(env_name)
        if env_value:
            raw[key] = env_value

    # ${USER} 在这里、也只在这里展开：profile 值是字符串，Path 转换之前必须干净。
    raw = {key: _expand_user_placeholder(value) for key, value in raw.items()}

    paths: dict[str, Any] = {
        k: (Path(v) if (k in FILESYSTEM_PATH_KEYS and v) else (v or None))
        for k, v in raw.items()
    }

    weights_dir = paths.get("weights_dir")
    if weights_dir is not None:
        paths["best_ema_path"] = weights_dir / "predictor_best_ema.pt"
        paths["best_raw_path"] = weights_dir / "predictor_best_raw.pt"
        paths["final_model_path"] = weights_dir / "predictor.pt"
    else:
        paths["best_ema_path"] = None
        paths["best_raw_path"] = None
        paths["final_model_path"] = None

    paths["profile"] = selected
    return paths


def ensure_paths(paths, required_keys):
    missing = [k for k in required_keys if paths.get(k) is None]
    if missing:
        names = ", ".join(missing)
        profile = paths.get("profile", "unknown")
        raise ValueError(
            f"Missing path values for profile '{profile}': {names}. "
            "Please set CATA_PROFILE or corresponding CATA_* environment variables."
        )
