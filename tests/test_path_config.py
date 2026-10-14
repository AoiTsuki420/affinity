"""`path_config` 的 profile 解析 —— 重点是本次新增的 `k8s` profile。

R49：`path_config` 不做任何 shell 展开。它只在 `resolve_runtime_paths` 里读
`ENV_OVERRIDES` 指定的环境变量，然后把值直接交给 `Path(v)`。所以 profile 里写的
`${USER}` 必须在解析时就地展开，否则每个 k8s run 都会写到字面量为 `${USER}` 的
目录下——而只断言 `is not None` / `isinstance(Path)` 的测试完全看不出这一点。
"""
from pathlib import Path

from path_config import resolve_runtime_paths


def test_k8s_profile_resolves():
    paths = resolve_runtime_paths(profile="k8s")
    assert paths["profile"] == "k8s"
    assert paths["train_csv_path"] is not None
    assert paths["esm_model_path"] is not None
    assert paths["molt5_model_path"] is not None


def test_k8s_profile_backbone_paths_are_path_objects():
    paths = resolve_runtime_paths(profile="k8s")
    assert isinstance(paths["esm_model_path"], Path)
    assert isinstance(paths["molt5_model_path"], Path)


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("CATA_ESM_MODEL_PATH", "/tmp/esmc-local")
    paths = resolve_runtime_paths(profile="k8s")
    assert str(paths["esm_model_path"]).replace("\\", "/") == "/tmp/esmc-local"


def test_existing_profiles_still_resolve():
    for profile in ("autodl", "windows"):
        assert resolve_runtime_paths(profile=profile)["profile"] == profile


def test_k8s_profile_resolved_values_contain_no_shell_placeholder():
    """R49：解析结果里不能残留 `$`——那会变成真实存在的目录名。"""
    paths = resolve_runtime_paths(profile="k8s")
    for key, value in paths.items():
        if isinstance(value, (str, Path)):
            assert "$" not in str(value), f"{key} 仍含未展开的占位符: {value!r}"


def test_k8s_profile_expands_the_user_at_resolve_time(monkeypatch):
    """R49：用户名在 resolve 时从环境读，且 runs/weights 辅助路径一起跟随。"""
    monkeypatch.setenv("USER", "alice")
    paths = resolve_runtime_paths(profile="k8s")

    assert str(paths["runs_dir"]).replace("\\", "/") == "/pfs_beijing/alice/kineager_runs"
    assert str(paths["weights_dir"]).replace("\\", "/") == "/pfs_beijing/alice/kineager_runs"
    assert str(paths["train_csv_path"]).replace("\\", "/") == (
        "/pfs_beijing/alice/kineager_data/pdbbindv2020_affinity.csv"
    )
    assert str(paths["best_ema_path"]).replace("\\", "/") == (
        "/pfs_beijing/alice/kineager_runs/predictor_best_ema.pt"
    )


def test_k8s_profile_falls_back_when_no_user_variable_is_set(monkeypatch):
    """容器里 USER/USERNAME 常常都没设：回退成一个固定目录名，但绝不留 `$`。"""
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("USERNAME", raising=False)
    paths = resolve_runtime_paths(profile="k8s")
    assert str(paths["runs_dir"]).replace("\\", "/") == "/pfs_beijing/kineager/kineager_runs"
