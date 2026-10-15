# KinEAGER PDBbind v2020 affinity — cluster runbook

Operator runbook for the `--task affinity` path added by the
`feat/pdbbind-affinity` branch: single-task regression over `-log10(Kd/Ki/IC50)`
with **ESMC-6B + MoLFormer** backbones under last-30-layer LoRA, one GPU per pod,
launched on Kubeflow either as a bare PyTorchJob
(`yamls/kubeflow/kineager_affinity_k8s.yaml`, §6.1) or through Kubeflow
Pipelines with the elastic, Worker-only pipeline
(`yamls/kubeflow/kineager_affinity_pipeline.yaml`, §6.2).

Companion documents: the design spec
`docs/superpowers/specs/2026-10-13-kineager-pdbbind-affinity-design.md`, the plan
`docs/superpowers/plans/2026-10-13-kineager-pdbbind-affinity.md`, the ruling
appendix `docs/superpowers/plans/2026-10-13-kineager-pdbbind-affinity-rulings.md`,
and the README section `README.md` → "PDBbind v2020 Protein-Ligand Affinity
Training".

状态（2026-10-15）：该分支已在集群（2×A800）上端到端跑通——探针、数据集构建、LoRA
训练（含 EMA 验证）、每轮断点与续跑均已实测。**第 0 节是本部署的具体镜像与命令**
（弹性流水线 KFP 路线的机制说明见 §6.2）；第 1~9 节保留为原始通用操作指南与验收清单。

---

## 0. 部署速查（本集群）

`xinran_2026` 在 biomap 集群上的具体、已验证配置；第 1~9 节保留为通用操作指南。

### 0.1 训练镜像

```
harbor.biomap-int.com/xinran_2026/kineager:affinity-v1
digest sha256:73c485d81e021eac1faa267f2e1e27a13a165180179893c1a015dc945518674e   （构建于 2026-10-15）
```

基于本仓库的 `Dockerfile`、以 `xinran_2026/esmc20260915:v1` 为底构建：CUDA 12.6、
torch 2.11+cu126、esm 3.4.1、transformers 4.57.6、peft、rdkit、swanlab 0.10.2、
pytest；仓库代码烤在 `/workspace/kineager`。

**什么时候需要重建（本部署约定）**：走 `/pfs` 路线的运行（§0.4 单机、§0.7 泛用
流水线，`bash_command` 都是 `/pfs` 绝对路径）读取的是**共享盘上的仓库代码**——改
脚本/训练代码/超参都**不需要重建镜像**；只有两种情形需要：① 新增 Python 依赖；
② 改用镜像内的仓库副本运行（例如专用流水线的 `working_dir=/workspace/kineager`，
此时要么重建，要么把它的 `working_dir` 同样指到 `/pfs` 路径）。重建命令：

```bash
cd /pfs_beijing/xinran_2026/project/affinity
docker build --network host -t harbor.biomap-int.com/xinran_2026/kineager:affinity-v1 .
docker push harbor.biomap-int.com/xinran_2026/kineager:affinity-v1
```

### 0.2 数据集

`/pfs_beijing/xinran_2026/kineager_data/pdbbindv2020_affinity.csv` —— 16,446 个复合物
（15,536 主训练集 + 881 新增，剔除了一条混入的 CASF 泄漏样本 `3qgy`），列为
`pdb_id,Sequence,Smiles,pKaffinity,affinity_type`。CASF-285 核心集已排除（外部测试集）。
重建方法（在 `xinran-xtrimofold3-a800` 容器内执行；bio pickle 反序列化需要 protenix
在 `sys.path` 上）：

```bash
/opt/miniconda3/envs/xtrimoengine/bin/python \
  /pfs_beijing/xinran_2026/general_protein_dir/affinity_plm_runs/scripts/build_kineager_affinity_csv.py \
  --split "/pfs_beijing/xinran_2026/general_protein_dir/affinity_plm_runs/splits/pdbbindv2020_train_covered_minus_val_add881.csv,/pfs_beijing/xinran_2026/general_protein_dir/affinity_plm_runs/splits/pdbbindv2020_val_covered.csv" \
  --exclude-csv /pfs_beijing/xinran_2026/general_protein_dir/affinity_plm_runs/splits/casf2016_285_ids.csv \
  --workers 32 \
  --out /pfs_beijing/xinran_2026/kineager_data/pdbbindv2020_affinity.csv
```

### 0.3 裸 PyTorchJob 路线（kubectl）

```bash
cd /pfs_beijing/xinran_2026/project/affinity
bash bin/train/launch_kineager_affinity_k8s.sh              # 删旧 job -> apply -> 跟 rank-0 日志
bash bin/train/launch_kineager_affinity_k8s.sh --dry-run    # 只打印将执行的 kubectl 命令
```

包装脚本会自动发现 rank-0 pod（裸 manifest 是 `*-master-0`；弹性流水线没有 master
pod，是 `*-worker-0`），不需要手动指定。等价原生命令：

```bash
kubectl delete pytorchjob kineager-affinity 2>/dev/null     # 固定名字：不删旧的直接 apply 不会启动任何东西
kubectl apply -f yamls/kubeflow/kineager_affinity_k8s.yaml
kubectl get pytorchjob kineager-affinity -o wide
kubectl logs -f kineager-affinity-master-0 -c pytorch
```

### 0.4 单机启动（一台机器，示例为双卡）

```bash
LOG_NAME="affinity_$(date +%Y%m%d_%H%M%S).log"
NEW=/pfs_beijing/xinran_2026/general_protein_dir/kineager_runs
docker run -d --name kineager-affinity-local \
  --gpus '"device=7,5"' --shm-size 16g \
  -v /pfs_beijing/xinran_2026/project/affinity:/workspace/kineager \
  -v /pfs_beijing/xinran_2026/models:/pfs_beijing/xinran_2026/models:ro \
  -v /pfs_beijing/xinran_2026/kineager_data:/pfs_beijing/xinran_2026/kineager_data \
  -v $NEW:$NEW \
  -v /pfs_beijing/xinran_2026/project/affinity/run_logs:/run_logs \
  -e LOG_NAME="$LOG_NAME" -e USER=xinran_2026 \
  -e CATA_RUNS_DIR=$NEW -e CATA_WEIGHTS_DIR=$NEW -e CATA_TRAIN_DATA_DIR=$NEW \
  -e TRAIN_CSV=/pfs_beijing/xinran_2026/kineager_data/pdbbindv2020_affinity.csv \
  -e KINEAGER_ESMC_PATH=/pfs_beijing/xinran_2026/models/ESMC-6B2026.9.15 \
  -e KINEAGER_MOLFORMER_PATH=/pfs_beijing/xinran_2026/models/molformer/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314 \
  -e NPROC_PER_NODE=2 -e NNODES=1 -e NODE_RANK=0 -e MASTER_ADDR=127.0.0.1 -e MASTER_PORT=29517 \
  -e NUM_WORKERS=4 -e OMP_NUM_THREADS=8 \
  -e SWANLAB_API_KEY=<与流水线 yaml 里同一个 key> \
  -w /workspace/kineager \
  harbor.biomap-int.com/xinran_2026/kineager:affinity-v1 \
  bash -lc 'python bin/train/prewarm_remote_code.py > /run_logs/${LOG_NAME} 2>&1 && bash bin/train/kineager_k8s.sh -epochs 30 -batch_size 4 --split_mode group_pair --split_seed 42 --val_ratio 0.05 --test_ratio 0.01 --run_name affinity_local_v1 2>&1 | tee -a /run_logs/${LOG_NAME}'
```

说明：单卡用 `--gpus '"device=7"'` 和 `-e NPROC_PER_NODE=1`；`prewarm_remote_code.py`
只在"同一容器内多 rank"时需要。选卡前先 `nvidia-smi`——rank 会和卡上其他任务分享算力。

### 0.5 监控 / 停止 / 续跑

```bash
tail -f run_logs/<LOG_NAME>.log              # 单机后台运行的完整日志 tee 在这里
docker logs -f kineager-affinity-local       # 或者用 docker 看
docker stop kineager-affinity-local          # 停止（断点不受影响）
kubectl delete pytorchjob kineager-affinity  # k8s 停止
```

续跑语义：每轮结束写 `predictor_checkpoint.pt`（模型 + 优化器 + scaler + EMA，约 27GB）。
用**同样的命令**重跑（`--run_name` 与 `--split_seed` 不变）会打印
`[RESUME] ... start_epoch=N+1` 并从下一轮继续；LR 调度器状态无法序列化，续跑后会重新
warmup（已知）。第一轮完成前中断则退回 `predictor_best_ema.pt`（仅权重、优化器全新）。

### 0.6 产物

- run 目录 `/pfs_beijing/xinran_2026/general_protein_dir/kineager_runs/<run_name>/seed<seed>/`：
  `predictor_checkpoint.pt`、`predictor_best_ema.pt`、`predictor_best_raw.pt`、
  `train/val/test_set.csv`、`split_meta.json`、`metrics.json`
- 每轮 `preds_epoch*.npy` / `labels_epoch*.npy` 在 `.../kineager_runs/` 下
- 日志在 `project/affinity/run_logs/`
- swanlab：项目 `enzyme-kinetics-cv`（key 在流水线/manifest 的环境变量里）

当前训练配置：每卡 `-batch_size 4`、不做梯度累积（累积 + DDP 会触发
`expect_autograd_hooks_` 断言与 epoch 边界卡死）、`--val_ratio 0.05 --test_ratio 0.01`
——PDBbind 的 test 切片只是占位，真正测试集是 CASF-285。

### 0.7 弹性流水线路线（kineager-affinity-train，KFP UI）

机制、参数含义、`logpath` 与 `LOG_PATH` 的区别见 §6.2；这里给**本部署当前提交的
具体填法**（12 卡 = 3 pod × 4 GPU；续跑本地单卡已完成的前两轮。换卡数只改本表
参数，脚本不用动——`NPROC_PER_NODE` 由 wrapper 按容器内可见 GPU 数自动设定）：

| 参数 | 本部署取值 |
|---|---|
| `pod_nums` | `3`（= NNODES；世界大小 = `pod_gpus` × `pod_nums` = 12）|
| `pod_gpus` | `4`（每个 pod 的 GPU 数）|
| `min_replicas` / `max_replicas` | `3` / `3`（固定 = `pod_nums`，不要弹性）|
| `cpu` / `memory` | `40` / `320Gi`（按每卡比例；节点宽松时也可沿用 `80/640Gi`）|
| `image` | `harbor.biomap-int.com/xinran_2026/kineager:affinity-v1` |
| `working_dir` | `/pfs_beijing/xinran_2026/project/affinity`（专用流水线可用 `/workspace/kineager`）|
| `bash_command` | 见下方代码块 |
| `logpath` | `/pfs_beijing/xinran_2026/project/affinity/run_logs/affinity_k8s.log` |

```bash
bash /pfs_beijing/xinran_2026/project/affinity/bin/train/run_kineager_affinity.sh
```

这条一行脚本把本部署的 env 块、超参与 prewarm 全部封装在内（各项均可用环境变量
覆盖，见脚本头部注释）；**泛用的 `xtfold3-train` 流水线同样用它**——把它填进
`bash_command` 即可，拓扑解析由 launcher 兼容 `PET_*` / 裸名 / `RANK`。

要点：

- `pod_gpus > 1` 时每个 pod 有多 rank；`run_kineager_affinity.sh` 已包含 prewarm
  （避免 HF 远程代码模块缓存竞争），`pod_gpus=1` 时它也基本零成本。
- USER / TRAIN_CSV / 骨干路径 / CATA_* 写路径 / swanlab key 在流水线 yaml 的
  `manifest: |` env 块里；改这些值要**重新上传 yaml（新版本）**。
- 启动后核对每个 pod 日志首行：`[kineager] ... nnodes=3 node_rank=0/1/2 nproc_per_node=4`；
  若出现 `nnodes=1`、或 `nproc_per_node` 不等于 `pod_gpus`、或两个 pod 的
  `node_rank` 相同，立刻停掉（否则各 pod 单机训练、互相覆盖产物）。
- **同一 `run_name` 只能有一个任务在写**。本部署的本地单卡 run 已停止，断点在
  epoch 2（best val 0.6705），提交后应从 `[RESUME] start_epoch=3` 继续。
- 想先小规模验证路线：按 §6.2 的建议先 `pod_gpus=1` / `pod_nums=2` 跑一轮
  （记得换一个 `--run_name`，别和本部署的断点混写）。
- 本路线**不需要重建镜像**：wrapper、launcher 与训练代码全部从 `/pfs` 共享盘读取
  （镜像只提供 Python 环境）；只有新增 pip 依赖、或改用镜像内仓库副本（在
  `/workspace/kineager` 运行）时才需要重建。

---

## 1. Prerequisites on the cluster

### 1.1 The `esm` package, `>=3.4.1`

The ESMC tokenizer comes from the `esm` package, not from `transformers`, and the
version floor is load-bearing:

```bash
python -c "import importlib.metadata as m; print(m.version('esm'))"        # must be >= 3.4.1
python -c "from esm.models.esmc import EsmcTokenizer; print(EsmcTokenizer)" # must not raise
```

`requirements.txt` pins `esm>=3.4.1` for exactly this reason: the ESMC
checkpoint's own remote loader raises `pip install 'esm>=3.4.1'` when the package
is missing or too old (ruling R50), and an unpinned install can resolve to a
version without `EsmcTokenizer` — failing only at model-load time, after a
multi-gigabyte download. `EsmcTokenizer` is constructed with no arguments.

`pip install -r requirements.txt` also installs **PyYAML**, which
`tests/test_k8s_artifacts.py` imports (R60); without it the cluster's first
`python -m pytest -p no:cacheprovider tests/ test_scripts/ -v` errors at
collection.

### 1.2 The two backbone snapshots

Both must be **local directories visible inside the pod** (no hub download at
run time):

| Key | Path in the k8s profile | What it is |
|---|---|---|
| `KINEAGER_ESMC_PATH` | `/pfs_beijing/xinran_2026/models/biohub--ESMC-6B` | ESMC-6B, protein side, loaded with `trust_remote_code=True` |
| `KINEAGER_MOLFORMER_PATH` | `/pfs_beijing/xinran_2026/models/molformer/models--ibm-research--MoLFormer-XL-both-10pct/snapshots/7b12d946c181a37f6012b9dc3b002275de070314` | MoLFormer-XL-both-10pct, ligand side, pinned to revision `7b12d946c181a37f6012b9dc3b002275de070314` |

The launcher requires both (`KINEAGER_ESMC_PATH:?` / `KINEAGER_MOLFORMER_PATH:?`)
and exits 1 with the variable name if either is unset. The MoLFormer revision is
a design constraint: the tokenizer loads must honour the same revision as the
weights, or the pair can drift (R22).

### 1.3 The PDBbind v2020 index CSV

The reference cluster keeps it under
`/pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020`
(read-only for everyone else — never write there; R57 moved the run's write
targets away from it). You need the index table that carries the affinity
columns, e.g. `pdbbind_2020_feats.csv`.

`train_csv_path` in the k8s profile resolves to
`/pfs_beijing/${USER}/kineager_data/pdbbindv2020_affinity.csv`, i.e. the built
table lives in **your own** area; the manifest's `TRAIN_CSV` must point at it.

---

## 2. Confirm the LoRA target names against the real backbones

**Do this before the first run.** LoRA target discovery is **name-based**:
`models/plm_backbones.py::discover_lora_targets` attaches adapters to every
`nn.Linear` whose qualified name ends with one of `DEFAULT_LORA_SUFFIXES`
(27 entries: `attention.self.query`, `query`, `SelfAttention.q`, `wi_0`,
`q_proj`, `fc1`, …) inside the selected layers. If the real backbones name their
projections differently, the injection matches a near-empty target set, the
model trains almost nothing, and the run still reports a plausible-looking loss —
the failure that unit-test fixtures cannot catch, because the fixtures are built
from the names we assumed.

### 2.1 ESMC-6B

```bash
python tools/probe_backbone_modules.py \
  --model "$KINEAGER_ESMC_PATH" \
  --kind esmc \
  --out probe_esmc.json
```

### 2.2 MoLFormer

```bash
python tools/probe_backbone_modules.py \
  --model "$KINEAGER_MOLFORMER_PATH" \
  --kind molformer \
  --revision 7b12d946c181a37f6012b9dc3b002275de070314 \
  --out probe_molformer.json
```

`python -m tools.probe_backbone_modules …` is equivalent (both invocation forms
work; R9/R14).

### 2.3 What to look for in the JSON

Both runs must exit **0**. The probe writes `--out` and prints the JSON *before*
deciding its exit code, so even a failing probe leaves the module dump behind —
but a resolve failure adds a `[PROBE-FAILURE]` block on stderr and returns
**2**. Never accept a probe by its exit code alone; read these keys:

| Key | Expected | Why it matters |
|---|---|---|
| `hidden_size` | ESMC **2560**, MoLFormer **768** | `None` means `resolve_hidden_size` could not read the config — the probe exits 2 |
| `num_layers` | ESMC **80**, MoLFormer **12** | `--lora_last_k 30` selects *the last 30 of this number*; a wrong count puts LoRA on the wrong layers |
| `leaf_names` | the last dot-segment of every `nn.Linear` name | this is the input to the suffix decision below |
| `layer_names_sample` | module names containing `layer`/`layers`/`block` | tells you whether ESMC's transformer stack sits in a container `_layer_containers` recognises (`encoder.layer`, `encoder.block`, `layers`, `h`) |
| `linear_names` | fully-qualified `nn.Linear` names | the ground truth if a leaf name looks ambiguous |

**If the ESMC layer container is not one of those four**, `resolve_num_layers`
and `inject_lora` abort loudly with a message listing what they tried — a blocked
run with an actionable error, not a silent mis-attachment. The fix is a code
change in `models/plm_backbones.py` (`_NUM_LAYER_FIELDS` / `_layer_containers`)
driven by `layer_names_sample`. `--lora_explicit_layers` is **not** an escape
hatch here: it still needs `num_layers` for its range filter.

### 2.4 The suffix decision: pass the **union**, never the difference

`--lora_target_suffixes` **replaces** the default whitelist — that is the tested
contract of `discover_lora_targets` (`tuple(suffixes) if suffixes else
DEFAULT_LORA_SUFFIXES`). It does not extend it.

So compare the probe's `leaf_names` against `DEFAULT_LORA_SUFFIXES` (27 entries
in `models/plm_backbones.py`) and:

* if every `leaf_name` is already covered → **omit the flag entirely**;
* if the probe found projections the defaults miss → pass the **union**:
  `DEFAULT_LORA_SUFFIXES ∪ probe leaf_names`.

Passing the *difference* (as an earlier draft of the plan said) would silently
drop every default suffix for that backbone and attach LoRA to almost nothing
(ruling R20). Do not edit the code default; the CLI flag is the designed escape
hatch. Two further details: an explicitly **empty** sequence means "use the
defaults" (the truthiness test above), and `--lora_target_suffixes` takes
`nargs="+"`, so pass the names space-separated.

The launcher does **not** set this flag: it sets only the five LoRA/throughput
flags (`--lora_layer_policy`, `--lora_last_k`, `--grad_checkpoint_backbone`,
`--iters_to_accumulate`, `--num_workers`) plus the required
`--task`/`-csv`/`-esm`/`-molt5`, and then appends the caller's `"$@"`. In the
Kubeflow route, add `--lora_target_suffixes` to the `run_trainer()` argument list
in the manifest; standalone, append it to the launcher command line.

---

## 3. Confirm the PDBbind index column names

`tools/build_pdbbind_affinity_csv.py` ships
`DEFAULT_AFFINITY_COLUMNS = {"Kd": "Kd", "Ki": "Ki", "IC50": "IC50"}`, which is an
**unconfirmed placeholder** — the real PDBbind header has never been readable
from the development host. Read it before building anything:

```bash
head -1 /pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020/pdbbind_2020_feats.csv \
  | tr ',' '\n' | nl
```

or, more robustly, one column per line with pandas:

```bash
python -c "import pandas as pd; print('\n'.join(pd.read_csv('/pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020/pdbbind_2020_feats.csv', nrows=0).columns))"
```

Map what you find onto the builder's flags:

| Flag | Default | What it must be |
|---|---|---|
| `--id-col` | `pdb_id` | the PDB identifier; values are lower-cased and de-duplicated |
| `--sequence-col` | `Sequence` | the protein sequence |
| `--smiles-col` | `Smiles` | the ligand SMILES |
| `--kd-col` | `Kd` | the **Kd** column, in M |
| `--ki-col` | `Ki` | the **Ki** column, in M |
| `--ic50-col` | `IC50` | the **IC50** column, in M |

A wrong guess fails loudly rather than silently: a missing column raises
`ValueError: 索引表缺少亲和力列 [...]；实际列: [...]` naming the real header. The
label is `-log10(value)` with `Kd > Ki > IC50` priority, and non-positive or
non-finite values are treated as missing.

---

## 4. Build the training CSV

```bash
python tools/build_pdbbind_affinity_csv.py \
  --index-csv /pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020/pdbbind_2020_feats.csv \
  --out-csv /pfs_beijing/$USER/kineager_data/pdbbindv2020_affinity.csv \
  --id-col <real id column> \
  --kd-col <real Kd column> \
  --ki-col <real Ki column> \
  --ic50-col <real IC50 column>
```

Optional: `--split-csv <existing split file>` prints how many of that file's ids
the built table covers.

Expected output — three lines of statistics and a silent exit 0:

```
[PDBbind] 输入行数=<rows in the index> 输出复合物数=<complexes in the table>
[PDBbind] 标签类型分布={'Kd': …, 'Ki': …, 'IC50': …}
[PDBbind] pKaffinity 统计: min=… max=… mean=… std=…
```

Sanity checks: the complex count is non-zero and comfortably smaller than the
input row count (multi-row chain/interface entries are folded to one row per
`pdb_id`); the pKaffinity range is roughly 2–12; the built CSV has the columns
`pdb_id,Sequence,Smiles,pKaffinity,affinity_type`. Eight known-missing PDB ids
(`2pll, 2vgc, 3d45, 3pjg, 4b70, 6ajz, 6e5b, 6fmp`) are excluded so the sample set
matches the xtrimofold3 MoLFormer cache.

The file must end up where `TRAIN_CSV` points. Nothing writes to it afterwards —
it is a read input — but it must be readable by every pod.

---

## 5. Fill in the manifest

`yamls/kubeflow/kineager_affinity_k8s.yaml` is apply-ready except for **16
`<REPLACE_ME …>` values**. They are deliberately not valid Kubernetes
quantities, so an unedited apply fails loudly instead of landing on wrong
hardware.

| # | Where | Value |
|---|---|---|
| 1 | `image` | KinEAGER training image reference — must have this repo on disk and `esm>=3.4.1` installed |
| 2 | `workingDir` | KinEAGER repo root inside that image (the launcher is invoked as `bash bin/train/kineager_k8s.sh`) |
| 3 | `env: USER` | **your username** on the shared filesystem — see the invariant below |
| 4 | `env: TRAIN_CSV` | the CSV built in §4 |
| 5 | `env: KINEAGER_ESMC_PATH` | local ESMC-6B directory, visible in the pod |
| 6 | `env: KINEAGER_MOLFORMER_PATH` | local MoLFormer snapshot, visible in the pod |
| 7 | `env: LOG_PATH` | rank-0 log file, on storage **every pod can write** |
| 8 | `env: NNODES` | `Worker.replicas + 1` |
| 9–11 | `resources.requests`: `rdma/hca`, `cpu`, `memory` | per-pod request, e.g. `1`, `16`, `200Gi` |
| 12–14 | `resources.limits`: `rdma/hca`, `cpu`, `memory` | keep **identical** to the requests (Guaranteed QoS) |
| 15 | `volumes[shm-volume].emptyDir.sizeLimit` | `/dev/shm` size, e.g. `64Gi` |
| 16 | `Worker.replicas` | `NNODES - 1` |

Three invariants that are not obvious from the file:

1. **`NNODES` must equal `Master.replicas + Worker.replicas`**, i.e.
   `1 + Worker.replicas`. `Master.replicas` is fixed at 1. The training-operator
   injects `PET_NNODES`; `NNODES` is the fallback the embedded script uses when
   it is absent, and `bin/train/kineager_k8s.sh` fails fast on
   `NODE_RANK >= NNODES`. If the two disagree, the pods rendezvous with the wrong
   world size (or, at `NNODES=1`, each pod trains its own copy — see §7).
2. **`LOG_PATH` must be writable by every pod.** Only rank 0 appends to it
   (`run_trainer >> "${LOG_PATH:?…}" 2>&1`), the other ranks stay on stdout, but
   which physical pod ends up as rank 0 is not something to plan around, and the
   file lives on shared storage.
3. **`USER` must be set.** The k8s path profile resolves `${USER}` in `runs_dir`,
   `weights_dir` **and** `train_data_dir`; a pod sets neither `USER` nor
   `USERNAME`, and `path_config` would then fall back to the literal `kineager`,
   making the write paths the **shared** `/pfs_beijing/kineager/kineager_runs`.
   `train_data_dir` is a write target (`main_train_predictor_multigpu.py:1873`
   creates it and `:1874-1875` save the per-epoch `preds_epoch*.npy` /
   `labels_epoch*.npy` there), so two concurrent users would overwrite each
   other's artifacts while both runs look healthy. `bin/train/kineager_k8s.sh`
   therefore refuses to start (R61) when the k8s profile is in use and the value
   is unsafe. Three details of that refusal matter to an operator:
   * it accepts **`USER` or `USERNAME`**, the same two forms and the same
     precedence `path_config` uses (R63), and exports whichever one supplied the
     value so the trainer's environment carries exactly the value that was
     checked (R64) — a plain `USER=alice; bash bin/train/kineager_k8s.sh …`
     without `export` would otherwise pass the check and still resolve to the
     shared directory;
   * it also rejects a value that still starts with `<REPLACE_ME` (R65): the
     manifest's placeholder is a non-empty string, which Kubernetes happily
     accepts, and the run would otherwise create a directory named after the
     placeholder;
   * the remedy is in the message itself, so a failure tells you what to export.

   Leave `CATA_PROFILE` at `k8s`: without it the trainer falls back to the
   platform default profile (`autodl` on Linux) and silently reads the wrong
   paths (R58).

Also worth knowing before you edit: `nodeSelector` (`gpu.device: a100`,
`rdma: roce2`) and the `NBTopic` / `006_mix` tolerations are copied verbatim from
the xtrimofold3 reference. Replace them if your GPU nodes are labelled or tainted
differently — a mismatch shows up as a pod stuck `Pending`.

---

## 6. Launch

There are **two** launchers for the same training run, and the launcher script
(`bin/train/kineager_k8s.sh`) is deliberately identical for both: it resolves the
topology from `PET_*` first, then the bare names, then `RANK`/`WORLD_SIZE`, then
its own defaults (§6.4), so a pod cannot end up on the wrong world size just
because a different orchestrator started it.

| | bare PyTorchJob (§6.1) | Kubeflow Pipelines (§6.2) |
|---|---|---|
| Manifest | `yamls/kubeflow/kineager_affinity_k8s.yaml` | `yamls/kubeflow/kineager_affinity_pipeline.yaml` |
| Topology | Master + Worker, `Worker.replicas + 1` = `NNODES` | elastic, Worker-only, `pod_nums` = `NNODES` |
| Rank 0 | `kineager-affinity-master-0` | `kineager-affinity-<run>-worker-0` |
| How to start | `kubectl apply -f …`, or the wrapper | upload once in the UI, then **Create run** |
| Values live in | the manifest's `env:` block (edit the file) | the run's parameter form (no file edit) |
| Job object after the run | survives (TTL 6000 s) | deleted ~30 s after it finishes |
| Prefer it when | you are iterating on the container image / env / mounts, you want `kubectl describe`, or Argo/KFP is not usable | you want to re-launch with different parameters, keep a history of runs, or hand the run to someone else |

### 6.1 The bare PyTorchJob (`kubectl apply`)

```bash
kubectl apply -f yamls/kubeflow/kineager_affinity_k8s.yaml
kubectl get pytorchjob kineager-affinity -o wide
kubectl get pods                      # pod names follow <job>-master-0 / <job>-worker-N
kubectl logs -f kineager-affinity-master-0 -c pytorch
```

The container runs `exec bash bin/train/kineager_k8s.sh …`, which validates the
topology and then `exec`s `torch.distributed.run` with
`--nproc_per_node=1` (one GPU per pod), so the workers' own logs carry their
rank and the master's log is the one that is also appended to `LOG_PATH`.

#### The re-run footgun

`metadata.name` is fixed at `kineager-affinity`, `cleanPodPolicy: None` keeps the
pods after the job finishes, and `ttlSecondsAfterFinished: 6000` deletes the job
object only ~100 minutes later. While the finished job object still exists, a
second `kubectl apply` of the **unchanged** manifest reports `unchanged`, exits 0,
and launches **nothing** — the operator sees a successful apply and no new run:

```bash
kubectl delete pytorchjob kineager-affinity     # do this before a re-run
kubectl apply -f yamls/kubeflow/kineager_affinity_k8s.yaml
```

The same fixed name means **two concurrent runs collide**: the second apply (even
with edited values) modifies the same job object. Never start two at once — give
one of them its own namespace, or edit `metadata.name` and the two
`kubectl … kineager-affinity` commands above together.

### 6.2 The Kubeflow Pipelines route (elastic, Worker-only)

`yamls/kubeflow/kineager_affinity_pipeline.yaml` is a KFP v1 pipeline whose single
`resource: create` template submits a PyTorchJob. It is a **copy in the shape of
the pipeline already running on this cluster** (`xtfold3-train`, from
`xtrimofold3-affinity-plm`), deliberately: that file is proof of the schema the
cluster's KFP accepts.

**Upload it once.** Kubeflow Pipelines UI → **Pipelines** → **Upload pipeline** →
**Upload a file** → pick `yamls/kubeflow/kineager_affinity_pipeline.yaml`. The
pipeline is named `kineager-affinity-train`; the annotations
(`pipelines.kubeflow.org/pipeline_spec` et al.) are what the UI reads to build the
parameter form. Uploading again with the same name creates a **new version** of
the pipeline — that is the normal way to pick up a change to the file.

**Then** **Create run** and fill in the eight parameters:

| Parameter | Recommended first value | Why |
|---|---|---|
| `pod_nums` | `2` | the number of worker pods **and** the world size: `minReplicas == maxReplicas == pod_nums == NNODES`. One GPU per pod, so 2 pods = 2 GPUs. Raise it only after a 2-pod run is green |
| `pod_gpus` | `1` | one GPU per pod; the launcher sets `NPROC_PER_NODE` to 1 (`PET_NPROC_PER_NODE` is injected from this), and `nproc_per_node × pod_nums` is the world size |
| `cpu` | `16` | matches the bare manifest's per-pod request |
| `memory` | `200Gi` | matches the bare manifest |
| `image` | `harbor.biomap-int.com/xinran_2026/kineager:affinity-v1` | the same image the bare manifest uses |
| `working_dir` | `/workspace/kineager` | repo root inside that image (the container runs `bash bin/train/kineager_k8s.sh` relative to it) |
| `bash_command` | `bash bin/train/kineager_k8s.sh -epochs 30 -batch_size 4 --split_mode group_pair --split_seed 42` | the launcher takes the training hyperparameters through `"$@"`; **do not** put a shell redirect (`>> log 2>&1`) here — the pipeline's container command already appends to `logpath` on rank 0 only |
| `logpath` | `/pfs_beijing/xinran_2026/general_protein_dir/kineager_runs/kineager_affinity/rank0.log` | rank 0's append target, on storage **every pod can write**. Must be non-empty and free of spaces/quotes: it is substituted into a double-quoted shell string, and an empty value stops the container with a named error |

`pod_nums` is the only parameter that changes the topology. Keep
`minReplicas == maxReplicas == pod_nums`: if the elastic bounds are allowed to
differ, the operator can run fewer replicas than the `NNODES` the launcher
validates, and torchrun then waits forever for a rank that never exists. That is
a hang, not an error message — which is why the pipeline exposes one number
instead of two.

#### How `logpath` and `LOG_PATH` interact

There are two different "log files", and they are not the same thing:

* **`logpath` (a pipeline parameter, required).** The container command is:

  ```bash
  if [ "${NODE_RANK:-0}" -eq 0 ]; then
    <bash_command> >> "<logpath>" 2>&1
  else
    <bash_command>
  fi
  ```

  So **only rank 0** appends to that file; the other ranks stay on their own pod
  stdout. This is what stops N pods from interleaving into one unreadable file.
  The gate lives in the pipeline's container command (not in the launcher) because
  it has to wrap the whole command; the launcher implements the same gate for
  `LOG_PATH` (§6.4), so the two agree.
* **`LOG_PATH` (an env var in the embedded job's `env:` block, empty by default).**
  After the gate above, the container runs `bash bin/train/kineager_k8s.sh …`,
  and that script independently checks `LOG_PATH`: **if it is set, rank 0 appends
  there too and the other ranks do not.** Two consequences:

  * with the shipped default (`LOG_PATH: ""`), rank 0's output lands in `logpath`
    and each rank's output is *also* in its own `kubectl logs`;
  * if you set `LOG_PATH` as well, rank 0's stream goes to **both** files. Set it
    only if some other tooling expects that exact path; otherwise leave it empty
    so there is exactly one rank-0 file.

  The path itself is the one thing `logpath` cannot express: `LOG_PATH` is for
  tooling that reads a fixed location, `logpath` is the per-run parameter.

To edit `LOG_PATH`, `USER`, `TRAIN_CSV`, the backbone paths, the hyperparameter
env vars, or the swanlab key, edit the `env:` block inside this file's
`manifest: |` and re-upload (new pipeline version). A run's parameter form cannot
reach them.

#### Finding the pods and the logs

There is **no master pod** in this topology. Do not look for
`kineager-affinity-master-0`; the elastic job's pods are named
`kineager-affinity-<something>-worker-<N>`, and the `<something>` is generated:

```bash
# which pods exist (label set by the training-operator - works for both topologies)
kubectl get pods -l training.kubeflow.org/job-name=kineager-affinity -o wide

# rank 0's stdout (the pod whose name ends in -worker-0, or in -master-0 for §6.1)
kubectl logs -f kineager-affinity-<run>-worker-0 -c pytorch

# rank 1 and up (useful to check that each pod reports a different node_rank)
kubectl logs kineager-affinity-<run>-worker-1 -c pytorch | head -5

# the rank-0 log file, independent of which pod got rank 0
tail -f /pfs_beijing/xinran_2026/general_protein_dir/kineager_runs/kineager_affinity/rank0.log
```

`bin/train/launch_kineager_affinity_k8s.sh` does this discovery for you when it
follows the log (`-l training.kubeflow.org/job-name=…`, preferring a `master` pod
when one exists and falling back to `worker-0`), so it works for both topologies;
it only applies the bare manifest, so use it for §6.1.

Two operational consequences of the embedded `runPolicy`, kept as-is from the
cluster's pipeline for fidelity:

* `cleanPodPolicy: Running` — the pods are deleted as soon as the job leaves
  `Running`, so **a pod log is gone once the job finishes**. Copy what you need
  while it runs, or rely on `logpath` / `LOG_PATH`.
* `ttlSecondsAfterFinished: 30` — the **PyTorchJob object** is deleted ~30 s after
  the job finishes, so `kubectl describe pytorchjob` / `kubectl get pytorchjob -o
  yaml` post-mortems are impossible. Raise the TTL in the embedded manifest if you
  want them. Only rank 0's log file survives a successful run.

### 6.3 What both routes share

`USER` is required in both, and the launcher defaults `CATA_PROFILE` to `k8s`.
Export the values first, so that `$USER` in `TRAIN_CSV` expands to the username
you just set:

```bash
export CATA_PROFILE=k8s
export USER=<user>                       # required: the write paths derive from it
export TRAIN_CSV=/pfs_beijing/$USER/kineager_data/pdbbindv2020_affinity.csv
export KINEAGER_ESMC_PATH=/models/ESMC-6B
export KINEAGER_MOLFORMER_PATH=/models/MoLFormer-XL-both-10pct

NNODES=2 NODE_RANK=0 MASTER_ADDR=<master-ip> MASTER_PORT=29517 \
  bash bin/train/kineager_k8s.sh -epochs 30 -batch_size 8 --split_mode group_pair \
    --task affinity
```

(That standalone form is useful for a 2-GPU smoke on one host.) Use
`MASTER_PORT=29517` rather than `29300`, which was recorded as occupied on the
reference cluster. `29500` is the launcher's own default; in the Kubeflow route
the training-operator derives the rendezvous port from the container port named
`pytorchjob-port`, and the manifest's comment says to keep the two in sync.

Two option notes for the training side:

* **`--lora_layer_policy` defaults per task** (R70). Left unset, a kinetics run
  uses `all` (the historical behaviour) and an affinity run uses `last_k` with
  `--lora_last_k 30` (the design's last-30-layer LoRA). The launcher passes
  `--lora_layer_policy "${LORA_LAYER_POLICY:-last_k}"` explicitly because it is
  the affinity entry point — a deliberate kinetics run through this launcher
  needs `LORA_LAYER_POLICY=all` to keep the historical layer set.
* **`--lr_find` is refused under `--task affinity`.** The LR range test scores
  batches with the three-head `compute_multihead_loss`, which cannot consume the
  affinity path's `(mu, log_sigma2)` output; the trainer now raises a named error
  up front instead of failing mid-scan. Scan the LR on a kinetics run, or add a
  dedicated range test for the affinity criterion if you need one.

### 6.4 Which topology variable wins

`bin/train/kineager_k8s.sh` resolves all five topology inputs before validating
anything, in this order — first match wins:

| Variable | Resolution order |
|---|---|
| `NNODES` | `PET_NNODES` → `NNODES` → `WORLD_SIZE` → `1` |
| `NODE_RANK` | `PET_NODE_RANK` → `NODE_RANK` → `RANK` → `0` |
| `NPROC_PER_NODE` | `PET_NPROC_PER_NODE` → `NPROC_PER_NODE` → `1` |
| `MASTER_PORT` | `PET_MASTER_PORT` → `MASTER_PORT` → `29500` |
| `MASTER_ADDR` | `PET_MASTER_ADDR` → `MASTER_ADDR` → (empty; required only when `NNODES > 1`) |

The `PET_` names come first because the elastic Worker-only pipeline injects
**only** those (plus `RANK`/`WORLD_SIZE`): with the bare names alone, every pod
would resolve `NNODES=1, NODE_RANK=0`, and that is not an error the run can
notice — either every pod trains its own copy of the whole dataset into the same
`runs_dir`, or torchrun waits forever for ranks that never join. If you ever see
`nnodes=1` in a pod's `[kineager]` line of a multi-pod run, this block is the
first place to look.

---

## 7. Verify multi-GPU actually engaged

Everything below is in the pod log. The decisive lines are 3 and 4.

1. **Topology line, once per pod** (launcher):
   `[kineager] task=affinity profile=k8s nnodes=8 node_rank=0 nproc_per_node=1 master=<addr>:<port>`
   Check that `nnodes > 1`, that `nproc_per_node=1` (one GPU per pod), and that
   **each pod reports a different `node_rank`**.
2. **Resolved paths, on every rank** (trainer):
   `[PATHS] profile=k8s`, `[PATHS] train_data_dir=/pfs_beijing/<your user>/kineager_runs`.
   This is the cheapest proof that the k8s profile really took effect inside the
   pod and that the write paths are user-scoped. `<your user>` must be the `USER`
   you filled in — if it reads `kineager`, stop the run.
3. **`DistributedSampler enabled. World size = <N>`** — printed on every rank,
   and only inside `if is_ddp:`. Its absence means `WORLD_SIZE > 1` was false.
4. **`[DDP] trainable params total=<n> breakdown={'lora': …, 'proj': …, 'interactions': …, 'head': …, 'other': …}`**
   — rank 0 only, and only inside the DDP branch. **This is the single most
   decisive check**: no such line means DDP never wrapped the model, whatever the
   pod count says. `breakdown['lora']` must be non-zero; if it is `0`, the LoRA
   target set (§2) is empty.
5. **One GPU busy per pod** while training (cluster GPU dashboard, or
   `nvidia-smi` inside a pod). Peak memory can also be read from
   `torch.cuda.max_memory_allocated()` in an operator-supplied probe.
6. **`WORLD_SIZE` / `RANK` / `LOCAL_RANK` / `MASTER_ADDR`** are injected by
   torchrun (and, in the Kubeflow route, `RANK` / `PET_NNODES` by the
   training-operator). The trainer derives `use_ddp` from `WORLD_SIZE > 1` and
   then reads `LOCAL_RANK`/`RANK`/`WORLD_SIZE`. Inspect them directly with
   `env | grep -E 'WORLD_SIZE|RANK|LOCAL_RANK|MASTER_ADDR'` inside a pod.

### Failure signs that mean it silently trained on one GPU

* **`DistributedSampler enabled.` and `[DDP] trainable params` are both missing.**
  With `NNODES=1` every pod runs an independent single-process job over the whole
  dataset, and they all write the same `runs_dir` / `train_data_dir` — the
  per-epoch `.npy` artifacts overwrite each other. Worst of all, the launcher's
  own `NODE_RANK < NNODES` check *passes*, because each pod has `NNODES=1`,
  `NODE_RANK=0`. That is the most likely cause of this shape: an unfilled
  `NNODES` placeholder, or `Worker.replicas: 0` so `PET_NNODES` never arrives.
* **Every pod's topology line shows `node_rank=0`** (compare the pods, not just
  the master's log).
* **Only the master pod logs anything** past the topology line, or the worker
  pods exited immediately — a rendezvous that never happened.
* **`[kineager] … nnodes=1 …`** in every pod, or no `[kineager]` line at all
  (meaning the manifest's embedded script did not reach the launcher).
* **`LOG_PATH` only contains one rank's output** while several pods were supposed
  to run: either only rank 0 started, or the other pods died before writing.
* **NCCL says nothing about peers.** The manifest sets `NCCL_DEBUG=INFO` by
  default; a real multi-rank init logs the rank count and channel setup. Silence
  about other ranks plus no `DistributedSampler` line is a single-process run.

---

## 8. Run inference on a trained checkpoint

The branch's inference entry point is `main_infer_predictor.py`. It reads a CSV
row by row, builds the model **from the checkpoint**, and writes one CSV.

```bash
export CATA_PROFILE=k8s
export USER=<user>

python main_infer_predictor.py \
  -csv /pfs_beijing/$USER/kineager_data/pdbbindv2020_affinity_val.csv \
  -weights /pfs_beijing/$USER/kineager_runs/predictor_best_ema.pt \
  -device cuda \
  -esm /models/ESMC-6B \
  -molt5 /models/MoLFormer-XL-both-10pct \
  --task affinity
```

What the operator must know:

* **The task comes from the checkpoint, not from the flag** (R69). The script
  builds through `PredictorConfig.for_task("affinity")` as soon as the checkpoint
  contains `multi_head.heads.affinity.*`, and it also refines the head's
  `hidden_dim` from those weights. Before this was fixed the script always built
  the kinetics three-head config, so an affinity checkpoint loaded with 6
  unexpected keys (the whole trained head), the forward returned a 7-tuple, and
  the script died with `ValueError: too many values to unpack`. Passing
  `--task affinity` explicitly is still correct and is what the command above
  does; passing it with a **kinetics** checkpoint is now a named error instead of
  a confusing crash.
* Input CSV: needs a `sequence` column and a SMILES column (`smiles`; the header
  is lower-cased and `--smiles-col` overrides the guess), plus `pdb_id` if you
  want to align against the truth table. The builder's table from §4 satisfies all
  three.
* Output: `predictions_affinity.csv` in the current working directory, with
  exactly four columns — `pdb_id`, `pred_pKaffinity`, `sigma`, `sigma2`. `sigma2`
  is the predictive variance `exp(log_var)` and `sigma` its square root; the
  log-variance is clamped to ±10, the same bound the training loss uses, so a
  badly-calibrated checkpoint cannot emit an astronomically large `sigma2`.
* Every row also prints `Predicted affinity: pKaffinity=… (sigma=…)`, so a quick
  `head` of the run shows whether the numbers are sane before you read the CSV.
* The script does **not** compute metrics: it predicts and exports. `true_kcat` /
  `true_Km` columns exist only for the kinetics tasks, and the affinity path never
  writes them (R59).

---

## 9. Cluster-only smoke checklist

Every item below needs GPUs, the weights or the `esm` package, so none of it
could be checked while building this branch. Run them in order; item 6 is the
one that decides a code default, and item 14 is the check that would have caught
the inference defect described in §8.

| # | Check | How | Expected result |
|---|---|---|---|
| 1 | ESMC probe resolves | `python tools/probe_backbone_modules.py --model "$KINEAGER_ESMC_PATH" --kind esmc --out probe_esmc.json` | exit 0, `hidden_size=2560`, `num_layers=80`, non-empty `leaf_names`, no `[PROBE-FAILURE]` on stderr |
| 2 | MoLFormer probe resolves | same with `--kind molformer --revision 7b12d946c181a37f6012b9dc3b002275de070314` | exit 0, `hidden_size=768`, `num_layers=12`, non-empty `leaf_names` |
| 3 | ESMC's layer container is recognised | read `layer_names_sample` from `probe_esmc.json` | container is one of `encoder.layer` / `encoder.block` / `layers` / `h`; otherwise the run aborts with `resolve_num_layers` / `inject_lora`'s message and `_NUM_LAYER_FIELDS` / `_layer_containers` must be extended from that sample |
| 4 | LoRA target decision is the union | compare `probe_*.json`'s `leaf_names` with `DEFAULT_LORA_SUFFIXES` | flag omitted when the defaults already cover them, otherwise `DEFAULT_LORA_SUFFIXES ∪ leaf_names`; `breakdown['lora'] > 0` in the rank-0 line |
| 5 | The injected target count is real | rank-0 `[DDP] trainable params …` line, plus the layers LoRA actually hit | LoRA parameters and a target count that scale with `layers × matched suffixes` (peft creates 2 tensors and 16·r parameters per target), rather than a small number that merely looks plausible |
| 6 | Gradient checkpointing is genuinely in effect | run the same config with `GRAD_CHECKPOINT_BACKBONE=off` and `=active`; compare peak memory | a clear peak-memory reduction with `active`, and no `ValueError` from `set_backbone_gradient_checkpointing`. A positive return value is **not** proof on the remote-code ESMC model: the function only switches per-layer attributes, so a backbone gating on an encoder-level flag is not covered |
| 7 | ESMC boundary tokens are stripped | in a pod, print the tokenizer's type, its `all_special_ids` (or `*_token_id` ints) and the token ids of one sample sequence; `build_token_mask` is the consumer | the tokenizer exposes special-token ids and `build_token_mask` removes exactly the 2 boundary tokens (`<cls>`/`<eos>`-style). If it exposes **none**, the raw attention mask is returned **silently** and those two tokens pollute `masked_mean` pooling. The exact tokenizer call spelling depends on the installed `esm` version — what matters is that the ids are visible and 2 positions disappear |
| 8 | 2-GPU smoke run completes | the standalone launcher command in §6 with `ITERS_TO_ACCUMULATE=2`, `-epochs 1 -batch_size 2`, a small CSV subset | both ranks rendezvous, 1 epoch finishes, both `DistributedSampler enabled. World size = 2` and `[DDP] trainable params` appear, no NaN/Inf loss |
| 9 | Accumulation really runs | the same run with `ITERS_TO_ACCUMULATE=2` vs `1` | optimizer steps ≈ `ceil(batches/2)`, and the LR actually anneals across the epoch count instead of stalling or completing early. The three schedule-length wiring sites are verified only by code review plus a unit-tested pure helper, so this is the end-to-end confirmation |
| 10 | Checkpoint is written and loadable | look for `predictor_best_ema.pt` under `/pfs_beijing/$USER/kineager_runs`; load it in a pod | the file exists and loads with `strict=True` |
| 11 | `USER` scoping holds in the pod | `[PATHS] train_data_dir=…` in the log; then, in a pod shell: (a) `unset USER USERNAME; bash bin/train/kineager_k8s.sh -epochs 1` (the guard runs before the other required-input checks, so nothing else needs setting); (b) `USER='<REPLACE_ME: x>' bash bin/train/kineager_k8s.sh -epochs 1`; (c) `USER=alice bash bin/train/kineager_k8s.sh -epochs 1` | the log names **your** user (`/pfs_beijing/<you>/kineager_runs`), never `kineager`; (a) exits 1 naming `USER`/`USERNAME` and the shared fallback; (b) exits 1 saying the placeholder is still in place; (c) gets past the guard — the next failure is the `TRAIN_CSV` check, not a `USER` refusal — which a full run sees as a user-scoped `[PATHS]` line, proving the validated value reached the trainer's environment |
| 12 | The training CSV and the split look sane | the builder's statistics from §4, plus the run's `Train samples:`/`Val samples:` lines | non-zero counts, pKaffinity in a plausible range, `--split_mode group_pair --split_seed 42` recorded in the run's split metadata |
| 13 | bf16 is stable | watch the first epoch's loss and the recorded metrics | no NaN/Inf loss; if one appears, note the module and consider `autocast(enabled=False)` for it |
| 14 | **The affinity inference path runs end to end** | run §8's command on a small CSV (the val split from §4) | exit 0, `predictions_affinity.csv` exists with exactly the columns `pdb_id,pred_pKaffinity,sigma,sigma2`, and `pred_pKaffinity` is non-degenerate: finite, not all-identical, and in the rough range of the training labels (~2–12). The run must also print `[AUTO-CONFIG] task -> affinity (checkpoint has multi_head.heads.affinity.*)`, and the load must **not** report the affinity head as unexpected keys. This is the check that would have caught the defect §8 describes — a checkpoint that loads and predicts nothing useful still passes item 10 |

### Recorded acceptance data (plan Step 5)

Write these into the run's smoke report: LoRA hit layers (ESMC's actual layer
range and target count; MoLFormer's likewise), the full
`[DDP] trainable params total=… breakdown={…}` line, seconds per step with and
without accumulation, peak memory, whether two ranks agree on the same LoRA
tensor's gradients, that `predictor_best_ema.pt` loads, and the bf16 result.

### Then decide `--grad_checkpoint_backbone` (plan Step 6)

This is a **measured** decision, not a guess: if peak memory exceeds half of one
GPU's capacity, change the default from `off` to `active`
(`main_train_predictor_multigpu.py`, the `--grad_checkpoint_backbone` argument)
and update spec §5 D2 and §11 to match. The manifest and the launcher both carry
the value explicitly, so a run can be switched per job without a code change.
