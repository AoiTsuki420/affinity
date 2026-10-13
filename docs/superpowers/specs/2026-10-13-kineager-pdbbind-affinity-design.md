# KinEAGER 适配 PDBbind v2020 蛋白-小分子亲和力任务 — 设计文档

**日期：** 2026-10-13
**状态：** 待用户复核
**上游计划：** 本文件批准后由 `writing-plans` 产出实施计划

---

## 1. 目标

在 KinEAGER 框架内新增一条**单任务蛋白-小分子亲和力回归**通路，训练数据为 PDBbind v2020：

- 蛋白骨干：`biohub/ESMC-6B`（替换 ESM-2）
- 配体骨干：`ibm-research/MoLFormer-XL-both-10pct`（替换 MolT5）
- 微调方式：**在线前向 + LoRA**（backbone 不落盘缓存，梯度回传到 LoRA 适配器）
- 输出：单 head，`-log10(Kd/Ki/IC50)` 一个标量，head 输出 `mu` 与 `log_sigma2`
- 结构分支：**关闭**（不传 `--use_struct_branch`、不传 `--final_contact_npz`）
- 并行：k8s 上**每 pod 1 卡**，`torch.distributed.run --nnodes=N --node_rank=k --nproc_per_node=1 --master_addr --master_port`

**非目标：** 不改动、不删除现有 kcat/Km/kcat-Km 三任务通路；不启用结构分支；不移植 xtrimofold3 的预计算缓存与 `AffinityPLMAdapter`；不引入 DeepSpeed/Lightning。

---

## 2. 用户已确认的决策

| 项 | 决定 |
|---|---|
| PLM 特征路线 | 在线前向 + LoRA（不用预计算缓存） |
| ESMC 规格 | `biohub/ESMC-6B`，与 xtrimofold3 对齐 |
| 亲和力标签 | Kd/Ki/IC50 统一成 `-log10` 单值 |
| Head 输出 | `mu` + `log_sigma2`（保留不确定性机制） |
| k8s 形态 | 每 pod 1 卡，对齐 xtrimofold3 的 rendezvous 契约 |
| LoRA 层范围 | **last-K，K = 30**（每骨干按实际层数 clamp） |
| 入口形态 | 在现有 `main_train_predictor_multigpu.py` 加 `--task` 开关，不新建训练入口 |

---

## 3. 现状勘察结论（KinEAGER）

### 3.1 三任务耦合点（单 head 必须切断的地方）

| 位置 | 内容 |
|---|---|
| `main_train_predictor_multigpu.py:2632` | `active_heads=["kcat","Km","ratio"]` 硬编码 |
| `main_train_predictor_multigpu.py:2637-2641` | `head_specs` 按三任务名索引 |
| `models/predictor.py:965` | `pred_kcat_log, pred_Km_log, pred_ratio_log = outs_list[:3]` |
| `models/predictor.py:766-772` | `s_kcat`/`s_Km`/`s_ratio` 三个 `nn.Parameter`，`s_domain_task` 形状 `(num_domains, 3)` |
| `models/predictor.py:1040` | `saliency()` 里 `{'kcat':0,'Km':1,'kcat_Km':2}` |
| `main_train_predictor_multigpu.py:1239-1258` | 三路解包 + `pred_act = 0.4*pred_act_tmp + 0.6*(pred_kcat - pred_Km + 3.0)` |
| `main_train_predictor_multigpu.py:1360-1401` | 三任务异方差加权、`cap_penalty`、`s_reg` |
| `main_train_predictor_multigpu.py:921-960` | `compute_multihead_loss` 固定三任务 |
| `main_train_predictor_multigpu.py:1554` | `PredictionRecorder(n_samples, output_dim=3)` |
| `utils.py:50-118` | `KineticsScaler` 只产出 `(kcat, Km, ratio)` 三列 |
| `data.py:91-117` | `EnzymeDataset` 只认 `kcat(s^-1)` / `Km(M)` 列 |
| `main_infer_predictor.py:303-305` | 推理侧三路解包 |

**结论：单 head 是一条 data → scaler → `model.forward` → loss → metrics → inference 的纵向改造，不是开关。**

### 3.2 LoRA 注入是骨干相关的硬编码

- `models/predictor.py:568-587` `_build_esm_targets_exact` 拼 ESM-2 名：`encoder.layer.{i}.attention.self.{query,key,value}`、`attention.output.dense`、`intermediate.dense`、`output.dense`
- `models/predictor.py:594-615` `_build_t5_encoder_targets_exact` 拼 T5 名：`encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}`、`layer.1.DenseReluDense.{wi_0,wi_1,wo}`
- 两处都有 `assert targets, ...`，换骨干后必然失败
- `models/predictor.py:618` 策略写死 `policy="all"`（全层注入）

### 3.3 编码器对骨干的隐含假设

- `models/encoders.py:15` 硬编码 `EsmTokenizer`
- `models/encoders.py:36` 读 `backbone.config.hidden_size`
- `models/encoders.py:97` 读 `backbone.config.d_model`（MoLFormer 是 BERT 系，字段为 `hidden_size`）
- `models/encoders.py:41` 蛋白 `max_length=640`；`models/encoders.py:102` 配体 `max_length=384`（MoLFormer token 上限为 **202**）
- `models/predictor.py:676` 调用 `self.molt5.get_encoder()` —— **MoLFormer 是 encoder-only，没有 `get_encoder()`**
- `models/predictor.py:480-481` 用 `AutoModel.from_pretrained` / `AutoModelForSeq2SeqLM.from_pretrained`，对 ESMC 不适用
- `main_train_predictor_multigpu.py:2597-2603` 用 `AutoConfig.num_hidden_layers` 推断层数，remote-code 模型不保证有此字段

### 3.4 已有、可复用的设施

- `models/interactions.py`：`CrossAttentionBlock`（双向 cross-attention + orthogonality + pocket loss）、`ResidueMask`、`LigandMask` —— 与任务无关，直接复用
- `models/predictor.py:307-359` `MultiHead` 已是任务名 → head 的注册容器，`MLPHead` 支持任意 `out_dim`，**单 head 不需要新写 head 模块**
- `models/kcat_expert.py:276-332` `kcat_expert_loss` **已是单任务损失**（Huber + ranking + var_reg + NLL + CCC）
- `main_train_predictor_multigpu.py:247-261` `_barron_loss_per_sample` 可直接复用
- DDP 全套已在：`main_train_predictor_multigpu.py:506-522`（`setup_ddp`/`cleanup_ddp`）、`2446-2470`（`init_process_group` + `set_device`）、`2563-2588`（`DistributedSampler`）、`2698-2704`（`DDP(...)`）、`540-567`（`EMA`）、`2094`（`GradScaler`）、rank0 守卫与 checkpoint 存取
- `models/predictor.py:41-46` 已有 `use_struct_branch` / `enable_avail_gate` 开关；`main_train_predictor_multigpu.py:2319` `--use_struct_branch` 默认 `False`
- `models/kcat_expert.py:184-241` `KcatExpert` 是现成的"结构可用性感知 + 单任务 + `(mu, log_var)` 输出"范式，可作设计参照

---

## 4. 参考实现的硬约束（xtrimofold3-affinity-plm）

这些是踩过坑的结论，本设计必须遵守：

1. **ESMC 的 tokenizer 来自 `esm` 包**，不是 transformers：`from esm.models.esmc import EsmcTokenizer`（`protenix/data/esmc.py:1174-1179`）。transformers 5.x 未注册 `EsmcTokenizer`。→ **新增 `esm` 依赖**。
2. **ESMC 通过 `AutoModelForMaskedLM.from_pretrained(name, trust_remote_code=True)` 加载**（`protenix/data/esmc.py:1182`）。remote-code 模块命名未知 → **必须先跑探针脚本确认 `target_modules` 能命中**，不能猜。
3. **MoLFormer 通过 `AutoModel.from_pretrained(name, trust_remote_code=True)` 加载**，encoder-only（`protenix/data/molformer.py:920-927`）。
4. **MoLFormer revision 必须钉 `7b12d946c181a37f6012b9dc3b002275de070314`**（`docs/HANDOFF_molformer_esmc.md:191-193`）。
5. **MoLFormer token 上限 202**（`docs/pdbbind_molformer.md:127`），hidden 768、12 层。
6. **ESMC-6B：hidden 2560，81 个 hidden states（含 embedding 层），即 80 个 transformer 层**（`docs/HANDOFF_molformer_esmc.md:264`；`launch_affinity_plm.py:379`）。
7. **ESMC 缓存保留 BOS/EOS 两个特殊 token**（`includes_special_tokens = valid_length == len(sequence) + 2`，`protenix/data/esmc.py:1295-1314`）→ 喂进 KinEAGER 时必须从 attention mask 中剔除，否则 BOS/EOS 会污染 `masked_mean` 池化。
8. **bf16 陷阱**：该实现记录过"plain `nn.Linear` 保持 float32，在 bf16 激活下会崩"（`protenix/model/modules/affinity_plm.py:22-24`）。KinEAGER 的 `proj` 在 `autocast` 下由 autocast 接管，风险较低，但 **ESMC remote-code 内部若有绕过 autocast 的算子需实测**。
9. **PDBbind 数据在集群**：`/pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020`，索引 `pdbbind_2020_feats.csv`（链/界面级，同一 `pdb_id` 多行），复合物目录 `P-L/<pdb_id>/`。
10. **已有可用划分**：`pdbbindv2020_train_covered_minus_val.csv` / `pdbbindv2020_val_covered.csv`（`bin/train/train_affinity_v3_esmc_molformer.sh:35-36`）→ 复用以保证与现有亲和力结果可比。
11. **已知需排除的样本**：8 个无 MoLFormer 缓存（`2pll, 2vgc, 3d45, 3pjg, 4b70, 6ajz, 6e5b, 6fmp`）+ 29 个显式氢导致 RDKit canonicalization 分歧的样本（`docs/HANDOFF_molformer_esmc.md:74-81`）。

---

## 5. 关键设计决策

### D1 — LoRA 策略：last-30 层

**决策：** `lora_layer_policy="last_k"`，`lora_last_k=30`；每骨干按实际层数 clamp。

- ESMC-6B（80 层）→ 第 50..79 层
- MoLFormer（12 层）→ 第 0..11 层（全部）

**理由（性能）：** 当 LoRA 只注入第 `s..L-1` 层时，第 `0..s-1` 层不含任何 `requires_grad=True` 的参数，其输入也不 require grad，**PyTorch autograd 不会为这些层建图**。前向照常执行，但不保存激活、不参与反向。这是部分层 LoRA 的白捡收益。

**理由（收敛）：** 深层 LoRA 是 PLM 微调的常见有效配置，浅层多为通用序列特征。

**实测开关：** 增加 `--lora_layer_policy {all,last_k,explicit}` + `--lora_last_k`，保证可回退到全层或指定层做消融。

### D2 — 骨干梯度检查点：按需、分层

KinEAGER 现有 `use_checkpoint=True` **只作用于 `token_transformer_layers` 与 `interactions`**（`models/predictor.py:854-858`、`882-885`），**不作用于骨干**。

**决策：** 先依赖 D1 的自动剪枝（第 0..49 层无图），**默认不为骨干全局开启** gradient checkpointing；提供 `--grad_checkpoint_backbone {off,active,all}`，`active` 表示只对 LoRA 命中层开启（逐 block 设 `module.gradient_checkpointing = True`）。P5 实测后再决定默认值。

### D3 — 单 head：复用 `MultiHead`，不新写 head 模块

`MultiHead` + `MLPHead(out_dim=2)` 已能满足 `(mu, log_sigma2)`。affinity 通路配置：

```python
active_heads = ["affinity"]
head_specs = {"affinity": {"type": "mlp", "hidden_dim": 768, "depth": 3,
                           "dropout": 0.1, "out_dim": 2}}
```

`KineticsPredictor.forward` 在 `task == "affinity"` 时返回 **2 元组 `(mu, log_sigma2)`**，每项形状 `(B,)`。

### D4 — affinity 分支不得构造任何"无用但 requires_grad"的参数

DDP 在 `find_unused_parameters=False` 下遇到"参与建图但无梯度"的参数会报错。因此 `task == "affinity"` 时：

- **不构造** `s_kcat` / `s_Km` / `s_ratio`
- **不构造** `s_domain_task`
- **不构造** `kcat_source_residual`
- 不使用 `bias_tuple` / `caps` / `floor_m`

这些用 `if self.cfg.task == "kinetics":` 包裹（而不是构造后设 `requires_grad=False`，避免 state_dict 出现无用键）。

### D5 — 结构分支：保持代码、运行时关闭

不传 `--use_struct_branch`、不传 `--final_contact_npz`、`struct_in_dim_*` 不参与。`AvailabilityAwareFusion` / `QualityAwarePhysFusion` / `lambda_gate_*` / `struct_dropout_prob` / `lambda_cons` 在 affinity 通路下全部惰性，不做代码改动。

### D6 — 损失：泛化 `kcat_expert_loss`，不重写

`models/kcat_expert.py:276-332` 的 `kcat_expert_loss` 已是单任务形式，抽出为 `models/affinity_loss.py::affinity_loss`，保留 Huber/Barron 二选一 + ranking + var_reg + NLL，去掉 CCC（可选保留）。

### D7 — 数据：单标签 Dataset + 单列 Scaler

亲和力标签已是 `-log10` 空间，无需 `KineticsScaler` 的 log10 变换与 ratio 分位数裁剪，只需一个单列 `StandardScaler`。

### D8 — DDP 契约：只改启动层，不改同步机制

`torch.distributed.run` 已通过环境变量注入 `RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT`，而 `main()` 已经在读这些（`main_train_predictor_multigpu.py:2446-2458`）。因此：

- **训练脚本不需要新增 rendezvous 参数**
- 改动集中在：新启动脚本、k8s PyTorchJob yaml、配置环境变量化、梯度累积

**不引入** DeepSpeed/Lightning/FSDP。保留 `torch.nn.parallel.DistributedDataParallel`。

---

## 6. 组件设计

### 6.1 新增 `models/plm_backbones.py`

骨干加载与 LoRA 注入的唯一入口，消除对具体骨干命名的假设。

```python
KIND_ESMC      = "esmc"        # AutoModelForMaskedLM + trust_remote_code + esm.EsmcTokenizer
KIND_MOLFORMER = "molformer"   # AutoModel + trust_remote_code + AutoTokenizer (BERT 系)
KIND_ESM2      = "esm2"        # 兼容旧路径：EsmModel / EsmTokenizer
KIND_T5        = "t5"          # 兼容旧路径：T5EncoderModel / T5Tokenizer

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    kind: str
    trust_remote_code: bool
    max_length: int
    hidden_size: int
    num_layers: int

def detect_kind(name: str, config=None) -> str
    """按 name 关键字与 config.model_type 判定；未知时抛错并列出已支持类型。"""

def resolve_hidden_size(config) -> int
    """依次尝试 hidden_size / d_model / embedding_dim / n_embd / dim；全失败则抛错。"""

def resolve_num_layers(config, model=None) -> int
    """依次尝试 config.num_hidden_layers / num_layers / n_layers / n_layer，
    再回退到 len(model.encoder.layer) / len(model.encoder.block) / len(model.layers)。"""

def load_protein_backbone(name: str, *, trust_remote_code: bool | None = None
                          ) -> tuple[nn.Module, object, BackboneSpec]
def load_ligand_backbone(name: str, *, trust_remote_code: bool | None = None
                         ) -> tuple[nn.Module, object, BackboneSpec]
    """返回 (model, tokenizer, spec)。ESMC 的 tokenizer 走 esm.EsmcTokenizer。"""

def forward_hidden_states(backbone, inputs, *, need_hidden_states: bool = True
                          ) -> torch.Tensor
    """统一取 token 级 hidden state：
       1. 先尝试 outputs.last_hidden_state；
       2. 非有限或不存在时，改以 output_hidden_states=True 再前向一次，取 hidden_states[-1]；
       3. 两者都不可用则抛错，并打印 output 的实际属性名。
       返回值经 torch.nan_to_num，对齐 LigandEncoder 现有行为（models/encoders.py:109）。"""

def build_token_mask(tokenizer, inputs, *, strip_special: bool = True) -> torch.Tensor
    """从 attention_mask 出发，剔除 BOS/EOS。
       优先 tokenizer.get_special_tokens_mask；不可用时回退到
       tokenizer.build_inputs_with_special_tokens 推断首尾特殊 token 数。
       ESMC 必须 strip_special=True（约束 7）。
       若推断出的特殊 token 数 ∉ {0, 2}，告警但继续（不静默）。"""

def resolve_lora_layers(num_layers: int, policy: str, last_k: int,
                        explicit: Sequence[int] | None = None) -> list[int]
    """policy ∈ {all, last_k, explicit}；last_k 按 min(last_k, num_layers) clamp。"""

def discover_lora_targets(model: nn.Module, *,
                          layers: Sequence[int] | None = None,
                          suffixes: Sequence[str] | None = None,
                          min_hits: int = 1) -> list[str]
    """通用层名发现，取代 _build_esm_targets_exact / _build_t5_encoder_targets_exact。

    算法：
      1. 取所有 nn.Linear 的完整限定名。
      2. 末段命中后缀白名单（默认覆盖 ESM-2 / T5 / ESMC / BERT-MoLFormer 两套命名）：
         query key value q k v o wi_0 wi_1 wo dense out_proj Wqkv
         q_proj k_proj v_proj o_proj fc1 fc2
      3. 用正则 r"\\.(?:layer|layers|block|blocks|h)\\.(\\d+)\\." 抽取层号（取最后一次匹配）。
      4. layers 非 None 时只保留层号命中的项；无层号的项在 layers is None 时保留。
      5. 结果为空则抛 AssertionError，并把"实际见过的末段名集合"打印出来便于定位。
    """

def inject_lora(model: nn.Module, *, r: int, alpha: int, dropout: float,
                policy: str, last_k: int,
                explicit_layers: Sequence[int] | None = None
                ) -> tuple[nn.Module, list[str], list[int]]
    """封装 peft.LoraConfig + get_peft_model，返回 (peft_model, targets, layers)。
       task_type="FEATURE_EXTRACTION"，bias="none"。
       断言 LoRA 张量数 > 0。"""

def set_backbone_gradient_checkpointing(model: nn.Module, mode: str,
                                        layers: Sequence[int] | None = None) -> int
    """mode ∈ {off, active, all}。active 时逐 block 设 gradient_checkpointing 标志。
       返回被开启的 block 数。"""
```

### 6.2 新增 `models/affinity_loss.py`

```python
def affinity_loss(
    mu: torch.Tensor,            # (B,)
    log_var: torch.Tensor,       # (B,)
    y: torch.Tensor,             # (B,)
    *,
    loss_type: str = "huber",    # {"huber", "barron"}
    huber_beta: float = 1.0,
    barron_alpha: float = 1.0,
    barron_scale: float = 1.0,
    nll_weight: float = 0.05,
    rank_weight: float = 0.0,
    rank_pairs: int = 256,
    rank_margin: float = 0.0,
    var_reg_weight: float = 0.0,
    var_reg_mode: str = "abs",   # {"abs", "ratio"}
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
```

数值稳定性要求（照搬 `kcat_expert_loss` 的做法）：入参先 `nan_to_num`，`log_var` clamp 到 `[-10, 10]`，mask 掉非有限标签。

### 6.3 新增 `data_affinity.py`

```python
class AffinityScaler:
    """单列标准化。标签已在 -log10 空间，不做 log 变换、不做分位数裁剪。"""
    def fit(self, y: np.ndarray) -> "AffinityScaler"
    def transform(self, y: np.ndarray) -> np.ndarray          # -> (N,)
    def inverse_transform(self, z: np.ndarray) -> np.ndarray
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> "AffinityScaler"

class AffinityDataset(Dataset):
    """列约定：Sequence | Smiles | <label_col>，可选 source_id / phys_*。
    label_col 默认 "pKaffinity"，可配。
    非有限标签的行在构造时剔除。
    __getitem__ -> (seq: str, smi: str, y: torch.Tensor)   # 3 元组
    """
```

**为什么是 3 元组：** `utils.py:771-773` 的 `unpack_batch` 对 `len(batch) == 3` 已返回 `row_ids=None, source_ids=None, p_s=None, l_s=None`，affinity 通路不需要它们，零改动即可对接。

### 6.4 修改 `models/encoders.py`

- `ProteinEncoder` / `LigandEncoder` 的构造签名改为接受 `BackboneSpec`（或退化为 `name` + `kind`），内部调用 `plm_backbones.load_*`，不再自行 `from transformers import EsmTokenizer` 等
- `hidden = resolve_hidden_size(backbone.config)`
- `max_length` 来自 `spec.max_length`（ESMC 可配，MoLFormer **202**）
- `encode_tokens` 改为调用 `forward_hidden_states` + `build_token_mask`（`strip_special=True`）
- 保留 `train_backbone` 语义与 LoRA 参数筛选逻辑（`models/encoders.py:80-95`）

### 6.5 修改 `models/predictor.py`

1. `PredictorConfig` 新增：`task: str = "kinetics"`、`lora_layer_policy: str = "all"`、`lora_last_k: int = 30`
2. `KineticsPredictor.__init__`：
   - `self.esm = AutoModel.from_pretrained(...)` / `self.molt5 = AutoModelForSeq2SeqLM.from_pretrained(...)`（`models/predictor.py:480-481`）→ 改为 `load_protein_backbone` / `load_ligand_backbone`
   - `models/predictor.py:533-664` 整段 LoRA 逻辑 → `inject_lora(...)`
   - `models/predictor.py:676` `self.molt5.get_encoder()` → `spec.kind == KIND_T5` 时才调用，否则直接传 backbone
   - `models/predictor.py:766-772` 的 `s_kcat`/`s_Km`/`s_ratio`/`s_domain_task` 用 `if cfg.task == "kinetics"` 包裹（D4）
   - `models/predictor.py:741-750` 的 `kcat_source_residual` 同上
3. `KineticsPredictor.forward`：新增 `task == "affinity"` 早返回分支，返回 `(mu, log_sigma2)`；不改 kinetics 分支的任何一行
4. `KineticsPredictor.saliency`：补 affinity 分支（或对 affinity 抛 `NotImplementedError`）

### 6.6 修改 `main_train_predictor_multigpu.py`

新增 CLI：

```
--task {kinetics,affinity}          默认 kinetics
--lora_layer_policy {all,last_k,explicit}
--lora_last_k INT                   默认 30
--lora_explicit_layers INT [INT ...]
--plm_max_len_prot INT
--plm_max_len_lig INT
--grad_checkpoint_backbone {off,active,all}
--affinity_label_col STR            默认 pKaffinity
--affinity_loss_type {huber,barron}
--iters_to_accumulate INT           默认 1
```

按 `args.task` 分支的位置（**kinetics 路径保持逐字节不变**）：

| 行 | 分支内容 |
|---|---|
| `2610-2649` | `PredictorConfig` 构造：`active_heads` / `head_specs` / `task` |
| `2535-2538` | scaler：`KineticsScaler` vs `AffinityScaler` |
| `2542-2588` | Dataset/DataLoader：`EnzymeDataset` vs `AffinityDataset` |
| `2479-2480` | `normalize_training_dataframe` 是否适用（affinity 走独立列规范化） |
| `1239-1250` | `train_step` 的 forward 解包 |
| `1257-1401` | 三任务损失 → affinity 走 `affinity_loss` 的薄分派 |
| `1554` | `PredictionRecorder(output_dim=3)` → affinity `output_dim=2` |
| `1666-1910` | 验证指标：affinity 用 Pearson / Spearman / RMSE / MAE / NLL |
| `2726-2750` | `--eval_only` 与 test loader |
| `2793+` | 测试阶段预测写出 |

DDP 相关（`2678-2704`）：affinity 下 `find_unused_parameters=False`、`static_graph=True`；若实测报错再回退 `find_unused_parameters=True`。

**梯度累积：** 新增 `--iters_to_accumulate`，在 `train_step` 外按 `(step + 1) % iters_to_accumulate == 0` 控制 `optimizer.step()`；`no_sync()` 的判定从 `(t + 1) < steps`（`main_train_predictor_multigpu.py:1234`）改为同时考虑累积窗口。

**环境变量化（k8s 友好）：**

| 变量 | 用途 | 默认 |
|---|---|---|
| `KINEAGER_NUM_WORKERS` | 替换 `num_workers=8` / `4` 硬编码 | 8 / 4 |
| `KINEAGER_ITERS_TO_ACCUMULATE` | 梯度累积 | 1 |
| `KINEAGER_OUT_DIR` | 输出根目录 | 脚本默认 |
| `KINEAGER_ESMC_PATH` / `KINEAGER_MOLFORMER_PATH` | 骨干权重本地路径 | 必填 |

### 6.7 新增 `bin/train/kineager_k8s.sh`

对齐 xtrimofold3 契约（`bin/train/run_k8s_protenix_v2_multitask_PL_k8s.sh` 的结构）：

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 逐个校验为非负整数；nnodes >= 1；node_rank < nnodes；nproc_per_node >= 1
# NNODES > 1 时 MASTER_ADDR 必须已注入（PyTorchJob 会注入）
# 否则 MASTER_ADDR 默认 127.0.0.1

export TASK="${TASK:-affinity}"
export KINEAGER_ESMC_PATH="${KINEAGER_ESMC_PATH:?must point to a local ESMC-6B directory}"
export KINEAGER_MOLFORMER_PATH="${KINEAGER_MOLFORMER_PATH:?must point to a local MoLFormer snapshot}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

exec python3 -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${ROOT}/main_train_predictor_multigpu.py" \
        --task "${TASK}" \
        -csv "${TRAIN_CSV:?must point to the PDBbind affinity CSV}" \
        -esm "${KINEAGER_ESMC_PATH}" \
        -molt5 "${KINEAGER_MOLFORMER_PATH}" \
        --lora_layer_policy last_k \
        --lora_last_k "${LORA_LAST_K:-30}" \
        --iters_to_accumulate "${ITERS_TO_ACCUMULATE:-1}" \
        "${@}"
```

训练超参（`-epochs`、`-batch_size`、`-lr`、`--split_*` 等）一律由调用方通过 `"${@}"` 追加，脚本本身不写死。

### 6.8 新增 `yamls/kubeflow/kineager_affinity_k8s.yaml`

对齐 `yamls/kubeflow/xf3_pl_multitask_k8s.yaml` 的 PyTorchJob 结构：`Master.replicas: 1` + `Worker.replicas: {{worker_num}}`，容器内注入 `NNODES`/`NODE_RANK`/`NPROC_PER_NODE=1`/`MASTER_PORT`，`bash_command` 参数指向 `bin/train/kineager_k8s.sh`。

### 6.9 新增 `tools/build_pdbbind_affinity_csv.py`

输入：`pdbbind_2020_feats.csv`、`PDBbind_v2020/P-L/<pdb_id>/`、可选的既有划分 CSV。
输出列：`pdb_id, Sequence, Smiles, pKaffinity, affinity_type, source_id`。

规则：
- 标签优先级 **Kd > Ki > IC50**，统一取 `-log10`
- 每个 `pdb_id` 一行（复合物级去重；链/界面级多行折叠）
- 蛋白序列取自 `*_protein.pdb`，配体 SMILES 取自 `*_ligand.sdf`（RDKit canonical，去立体/同位素，与 MoLFormer 输入口径一致）
- 排除约束 11 的 37 个样本
- 可选按既有划分写入 `split` 列

### 6.10 其它修改

| 文件 | 改动 |
|---|---|
| `main_infer_predictor.py:303-305` | 加 `--task affinity` 单路推理分支：`mu, log_sigma2 = model(...)`，输出 `pred_affinity` / `sigma` |
| `path_config.py` | 新增 `k8s` profile；新增 `esmc_model_path` / `molformer_model_path` 键与 `CATA_ESMC_MODEL_PATH` / `CATA_MOLFORMER_MODEL_PATH` 覆盖 |
| `requirements.txt` | 加 `peft`、**`esm`**（ESMC tokenizer 必需）；pin `transformers` |
| `README.md`、`PAPER_REPRODUCTION.md` | 补 PDBbind 亲和力任务的运行说明 |

---

## 7. 数据流（affinity 通路）

```
PDBbind v2020 CSV  (pdb_id, Sequence, Smiles, pKaffinity)
      │
      ├─ AffinityScaler.fit(train split)         # 单列标准化，落盘 scaler_affinity.json
      │
      ▼
AffinityDataset  ──(seq, smi, y)──►  DataLoader ──► DistributedSampler ──► DDP rank k
      │
      ▼  train_step
model(seqs, smiles, use_mask, task=affinity)
      │
      ├─ ProteinEncoder ──► ESMC-6B (LoRA 50..79) ──► proj(hidden=2560 → C) ──► p_tok, p_mask
      ├─ LigandEncoder  ──► MoLFormer (LoRA 0..11) ─► proj(hidden=768  → C) ──► l_tok, l_mask
      │
      ├─ token_transformer_layers (checkpoint)
      ├─ ResidueMask / LigandMask  (gate_p, gate_s)
      ├─ interactions: 2 stages × 3 CrossAttentionBlock
      ├─ masked_mean 池化 ──► p_vec, l_vec
      ├─ (结构分支 / 物理证据分支: 关闭)
      ├─ cat[p_vec, l_vec] ──► LayerNorm ──► Linear(2C → 512)
      └─ MultiHead["affinity"] (MLP, out_dim=2) ──► mu, log_sigma2
      │
      ▼
affinity_loss(mu, log_var, y) ──► backward ──► 只回传到 LoRA(50..79) + proj + interactions + head
```

---

## 8. 错误处理

| 场景 | 行为 |
|---|---|
| LoRA `target_modules` 零命中 | `discover_lora_targets` 抛错，并打印实际见过的 Linear 末段名集合 |
| 骨干 `hidden_size` 字段未识别 | `resolve_hidden_size` 抛错，并列出 config 的全部字段名 |
| 骨干层数无法推断 | `resolve_num_layers` 抛错 |
| `esm` 包缺失（ESMC 通路） | 加载时抛出带安装提示的 `ImportError` |
| 标签列缺失或全为非有限值 | `AffinityDataset` 构造时抛错并报出标签列名与有效行数 |
| affinity 通路下 `task` 与 `active_heads` 不一致 | `PredictorConfig` 构造时断言 |
| DDP 下有参数无梯度 | 启动时打印 trainable 参数名分组；`find_unused_parameters` 报错则回退 `True` 并告警 |
| ESMC tokenizer 特殊 token 未剔除 | `build_token_mask` 断言特殊 token 数 ∈ {0, 2}，异常时告警 |

---

## 9. 测试策略

本地**无 GPU、无 CUDA、未装 peft/esm**，因此：

**CPU 单元测试（本地，pytest）：**
1. `resolve_hidden_size` / `resolve_num_layers` 对多套伪造 config（ESM-2 / T5 / BERT / 含 `d_model` / 含 `embedding_dim`）的取值
2. `resolve_lora_layers` 的 `all` / `last_k`(clamp) / `explicit` 三策略边界
3. `discover_lora_targets` 对**伪造的 ESM-2 命名模型**与**伪造的 BERT 命名模型**各自命中预期层；零命中时抛错
4. `AffinityScaler` 的 `fit → transform → inverse_transform` 往返误差
5. `AffinityDataset` 的标签剔除、列名回退、`__getitem__` 形状
6. `affinity_loss` 在 NaN / 极端 `log_var` / 单样本 batch 下的输出有限性
7. `forward(task="affinity")` 输出形状 `(B,)` × 2；**参数计数对比**确认不含 `s_kcat`/`s_Km`/`s_ratio`/`s_domain_task`
8. `unpack_batch` 对 3 元组的解包
9. `build_pdbbind_affinity_csv.py` 的标签优先级与去重（用小型 fixture CSV，不依赖真实 PDBbind）

**P0 探针（需集群或能装 `esm` 的环境）：**
- 打印 `biohub/ESMC-6B` 与 MoLFormer 的实际模块结构、全部 `nn.Linear` 限定名、`config` 字段、真实层数、hidden 维度
- 用探针输出固化 `discover_lora_targets` 的默认后缀白名单，并断言 last-30 命中预期数量
- **这是阻塞项**：探针未通过前，P1 的 LoRA 代码不可宣称完成

**集群 smoke（P5）：**
- 2 卡（`NNODES=2, NPROC_PER_NODE=1`）跑 200 步小样本子集
- 验证：梯度同步（各 rank 参数一致）、checkpoint 存取、EMA、`bf16` 数值、单步显存占用
- 记录：单步耗时、峰值显存、LoRA 命中层数与可训练参数量

---

## 10. 分阶段

| 阶段 | 内容 | 出口条件 |
|---|---|---|
| **P0 探针** | 打印 ESMC-6B / MoLFormer 真实结构 | 探针报告产出，`target_modules` 白名单确定 |
| **P1 骨干与 LoRA** | `plm_backbones.py` + `encoders.py` 改造 | CPU 单测通过；旧 ESM-2/MolT5 路径回归通过 |
| **P2 数据** | `data_affinity.py` + `tools/build_pdbbind_affinity_csv.py` | 单测通过；能在集群上产出 CSV 并统计标签分布 |
| **P3 单 head + 损失** | `affinity_loss.py` + `predictor.py` 分支 + 训练脚本 `--task` 分支 | 单测通过；`task=kinetics` 路径逐字节未变 |
| **P4 DDP/k8s 契约** | 启动脚本 + PyTorchJob yaml + 环境变量化 + 梯度累积 | 脚本 `shellcheck` 通过；dry-run 输出正确命令 |
| **P5 集群 smoke** | 2 卡小样本实跑 | 梯度同步与 checkpoint 验证通过；记录耗时/显存 |

---

## 11. 风险与未决

| 风险 | 影响 | 缓解 |
|---|---|---|
| ESMC-6B remote-code 内部存在 bf16 不兼容算子 | 训练崩溃或静默数值错误 | P0 探针 + P5 smoke 先跑；必要时对特定模块 `autocast(enabled=False)` |
| last-30 层 LoRA 的首层输入需要 grad（否则第 0..49 层仍会建图） | 显存未按预期下降 | P5 实测峰值显存；必要时在第 50 层入口做 `.detach()` 断面 |
| 6B 骨干 × 每卡一份 + DDP 广播 | 单 pod 显存与启动时间 | 记录启动耗时；`broadcast_buffers=False` |
| ESMC-6B 在线前向吞吐不足 | 训练周期过长 | 梯度累积放大有效 batch；必要时缩小 `max_len` 或改 ESMC-600M（配置项已留） |
| PDBbind 标签类型混杂（Kd/Ki/IC50 混合训练） | 标签噪声 | 保留 `affinity_type` 列，可用 `source_id` 做域不确定性；先按统一值训练，后续做分型消融 |
| 与 xtrimofold3 现有亲和力结果不可比 | 结论无法对照 | 复用其 `pdbbindv2020_train_covered_minus_val.csv` / `pdbbindv2020_val_covered.csv` 划分 |
| `main_train_predictor_multigpu.py` 3049 行、改动面广 | 回归风险 | `task=kinetics` 分支逐字节不变；每个任务结束跑一次 kinetics CPU smoke |

**未决：**
- ESMC 蛋白 `max_length` 取值（候选：1024 / 1536 / 2048）—— 需看 PDBbind 序列长度分布后定
- affinity 通路是否保留 `ResidueMask`/`LigandMask` 门控的 `l1`/`tv` 正则（当前设计：保留门控，正则权重默认 0）
- 是否启用 `--domain_uncertainty` 按 `affinity_type` 分域

---

## 12. 明确不改

- `models/interactions.py` 全文
- `models/kcat_expert.py`、`models/moe_kcat.py`、`main_train_kcat_expert.py`、`main_infer_ensemble.py`
- 结构分支全部代码：`final_contact.py`、`tools/build_struct_npz.py`、`tools/extract_geom_feats*.py`
- `models/predictor.py` 的 kinetics forward 分支、三任务异方差损失、`compute_multihead_loss`
- `utils.py::KineticsScaler`、`data.py::EnzymeDataset`
