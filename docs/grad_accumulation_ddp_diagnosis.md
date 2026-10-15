# 梯度累积在 DDP 下触发的问题 —— 诊断记录

适用代码：`main_train_predictor_multigpu.py`（含 2026-10-13 的累积实现与修复，以及
2026-10-15 的配置固定）。本文回答一个问题：**为什么本部署固定
`ITERS_TO_ACCUMULATE=1`（不做梯度累积）**。

原始出处（先列证据，后讲机制）：

- `147e475`（2026-10-15，把任务配置固定为稳定版的提交）原文：
  > `ITERS_TO_ACCUMULATE=1 (gradient accumulation with DDP hits the`
  > `expect_autograd_hooks_ assert and epoch-boundary hangs)`
- `docs/kineager_affinity_runbook.md` §0.5 沿用同一结论（"累积 + DDP 会触发
  `expect_autograd_hooks_` 断言与 epoch 边界卡死"）。
- 2026-10-13 的设计评审（SDD rulings，全文索引见 §7）逐条给出了代码级根因与修复。

---

## 1. 现象（触发条件与两种表现）

**触发条件**：多卡 DDP（`world_size > 1`；当时的测试是 2 卡本地） + `-iters_to_accumulate > 1`。
单卡（无 DDP）不触发——问题完全出在 DDP 的梯度归约契约上。

**表现**（两种，可单独或同时出现）：

1. **DDP reducer 断言（直接崩溃）**：
   `RuntimeError: Expected to have finished reduction in the prior iteration before
   starting a new one. ...` ——即 PyTorch Reducer 里的 `expect_autograd_hooks_` 检查
   失败（"上一轮的 DDP 归约还没走完，就开始了下一轮"）。
2. **epoch 边界卡死（无报错停滞）**：训练不再前进、GPU 空转；最终由 NCCL watchdog
   超时终止（`Watchdog caught collective operation timeout ... ran for 600007
   milliseconds before timing out`）。

## 2. 机制：为什么累积会踩到这两个坑

DDP 有两条硬约束，累积恰好同时触碰它们：

- **(A) 归约必须完整**：每个 rank 每轮 backward 开始时，Reducer 置位
  `expect_autograd_hooks_`，必须等**该轮全部注册参数的 autograd hook 触发完**
  （本轮梯度 allreduce 完整走完）才复位；否则下一次 forward/backward 直接触发断言。
- **(B) 集合通信序列必须逐 rank 一致**：任何 rank 多一次/少一次 allreduce、或顺序不同，
  对端就会永远等下去——表象就是"卡死"，直到 watchdog 超时。

梯度累积把"一批一次 forward/backward/归约"变成"**窗口内多批推进、窗口末尾一次
step + 一次归约**"。于是下面三处必须严格对齐，错一处就命中 (A) 或 (B)：

1. **窗口判定层级**：必须按 *optimizer step 序列*（`train_epoch` 的 batch 循环）判定，
   不能用批内**微批索引**判断；
2. **梯度清零时机**：只能在真正 `step()` 之后清，否则会清掉窗口里已攒下的梯度；
3. **`no_sync()` 层级**：窗口内部整批 `model.no_sync()`（不归约），窗口关闭那一批才归约。

## 3. 代码级根因与修复（2026-10-13 设计评审）

当时的实现恰好在这三处上都有缺陷，另有一批"不崩但结果错误"的记账户问题：

| 编号 | 问题 | 修复 |
|---|---|---|
| **R31** | 窗口判定挂在 `train_step` 内部的**微批**索引 `t/steps` 上（`steps = (B+mb-1)//mb`），不是 optimizer step 序列；且 `train_step` 无条件 `optimizer.zero_grad(set_to_none=True)`，天真的按批门控会**清掉上一批攒下的梯度**（累积白做） | 窗口判定上移到 `train_epoch`（`should_step` / `should_sync_grad`）；`train_step` 增加 `do_step` / `do_sync` 开关；窗口内部 `model.no_sync()` |
| **R46** | 尾部"不满窗口"的契约：epoch 最后一批必须 flush | `should_step` 在 `is_window_end or is_last_step` 才为 True（测试把 `[F,F,T,F,F,T,T]` 钉死）；否则尾批梯度被**静默丢弃** |
| **R41** | loss 归一化只对 affinity 除以 `iters_to_accumulate`，kinetics 没除 → 同一 flag 值下两任务有效学习率差 k 倍 | 在反传点统一 `loss = loss_mb / (steps * accum)`（`/steps` 是微批归一，`/accum` 是累积归一） |
| **R47** | `global_step` / LR 调度 / `ema.update()` / `_mse_ema` 仍**逐批**推进 → 累积 k 时 LR 计划提前 k 倍走完、EMA 多更新 k 倍（当时测试的 RED：`optimizer.step=3` 而 `scheduler.step=7`、`ema=7`） | 记账与 optimizer step 同频（`if do_step:` 门控） |
| **R51** | 只把计数器挂到 step 上还不够：计划长度若仍按 batch 计，LR 会"**永远退火不到底**"（从"提前 k 倍结束"变成"永不结束"） | 计划长度改用 `optimizer_steps_per_epoch(B,k) = ceil(B/k)`（`accum=1` 时与旧式逐字等价） |
| **R52** | tqdm / swanlab 的每批记录保留（窗口期内的心跳，非正确性问题） | 维持（`lr_step` 会有 k 条重复值，属仪表盘噪声） |
| **R53** | "老 checkpoint 的 `global_step` 按批计、新语义按 step 计"的兼容担忧 | 不成立：累积是新功能，所有旧 checkpoint 都由 `accum=1` 产生，两种计数恒等 |
| **R54** | OneCycle 非调度组的 `base_lr` 钉在 gate 内会被漏掉 | 提升到 gate 外 |

以上修复落在 `f5ed603`（窗口/no_sync/清零时机/R41 loss 归一/`broadcast_buffers=False`）
与 `1cc8dab`（R47 记账 + R51 计划长度）。

## 4. 修复之后的现状与"为什么仍然不用"

- **代码侧**：R31/R41/R46/R47/R51/R52/R53/R54 全部实现，并有单测
  （`tests/test_predictor_import.py` 的 Task 16 段落：窗口语义、`no_sync` 开关、
  R41/R47 的计数）。
- **配置侧**：**仍然固定 `ITERS_TO_ACCUMULATE=1`**（launcher / yaml / wrapper 三处
  默认都是 1；理由是 `147e475` 记录的现场故障）。可以理解为保守取舍：该链路
  （窗口 + `no_sync` + DDP `static_graph` / `find_unused_parameters` 的组合）在修复后
  没有以生产配置重新端到端验证过（runbook §6 清单第 9 条就是留给它的验证），而本部署
  扩大有效 batch 的主力手段是**多卡**，不需要累积。
- **要开启（或改动窗口/同步相关代码）时**，按 runbook §6 清单第 9 条做端到端验证：
  同一份配置 `ITERS_TO_ACCUMULATE=2` 对比 `1`——
  ① optimizer step 次数 ≈ `ceil(batches/2)`；
  ② LR 真的随计划退火（不提前结束、不停住）；
  ③ 至少整一个 epoch 无断言、无边界卡死。

## 5. 排查对照表（以后再见到这些字符串先查什么）

| 症状 / 报错 | 含义 | 先查 |
|---|---|---|
| `Expected to have finished reduction in the prior iteration...`（`expect_autograd_hooks_`） | 上一轮 DDP 归约未完成就开始了下一轮 | 是否有 forward 没配对的 backward；是否有参数没参与 loss（`--find_unused_parameters`）；`no_sync()` 用在哪一层；累积窗口两侧是否一致 |
| `Watchdog caught collective operation timeout ... ran for 600007 milliseconds` | 某 rank 的集合通信一直没人配 | **对端 rank 是不是已经死了**（先看对方有没有 OOM/异常退出）；两边 collective 序列/次数是否一致 |
| LR 提前走完 / 一直不退火 | 计划长度与 optimizer step 不同频 | `optimizer_steps_per_epoch` 的三处接线（见 §7 代码索引） |

## 6. 易混淆项：2026-10-15 的两次本地崩溃**不是**累积问题

`run_logs/affinity_local_v1_20261015_113043.log` 与 `_121159.log`（本地 2 卡）的崩溃
模式是 **rank0 CUDA OOM → rank1 在集合通信里等待 600s → NCCL watchdog abort**：

- 根因是 best-save（EMA 备份）导致的 rank0 显存泄漏（验证改善后从 ~52GB 涨到
  ~78GB，下一个长序列 batch 即 OOM），已由 `759a65b` 修复；**与梯度累积无关**。
- 判据：日志里有 `torch.OutOfMemoryError ... Process ... has 78.09 GiB memory in
  use`；而累积问题**不需要 OOM** 就能触发（断言/边界卡死）。
- 两者表象都有"卡死"成分（都可能经由 NCCL 等待），容易混为一谈——按
  "先看对端 rank 是否已死亡 → 先排除 OOM"的顺序排查即可区分。

## 7. 索引

- **提交**：`f5ed603`（累积主实现）、`1cc8dab`（R47/R51）、`147e475`（把配置固定为
  `ITERS_TO_ACCUMULATE=1` 并记录上述现象）、`759a65b`（best-save 泄漏修复）。
- **代码**（`main_train_predictor_multigpu.py`）：`should_step` L1222、
  `should_sync_grad` L1235、`optimizer_steps_per_epoch` L1243；`train_step` 的
  `do_step`/`do_sync` 与 R31/R41 注释（L1327 起、L1462、L1742）；`train_epoch`
  门控（L1849-1850）；R47 记账（L1897-1913）；LR 计划接线（L2436-2440）。
- **测试**：`tests/test_predictor_import.py` Task 16 段落
  （`test_should_step_*`、`test_should_sync_grad_*`、
  `test_train_step_do_step_false_accumulates_without_stepping` 等）。
- **设计评审全文**（工作区已删除，保留在 git 历史里）：
  `git show 9ac46f8^:docs/superpowers/plans/2026-10-13-kineager-pdbbind-affinity-rulings.md`
  §5（R31/R41/R44/R46/R47/R51–R54）。
- **runbook**：`docs/kineager_affinity_runbook.md` §0.5（配置结论）、§6 清单第 9 条
  （累积的端到端验证方法）。
