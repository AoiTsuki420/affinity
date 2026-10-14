# Ruling appendix — KinEAGER PDBbind v2020 affinity (R1–R70)

Controller rulings made while executing
`docs/superpowers/plans/2026-10-13-kineager-pdbbind-affinity.md` on the
`feat/pdbbind-affinity` branch, transcribed out of the (git-ignored) SDD ledger
`.superpowers/sdd/2026-10-13-kineager-pdbbind-affinity/progress.md`.

Scope: **R1–R61** are the execution rulings this appendix was commissioned for.
**R62** (the test-dependency/import decision), **R63–R68** (the task-review and
documentation rounds that amended R61's guard and this appendix's own statements)
and **R69–R70** (the whole-branch review's two CRITICAL findings) were recorded
afterwards and are marked as such where they appear.

This appendix exists because the rulings are the part of the work that is hardest
to re-derive: several of them record failures that the unit tests of the day
could **not** have caught — a green suite over a code path that aborts on the
first real backbone, a silently shared output directory, a loss that is not the
loss its name claims. Read an entry's *reason* before changing the code it
protects, and read the supersession list at the end before acting on any ruling
that a later one amended.

Format: **ruling number — what was ruled.** The concrete evidence or failure mode
that motivated it. Where a later ruling supersedes or corrects an earlier one,
the later entry says so, and §9 collects those links.

---

## 1. Environment and process

- **R4 — the SDD helper scripts were ported to Python.** `scripts/sdd-workspace`,
  `scripts/task-brief` and `scripts/review-package` are bash, and bash cannot
  start at all under this host's file sandbox (`couldn't create signal pipe,
  Win32 error 5` — confined processes may not open named pipes). Ported to
  `.superpowers/sdd/sdd.py` with identical semantics; the port is git-ignored
  scratch, so no project file changed.
- **R5 — one escalated `pip install`.** `utils.py` imports matplotlib and
  scikit-learn at module scope, so `models/predictor.py` cannot be imported
  without them, and `peft` gates Task 6's tests. Installing writes outside the
  workspace, which the file policy denies, so this was done once with approval.
- **R6 — the pytest cache is disabled for this run.** The sandbox leaves
  `.piptmp/` and `pytest-cache-files-*` directories whose ACLs make them
  undeletable even by their owner, and the cache provider recreates them on every
  run. Hence `python -m pytest -p no:cacheprovider …` everywhere.
- **R10 — implementers are dispatched in the background.** A foreground subagent
  call returns only the result text, not a continuable agent id, so fix rounds
  could not resume the original implementer's context. From Task 2 on, every
  implementer is dispatched in the background; Task 1's fix used the documented
  fallback (a fresh implementer carrying the brief, the report and the findings).
- **R12 — tests must not use pytest's `tmp_path` fixture.** Measured: every
  `tmp_path` test ERRORs here with `PermissionError: [WinError 5]` while pytest
  scans its basetemp, and redirecting the basetemp into the workspace fails
  identically. The cause is the **parent** directory's sandbox grant —
  `os.scandir(".pytest-tmp")` fails even though that directory reads `0o777` —
  while a fresh top-level directory works, nested ones included. Rule: take a
  repo-local scratch directory from a fixture that uses `Path.mkdir()` and remove
  it with `shutil.rmtree` in a `finally`.
- **R19 — do not normalise line endings in this worktree.** The host has
  `core.autocrlf=true`; the index blobs were verified byte-level pure LF (0 CRLF)
  even where the working tree is CRLF, so git normalises on commit, `git status`
  stays clean, and no effort should be spent on it.
- **R26 — Tasks 7 and 8 are dispatched as one batch.** Both are halves of one
  refactor (decoupling the model layer from hardcoded ESM-2/MolT5 backbones) with
  a single coherent review surface, so one review gate covering two briefs is
  cheaper than an extra full cycle.

## 2. Instrumenting the cluster-only unknowns

- **R7 — the Task 1 `_Cfg` fixture amendment is approved.** The plan's fixture
  used *class* attributes, so `vars(_Cfg())` is `{}` and
  `test_describe_backbone_reports_config_fields` could never pass for any
  implementation matching the plan's own `config_fields = sorted(vars(config))`.
  Real HuggingFace configs carry their fields on the instance `__dict__`.
- **R9 — the probe must work under any invocation form.** Running
  `python tools/probe_backbone_modules.py` puts `tools/` on `sys.path[0]`, so the
  deferred `from models.plm_backbones import …` can never resolve and the cluster
  probe prints `hidden_size: null` / `num_layers: null` — exactly the two values
  acceptance checks.
- **R13 — the probe must fail loudly but keep emitting its primary artifact.**
  `describe_backbone` caught bare `Exception`, buried the cause inside
  `layer_names_sample` (a field whose contract is layer names) and `main()` still
  returned 0, so a probe that resolved nothing looked like a successful one.
  Ruled: print the JSON and write `--out` **first**, then a `[PROBE-FAILURE]`
  block on stderr, then exit **2**. Sequencing: this lands in Task 2, not a Task 1
  fix round, because `main()` cannot return non-zero while
  `models/plm_backbones.py` does not exist — the subprocess regression test
  asserting exit 0 would go red permanently, and stubbing the resolver would
  destroy that test's discriminating power.
- **R14 — the `sys.path` bootstrap is scoped to direct execution.** A
  module-level insert at position 0 changes the whole pytest session's import
  precedence, letting repo-root `data.py`/`utils.py`/`tracing.py`/`path_config.py`
  shadow same-named installed distributions.

## 3. Model, backbone and LoRA plumbing

- **R1 — a module-level `importorskip("peft")` must not stand in a shared test
  module.** Placed after Tasks 2–5's tests it aborts the whole module at import,
  silently disabling ~40 unrelated tests on any machine without peft. Replaced by
  a `HAS_PEFT` try/except plus `skipif` marks on the seven peft-dependent tests;
  no `importorskip` **call** may appear in a `.py` file (the token itself survives
  as prose explaining this ruling, which is exactly why R62's check is AST-based).
- **R2 — the constructor must take and thread the new LoRA/PLM options.**
  `KineticsPredictor.__init__` gains `lora_layer_policy`, `lora_last_k`,
  `lora_explicit_layers`, `lora_target_suffixes`, `plm_max_len_prot`,
  `plm_max_len_lig`; `_build_backbones_and_lora` takes them and assigns the two
  tokenizers and two specs. Without this wiring the CLI flags Task 14 adds would
  be **inert**, which the spec's own §11 lists as the top risk. *(Completed by
  R36.)*
- **R11 — Task 6's LoRA tensor-count assertion is wrong.** Measured against peft
  0.21.2: peft creates one `lora_A` and one `lora_B` per targeted module, so a
  fixture with 6 targets produces **12** tensors, not the 6 the plan asserted —
  a deterministic failure. The same probe pinned the parameter arithmetic the
  plan relies on: 16·r per target.
- **R15 — gradient checkpointing must assign `_gradient_checkpointing_func`, not
  just set the boolean.** `EsmLayer` subclasses
  `transformers.modeling_layers.GradientCheckpointingLayer`, whose `__call__`
  ends in `self._gradient_checkpointing_func(...)` whenever the flag is set and
  the module is training; the base class defines no such attribute, so the plan's
  boolean-only `set_backbone_gradient_checkpointing` would abort at the first
  forward with `AttributeError` — **and Task 6's plain-`nn.Module` fixture would
  still pass**, a green suite over a fatal design. Ruled: assign
  `functools.partial(checkpoint, use_reentrant=False)` alongside the boolean, and
  make the fixture derive from the real base class with a real forward+backward.
- **R17 — Task 4's two deviations from the brief are approved.** (1) The layer
  regex needs a `(?:^|\.)` anchor: the brief's leading-dot-only form returns
  `None` for a root-level container (`layers.0.mixer`), making the brief's own
  custom-suffix test unsatisfiable. (2) `_fake_t5` was split so the FFN sits at
  `layer.1`, because real T5 does and the legacy code this replaces already
  assumes it (`layer.1.DenseReluDense.wi_0`). Same class of defect as Task 1's
  class-attribute fixture.
- **R18 — the spec overrides the plan's frozen suffix whitelist.** Spec §6.1
  lists `fc1 fc2`, which the plan's 19-entry `DEFAULT_LORA_SUFFIXES` omitted
  along with bare `query`/`key`/`value`, `wi_0`/`wi_1`/`wo`; extended to **27**.
  Bare `q`/`k`/`v`/`o`/`dense` stay deliberately excluded because they over-match
  unrelated modules, and `--lora_target_suffixes` is the designed escape hatch.
- **R20 — pass the union to `--lora_target_suffixes`, never the difference.**
  `discover_lora_targets`'s `suffixes` argument **replaces** the default whitelist
  (its tested contract, pinned in Task 4), so the plan's "take the difference"
  instruction would silently drop every default suffix for that backbone and
  attach LoRA to a near-empty target set. The flag's help text must say it
  replaces; narrowing stays possible for ablations.
- **R21 — `_load` must resolve the model config defensively.** With
  `trust_remote_code=True` the hub shim may hand back `esm`'s own ESMC object, a
  plain `nn.Module` with no `.config`, so the ESMC path would raise
  `AttributeError` instead of a diagnostic and ESMC LoRA would be unreachable.
  Fixed with `getattr(model, "config", None)` plus an in-`_load`
  `AutoConfig.from_pretrained` fallback — the same thing the probe already does.
- **R22 — tokenizer loads must honour the pinned revision and the resolved
  `trust_remote_code`.** `_load_molformer` hardcoded `trust_remote_code=True`,
  silently overriding an explicit `False`, and **no** tokenizer call received
  `revision` — while spec §4 constraint 4 pins MoLFormer to
  `7b12d946c181a37f6012b9dc3b002275de070314`. Weights pinned, tokenizer
  resolving `main`: the pair can drift.
- **R23 — a non-`off` checkpointing mode must enable at least one layer or
  raise.** The enabled counter sat **outside** the
  `hasattr(module, "gradient_checkpointing")` guard, so on a backbone whose
  layers lack the per-layer flag — the remote-code ESMC path, precisely R15's
  risk — `all`/`active` returned a success count while enabling nothing; and
  `active` with `layers=None` enabled nothing and returned 0. Ruled: count only
  enabled layers, raise when a non-`off` mode enables zero, and document the
  contract: only per-layer attributes are switched, so **a positive return is not
  proof** on an unfamiliar architecture.
- **R24 — the encoders must keep `esm_model=`/`molt5_model=` as deprecated
  aliases.** `main_train_kcat_expert.py:112-113` and
  `main_infer_ensemble.py:48-49` construct the encoders with those keywords, and
  the spec's §12 forbids editing those files — so Task 8's rename to
  `backbone_name=` would have broken both entry points with
  `TypeError: unexpected keyword argument` at import. Supplying both spellings
  raises a named `ValueError`.
- **R25 — `strip_special` is driven by the backbone kind.** The old
  `ProteinEncoder.encode_tokens` built its mask from `attention_mask` alone, so
  ESM-2's `<cls>`/`<eos>` positions **were included** in `masked_mean` pooling.
  Always passing `strip_special=True` would change kinetics pooling and with it
  every kinetics number, in direct conflict with the plan's own invariance
  constraint. Ruled: `strip_special=(spec.kind == KIND_ESMC)` — ESMC strips (its
  outputs keep two boundary tokens), ESM-2/T5 keep the old mask exactly. The
  condition is deliberately not widened, because `EsmTokenizer.all_special_ids`
  may include `<unk>` and stripping unknown-residue positions would discard real
  signal.
- **R27 — the tree is intentionally non-instantiable between Task 8 and Task 13:
  a sequencing note, not a defect.** Task 8's `_BackboneEncoder` requires `spec`
  whenever `backbone_obj` is injected, and `models/predictor.py:665-678` passes
  only `backbone_obj=`, so `KineticsPredictor(...)` raises until Task 13 lands.
  Import still works, and nothing in the suite constructed it — which is why the
  suite stayed green at 88. Consequence to keep straight: **no kinetics
  regression may be run or believed before Task 13 is complete.**
- **R28 — the changed gradient gating in `encode_tokens` is accepted, in the
  narrowed form `enable_grad = torch.is_grad_enabled() and any(p.requires_grad
  …)`.** *Reason revised after the controller's own premise was refuted:* the
  claim that the kcat expert's protein LoRA "silently never trained" was **false**
  — `main_train_kcat_expert.py:128` sets `train_backbone=True` after injecting
  LoRA, so the old gate was already `True` there. The real existing-path delta is
  `main_infer_ensemble.py:56-62`, which injects LoRA and never sets that flag: the
  bare `any(requires_grad)` would have re-enabled grad inside the caller's
  `@torch.no_grad()` and made the 650M backbone build an activation graph during
  ensemble routing (peak memory and latency, numerically identical). The `and`
  makes the encoder only ever *narrow* the ambient gradient context.
- **R35 — the brief's Step 4 snippet contradicts the brief's own test; the
  divergence is approved.** The snippet assigns `self.s_kcat = … = None` under
  non-kinetics tasks, but the brief's
  `test_affinity_model_does_not_construct_kinetics_parameters` asserts
  `not hasattr(...)` — and assigning `None` still creates the attribute. Spec §5
  D4 says 不构造 (do not construct). Note the binding reason is the test and the
  spec's wording, **not** the DDP mechanism the stated rationale cited: a bare
  attribute is not a parameter.
- **R36 — R2 was missing the checkpointing mode.** R2 lists the constructor's new
  options but no `mode` for `set_backbone_gradient_checkpointing`, while Task 14
  defines `--grad_checkpoint_backbone {off,active,all}` — so the CLI flag would
  have been dead, and a 6B run that asked for `active` would have exhausted memory
  with no diagnostic. Ruled: the construction call passes
  `grad_checkpoint_backbone=args.grad_checkpoint_backbone`, and both Task 14 and
  Task 21 must know checkpointing applies to the **protein** backbone only, and
  only when `use_lora=True`.
- **R39 — two silent-misconfiguration guards in the constructor.** (a)
  `use_lora=False` with a non-`off` `grad_checkpoint_backbone` is silently
  ignored today, so a user asking for `active` without LoRA gets no checkpointing
  and no diagnostic and discovers it as an OOM hours later; (b) supplying only
  one of `protein_encoder`/`ligand_encoder` silently discards it and falls
  through to real backbone loading. Both now raise a named `ValueError`; no
  legitimate configuration is rejected, because checkpointing genuinely cannot be
  honoured without LoRA.
- **R40 — `PredictorConfig` must reject an inconsistent task/head-spec pair** —
  and the literal wording could not catch its own example. Building
  `PredictorConfig(task="affinity")` directly keeps the kinetics `head_specs`, and
  `forward` then dies with a bare `KeyError: 'affinity'`. But a mutual-consistency
  check passes there, because `active_heads` and `head_specs` default to the
  kinetics trio **together**. Ruled: `task == "affinity"` must imply an
  `"affinity"` head exists. *(Corrected by R42; completed by R43.)*
- **R42 — R40's general rule was too broad and broke a supported inference path;
  narrowed to the affinity clause.** `main_infer_predictor.py:90-102` builds
  `head_specs` **only from the heads present in the checkpoint**, and its
  `if head_specs:` guard shows subsets are anticipated — so a checkpoint
  containing only `kcat` yields `head_specs = {"kcat": …}` against the default
  trio, and the general rule made that construction raise where it previously
  worked. `MultiHead` deliberately tolerates a missing spec
  (`cfg.head_specs.get(name, {})`). The affinity-only clause is what R40 was
  created for and cannot fire on a partial-head kinetics checkpoint.
- **R43 — the narrowed affinity guard has a hole; close it.** R42's literal
  wording asked for a check on `head_specs` only, but `forward` reads
  `outs_dict["affinity"]`, which `MultiHead` builds from **`active_heads`** — so
  `PredictorConfig(task="affinity", head_specs={"affinity": …})` with a defaulted
  `active_heads` passed construction and died later with a bare `KeyError`. The
  clause must test both fields, i.e. `task == "affinity" and ("affinity" not in
  head_specs or "affinity" not in active_heads)`. No production path was affected
  at the time — every real site uses `for_task`, an explicit config, the
  inference script, or the default.

## 4. Data, labels, loss and metrics

- **R3 — Task 14's `_make_affinity_dataset` drops the unused `source_ids`
  parameter.** An unused parameter is a defect the review rubric flags, and the
  affinity path has no source-id consumer (spec §5 D4 forbids domain
  uncertainty).
- **R8 — Task 2 must remove the conditional from Task 1's value assertion.**
  Task 1 shipped an `if info["hidden_size"] is None: … else: assert == 768/2`
  bridge because `models/plm_backbones.py` did not exist yet. Once Task 2 lands
  that is dead-branch test code that cannot fail on the value, silently weakening
  the integration check the brief intended; the failure path is separately
  testable with a config the resolver genuinely cannot satisfy.
- **R16 — Task 14 needs no affinity-specific column normalisation.**
  `normalize_training_dataframe` is already single-task tolerant — it back-fills
  `kcat(s^-1)`/`Km(M)` as all-NaN columns and raises only when
  `Sequence`/`Smiles` are missing — and `_canonical_pair_keys` depends only on
  those two, so `--split_mode group_pair` works on the affinity table as-is. Two
  normalisation paths would only add divergence risk.
- **R29 — the brief's `test_transform_preserves_nan_for_invalid_rows` is vacuous;
  the fix is an `inf` input.** Proven by mutation: replacing the masked write
  `out[finite] = …` with an unmasked `out[:] = …` still passes, because NaN
  propagates through either form. The two forms differ **only** for non-finite
  values that are not NaN — `±inf` — which is precisely what `transform`'s finite
  mask exists to handle.
- **R30 — two dataset parameters need a docstring contract.** `return_source=True`
  returns a 4-tuple that must not be fed to the training loaders, and `supervised`
  is stored but never consulted (the affinity path always carries a label).
  Documented rather than changed. *(Wording superseded by R32.)*
- **R32 — R30's `unpack_batch` slot mapping was stated backwards.** `utils.py` is
  `seqs, smis, a, y = batch`, so with `return_source=True`'s
  `(seq, smi, y, source)` the **label tensor** is bound as `row_ids` and the
  affinity-type **string** as `labels` — both slots wrong, which is worse than
  R30 described. The actionable guidance (offline analysis only) was right; the
  stated mechanism would have misled anyone debugging it.
- **R33 — the brief's `_barron_loss` is mathematically wrong, and at its own
  default it is not a robust loss at all.** The reference is the canonical
  CVPR-2019 form `(β/α)·((r²/β + 1)^(α/2) − 1)` with `β = |α−2|`; the brief used
  `f·((1 + r²/f)^f − 1)` with `f = |α−2|/α`, differing in **both** the inner
  divisor and the exponent. At the default `barron_alpha=1.0` the brief's form
  collapses to plain squared error, so `loss_type="barron"` would have silently
  been L2 and delivered none of the robustness the option exists for. Two extra
  pieces of evidence: the pre-fix form produced **negative losses** at `α = −1`,
  and the `alpha=1` test **alone** cannot catch a `β/α ⇄ α/β` prefactor swap
  (they are equal at `α = 1`), which is why a second non-special α is required.
- **R34 — a second inherited wording error in `models/affinity_loss.py`.** Line 17
  said `alpha=0 近似 L1`, but the α=0 branch is `log1p(0.5·r²)`, which grows
  **logarithmically** — the fix's own test states this correctly, so the module
  contradicted itself in-repo. The implementation was right; only the comment was
  wrong, and a reader choosing α for a desired robustness profile would have
  picked the wrong value.
- **R37 — `PredictionRecorder` needs an explicit label contract for the affinity
  path.** `utils.py:148-150` allocates **both** `preds` and `labels` at
  `(n_samples, output_dim)` while the affinity label is `(B,)`, so the label's
  handling was unspecified — and left implicit it is a silent mis-recording
  rather than an exception. Ruled: pass `labels` as an explicit `(B, 2)` array
  whose two columns are both the label, and read column 0 for the target and
  column 1 of `preds` for the predictive variance. **The action stands; the
  premise was corrected by evidence:** the implementer showed that numpy actually
  **raises** on `(4,)` → `(4, 2)`
  (`could not broadcast input array from shape (4,) into shape (4,2)`), and the
  shapes that are silently accepted with both columns identical are `(4, 1)` and a
  scalar. The hazard is the `(B, 1)` shape, not the `(B,)` one; the committed test
  targets the real silent case.
- **R38 — the construction path R27 restored has zero committed coverage and is
  locally testable.** Task 13's test file was mandated byte-for-byte from the
  brief, so the repo's only proof that `KineticsPredictor` constructs at all lived
  in a deleted scratch script — a future edit could silently re-break
  instantiation. Unlike the genuinely cluster-only gaps this one *can* be tested
  here: patch the two loaders with config-built backbones and call the real
  constructor. Landed as `tests/test_predictor_construction.py`.

## 5. Training loop: accumulation, schedule, EMA

- **R31 — the accumulation boundary belongs in `train_epoch`, not `train_step`.**
  Both `t` and `steps` inside `train_step` are **microbatch** quantities
  (`B = labels.size(0)`, `mb = microbatch_size or B`,
  `steps = (B + mb - 1) // mb`), so accumulating on them would measure the window
  in microbatches rather than in optimizer steps; and `train_step` called
  `optimizer.zero_grad(set_to_none=True)` unconditionally, so a naive per-call
  gate would wipe the gradients accumulated by the previous batches. Ruled:
  `train_step` gains `do_step`, `train_epoch` decides via `should_step(...)` and
  wraps the window in `model.no_sync()`. *(Completed by R47 and R51.)*
- **R41 — the accumulation scaling must be uniform across tasks.** The brief kept
  `loss = loss_mb / max(iters_to_accumulate, 1)` **inside the affinity branch
  only**. At the default `1` it is a no-op, but at `ITERS_TO_ACCUMULATE=2` the two
  tasks would have different effective learning rates for the same flag value —
  an asymmetry that looks like a tuning problem rather than a bug. Closed with one
  uniform `loss = loss_mb / (steps * accum)` at the backward point, with the
  factors multiplying: `/steps` the microbatch normalisation, `/accum` the
  accumulation normalisation.
- **R44 — the affinity NLL metric received a standard deviation where its formula
  expects a variance.** `main()` passed `exp(0.5·log_var)` (a std) into
  `_compute_affinity_metrics`, whose NLL expression treats its argument as a
  **variance** — so the logged affinity `nll` was not the quantity the model was
  being trained on, and a reader comparing the two would see a mismatch. Closed by
  keeping the value as a std (which `exp(0.5·log_var)` genuinely is, which the
  same array is saved as — `sigma`/`sigma_raw` in `test_predictions.npz` — and
  which the kinetics metric and `_temperature_scale_sigma` both assume) and
  changing the formula to the kinetics one, pinned by a hand-computed value. Had
  it kept the formula instead, every other consumer of that array would have been
  the inconsistent one.
- **R46 — the brief's `should_step` test contradicted itself; the implementer's
  correction stands.** The brief expected `flags[6] = False` for `(i, 3, 7)` while
  the very next test asserted `should_step(6, 3, 7) is True` — the same call
  cannot be both. The prose contract decides it ("the final partial window and the
  last batch of the epoch still step") and the arithmetic agrees: windows end at
  `i = 2` and `i = 5`, and `i = 6` is the epoch's last batch, so
  `[F, F, T, F, F, T, T]` is right. Taking the brief's literal list would have
  silently dropped the tail batch's gradient — the failure the test exists to
  prevent.
- **R47 — the schedule and the EMA still advanced per batch, so accumulation
  produced a wrong LR plan.** With `iters_to_accumulate > 1`, `global_step`, the
  LR scheduler step and its `total_steps`, `ema.update()` and the `_mse_ema`
  update all still advanced once per **batch**: at `accum = k` the LR plan
  completed roughly k× too early and the EMA was updated k× too often — on exactly
  the configuration accumulation exists for, which made the feature half-built
  rather than merely imperfect. Ruled: gate the per-step bookkeeping on the same
  decision that gates the optimizer step; at `accum = 1` this is provably a no-op,
  which the tests pin. The RED is exactly the predicted asymmetry:
  `optimizer.step=3` but `scheduler.step_and_clip=7`, `global_step=7`,
  `ema.update=7`, `_mse_ema_set=7`. *(Followed by R54 and R55.)*
- **R51 — sizing the LR plan in optimizer steps is approved** (a change beyond
  R47's literal four counters, and necessary rather than scope creep). Gating the
  counters alone creates the mirror image of the original bug: the scheduler would
  advance once per optimizer step while its plan length stayed
  `len(train_loader) * epochs`, so at `accum = k` OneCycle/cosine would complete
  only `1/k` of its schedule and the LR would **never anneal** — trading
  "finishes k× early" for "never finishes". `optimizer_steps_per_epoch(B, accum) =
  ceil(B / accum)` is exact given the flush semantics (`should_step` fires at
  `accum-1, 2·accum-1, …` and again at `B-1`), reduces literally to the old
  expression at `accum = 1`, and is bound to the real step count by a
  parametrised test rather than a copied constant.
- **R52 — the telemetry left per-batch is accepted.** The tqdm postfix and the
  per-batch `swanlab.log({"lr_step", "step"})` stay per-batch because they carry
  batch-level data (`avg_loss`) and keep the dashboard heartbeat alive inside long
  6B accumulation windows; the cost is k identical-valued step records — dashboard
  noise, not a correctness problem, and the heartbeat has real diagnostic value on
  a run whose individual windows are minutes long.
- **R53 — the cross-version-resume concern is vacuous; dismissed with evidence.**
  The implementer worried that old checkpoints stored `global_step` in batches
  while the new code interprets it as optimizer steps, under-setting
  `onecycle._step_count` on a resume. But accumulation is new in this task, so
  every pre-existing checkpoint was produced with `accum == 1`, where
  `should_step` fires on every batch: per-batch and per-optimizer-step counters are
  then identical by construction and `ceil(B/1) == B`. No migration is needed.
- **R54 — the R47 fix moved a manual `base_lr` pin inside the gate; hoist it.**
  `OneCycleLR`'s constructor sets every parameter group to `max_lr / div_factor`
  (i.e. `base_lr / 10` with the default `div`), so with the pin inside
  `if do_step:` the non-scheduled groups (tags `backbone`/`s`/`frozen`/`others`)
  trained at `base_lr/10` for batches `0..k-1` instead of only for batch 0. It
  self-corrects at the first window end and affects epoch 0 only under
  `accum > 1`, so it was graded Minor and no run was mistrained — but the line
  moves out of the gate.
- **R55 — the default-path equivalence is accepted, with its evidence limitation
  recorded.** The strongest clause of the implementer's claim — that all
  parameters come out bitwise equal at `accum = 1` — rests on a transcript from a
  scratch script that was deleted, not on a committed artifact; the in-tree test
  pins the counters but not the weights. Accepted because the analytic argument is
  stronger than the measurement it replaces: at `accum = 1`, `do_step` is
  unconditionally `True` and `ceil(B/1) == B`, so the statement order and every
  value are unchanged by construction, and the counter pin guards the regression
  that matters. A committed bitwise-weights test would require loading two
  versions of the module in one process, which is not worth its cost. Recorded so
  the final review can see what is pinned by a test and what is not.

## 6. Inference

- **R48 — Task 17's clamp test is vacuous; the hazard is overflow, not a negative
  variance.** The test passed `log_var=50.0`, and `math.exp(50.0)` is a perfectly
  ordinary float64 — so it would pass with the clamp deleted and proved nothing.
  What the clamp guards is **overflow**: `math.exp(1000.0)` raises
  `OverflowError`. Ruled: test with a value that overflows unclamped, and rename
  the test, since `exp` is never negative and "negative variance" describes a
  failure that cannot occur.
- **R59 — the legacy kinetics truth block can mislabel columns under
  `--task affinity`.** The block is inert on the builder's CSV, which carries no
  `kcat`/`km`/`value`/`log10_value` columns — but a CSV that did carry `value` or
  `log10_value` would be written out as `true_Km*` fields in the predictions CSV,
  a **mislabelled output** a downstream reader could not detect. Ruled: gate the
  block on the task. Kinetics is unchanged because the condition is true there
  (the guard defaults to the kinetics case when `args` has no `task`), and the
  guard removes a latent silent-wrong-answer for any future table with
  differently-named label columns.

## 7. Kubernetes delivery: launcher and manifest

- **R45 — Task 19's manifest must mirror the reference pod spec and be a bare
  PyTorchJob.** Two differences from
  `xtrimofold3-affinity-plm/yamls/kubeflow/xf3_pl_multitask_k8s.yaml` mattered:
  the reference uses a YAML **anchor** so `Worker.template` aliases the Master's
  pod spec (the plan wrote the spec out twice, inviting the two replicas to drift
  apart), and the reference is an Argo `Workflow` that creates the PyTorchJob as a
  `resource` with the manifest embedded as a string and parameterised through
  `{{inputs.parameters.*}}` — indirection only useful on an Argo-driven cluster,
  whereas a bare `kubeflow.org/v1` PyTorchJob is directly `kubectl apply`-able.
  Everything that actually matters must match the reference: `MASTER_ADDR`
  required via `: "${MASTER_ADDR:?…}"`, `NNODES = worker_num + 1`, `NODE_RANK`
  from `${RANK}`, the NCCL block (`eth0`/`0`/`5`/`AF_INET`), the a100/roce2
  nodeSelector, the `NBTopic`/`006_mix` tolerations, privileged + `IPC_LOCK`, the
  memory-backed `/dev/shm`, and rank-0-only logging.
- **R49 — the k8s path profile would have created a directory literally named
  `${USER}`.** `path_config` does no shell-style expansion: it reads env vars
  through `ENV_OVERRIDES` and hands values straight to `Path(v)`, so every k8s run
  would have written under a literal `${USER}` path component — and the brief's
  own tests would still have passed, because they only asserted `is not None` and
  `isinstance(Path)`. Fixed by expanding the placeholder at resolve time from the
  environment (with a fallback), plus a test asserting no resolved value contains
  `$`. *(Its fallback is made safe by R61.)*
- **R56 — four of the 18+19 Minors were ruled into the fix loop because they fail
  *silently*** (the default is that minor findings do not enter the loop). (1) The
  container's PID 1 is the outer `/bin/bash -lc` and `run_trainer` called the
  launcher as a **child**, so the launcher's `exec` made torchrun a child of PID 1
  rather than PID 1: the outer bash neither forwards SIGTERM nor dies on it, so on
  preemption torchrun is SIGKILLed at grace-period expiry and its own SIGTERM
  handling never runs — for a long 6B run, the difference between a clean
  checkpoint and a lost window. Fixed by `exec bash bin/train/kineager_k8s.sh`.
  (2) A fixed `metadata.name` plus `cleanPodPolicy: None` and no TTL made a second
  `kubectl apply` of an unchanged manifest a **silent no-op** that reports
  success; fixed with `ttlSecondsAfterFinished`. (3) A comment invited adding
  `NCCL_*` overrides to `env:` which the unconditional assignments below it would
  silently override; fixed by making those assignments `${VAR:-…}` defaults so the
  comment becomes true. (4) `MASTER_PORT` was validated only as a non-negative
  integer, so `0` and `70000` passed and failed later inside torchrun with a far
  less actionable message; a `1..65535` range check was added. On the sixth item
  (the caller-argument passthrough) the implementer chose `${1+"$@"}` over both
  forms the ruling offered: `"${@:-}"` expands to **one empty word** with no
  positional parameters, which argparse rejects as `unrecognized arguments:` —
  i.e. the suggested fix would have converted a rare bash<4.4 failure into a
  guaranteed one on every bare invocation.
- **R57 — the k8s profile pointed `train_data_dir` at another user's dataset
  directory.** Verified in the source: `main_train_predictor_multigpu.py:1873`
  creates `train_data_dir` and `:1874-1875` `np.save` the per-epoch
  `preds_epoch*.npy` / `labels_epoch*.npy` there, while the profile pointed it at
  `/pfs_beijing/xumeng/general_protein_dir/Affinity/dataset/PDBbind_v2020`. Two
  failure modes, and the quieter one is worse: epoch 0 dies with
  `PermissionError`, or it succeeds and training artifacts are written into a
  shared dataset directory that other people's jobs read. Ruled: point it at the
  user's own area (`/pfs_beijing/${USER}/kineager_runs`, where `runs_dir` and
  `weights_dir` already resolve), keep `CATA_TRAIN_DATA_DIR` overridable, and
  assert that no `xumeng` component survives. *(Opens the door R61 closes.)*
- **R58 — nothing activated the k8s profile, so a cluster run silently used
  `autodl`.** `CATA_` occurred **zero** times in the launcher and zero times in the
  manifest, while `resolve_runtime_paths` falls back to the platform default
  (`autodl` on Linux): the run would have read `train_csv_path`, `train_data_dir`
  and both backbone paths from the wrong profile — either an immediate crash or,
  worse, a run against stale data, with nothing in the logs saying which profile
  was chosen. Ruled, both halves: the launcher exports
  `CATA_PROFILE="${CATA_PROFILE:-k8s}"` before the `exec` (it is the k8s
  entrypoint by name, and `${VAR:-…}` keeps it overridable), **and** the manifest
  lists `CATA_PROFILE` in its `env:` block with a header mention, so an operator
  can see and change it without reading the script.

## 8. Dependencies and artifact hygiene

- **R50 — `requirements.txt` must floor the `esm` pin.** The ESM-C tokenizer port
  is version-bounded: the ESMC checkpoint's own remote loader raises
  `pip install 'esm>=3.4.1'`. An unpinned install can resolve to a version that
  lacks `EsmcTokenizer` and fail only at model-load time on the cluster — after a
  multi-gigabyte download, which is exactly the failure mode R9 and R13 exist to
  avoid. Pinned as `esm>=3.4.1` with the reason in a comment.
- **R60 — PyYAML is a real test dependency and belongs in `requirements.txt`.**
  `tests/test_k8s_artifacts.py` imports `yaml`. PyYAML happened to be installed on
  the development host but was declared nowhere, so a fresh
  `pip install -r requirements.txt` on the cluster followed by `pytest` would
  error at **import** on the module that verifies the k8s artifacts — an error
  about the environment rather than about anything real.
- **R61 — the R57 fix made a *write* target depend on a shared fallback; close
  it.** With neither `USER` nor `USERNAME` set — which is the case in a pod, since
  the manifest set neither — the corrected write path degrades to the **shared**
  `/pfs_beijing/kineager/kineager_runs`, where two concurrent users'
  `preds_epoch*/labels_epoch*.npy` overwrite each other. R49's fallback was ruled
  for robustness, but R49 also predates this path being a write target, and the
  R57 diff is what couples the two. Ruled, both halves: the manifest's `env:`
  block gains an operator-filled `USER` entry alongside the other placeholders,
  and the launcher fails fast when the k8s profile is in use and `USER` is unset.
  Failing loudly is the right trade for the same reason it was right in R23, R33
  and R40: a shared directory produces runs that look correct while their
  artifacts overwrite each other, which is far harder to notice than a refusal to
  start.
- **R62 — keep the hard `yaml` import in `tests/test_k8s_artifacts.py`; do not
  switch to `importorskip`.** *(Recorded after the R1–R61 range this appendix was
  commissioned for; included because it is the decision most likely to be
  re-litigated.)* A reviewer suggested `importorskip` to avoid a collection error
  in a minimal environment. But R60 already makes PyYAML a declared dependency,
  and `importorskip` would **silently skip the module that verifies the k8s
  artifacts** — hiding exactly the class of problem that module exists to catch,
  which is the opposite of what this project enforced all session. A collection
  error that says "install your dependencies" is the correct failure.

### Whole-branch review round (R69–R70)

Recorded after the R1–R68 body above: the whole-branch review returned "Ready with
caveats" with two **CRITICAL** findings, both gaps the per-task reviews could not
see because each task was reviewed against its own brief.

- **R69 — the affinity inference path could not run at all.** `main_infer_predictor.py`
  built `config = PredictorConfig(**cfg_updates)`, and `_infer_predictor_cfg_from_ckpt`
  never set `task`, so it defaulted to `"kinetics"` and `MultiHead` built the
  kcat/Km/ratio heads — while `--task affinity` was only consulted *inside the
  inference loop*, never at construction. Reproduced with a real affinity
  `state_dict`: the model built as `task='kinetics'` with
  `active_heads=['kcat','Km','ratio']`, reported `Missing keys: 21 | Unexpected
  keys: 6` (the six unexpected being the entire trained
  `multi_head.heads.affinity.*`), returned a **7-tuple**, and `main()` then died
  with `ValueError: too many values to unpack (expected 2)`. The whole Task 17
  path was unreachable in practice, and a "does the checkpoint load" check could
  not catch it because the load succeeds either way — the affinity head is simply
  ignored.
  **Why the tests missed it:** `tests/test_predictor_import.py` monkeypatched
  `build_model_from_ckpt` to a lambda, so no test ever loaded a checkpoint through
  the real builder. Ruled: infer the task **from the checkpoint**
  (`multi_head.heads.affinity.*`, which is also the right signal when the script is
  called without `--task`), build through `PredictorConfig.for_task("affinity", …)`
  in that case, refine the affinity head's `hidden_dim` from its weights the same
  way the kinetics heads already are, leave the kinetics construction path exactly
  as it is, and add a test that uses a real `state_dict` through the real builder —
  no lambda. Also added: `--task affinity` with a kinetics checkpoint is now a
  named error rather than a mis-typed run.
- **R70 — the kinetics default LoRA layer policy silently changed.** Task 14 made
  the trainer always pass `--lora_layer_policy` (argparse default `"last_k"` with
  `lora_last_k=30`), which **overrode** `KineticsPredictor.__init__`'s own default
  of `"all"` (`models/predictor.py:519`, verified unchanged from base). Same
  command, different trainable set: the historical recipe used to attach LoRA on
  ESM-2 layers 0..32 and now attached it on 3..32 — 18 fewer target modules out of
  198, about **−9%** of the protein-side LoRA — so kinetics numbers would differ
  from an identical invocation. This was the one place the branch failed the
  standing requirement that `task=kinetics` stays behaviourally unchanged; no
  existing ruling covered it (R18 is about the suffix whitelist, and the design
  mandates last-K for the **affinity** path only). Ruled: make the default
  task-conditional — argparse default `None`, resolved in `main()` as
  `lora_layer_policy or ("last_k" if task == "affinity" else "all")` — so kinetics
  keeps `"all"` exactly as before, affinity still defaults to last-30, and an
  explicit flag always wins. Landed as the pure helper
  `resolve_lora_layer_policy(explicit, task)`, pinned by tests that must fail
  against the old default.

**Decided in the same round (judgement calls, not rulings), for the record:**

* **The inference clamp is now ±10, matching the training loss.**
  `format_affinity_prediction` clamped `log_var` to ±20 while
  `models/affinity_loss.py` clamps to ±10. The loss is the training-time contract
  the model was optimised under, and ±20 would emit `exp(20) ≈ 4.85e8` as a
  "variance", which no calibration supports. The inference module now carries a
  named constant that a test ties to the loss's `_LOG_VAR_MIN`/`_LOG_VAR_MAX`, so
  the two can no longer drift apart silently.
* **`AffinityDataset` drops rows with a missing `Sequence`/`Smiles`, loudly.**
  `__getitem__` does `str(row[col])`, so a NaN cell became the literal string
  `"nan"` and was encoded as if it were a real sequence or molecule — no error,
  just a poisoned sample. The dataset already filtered rows by a finite label, so
  the same predicate now also requires the two input columns, and the dropped
  count is printed. Dropping rather than raising keeps a mostly-good table
  runnable while making the loss visible; `len(dataset)` and `self.df` stay in
  step, and `PredictionRecorder` is sized from `batch_size × n_batches` rather
  than the dataset length, so a shorter table cannot overrun it.

### Review rounds (R63–R68)

Recorded after the R1–R62 body above. R63–R65 amend R61's guard, which the task
review found to be too weak in three independent ways; R66 concerns the audit's
stated range; R67 and R68 are the documentation round that followed, and both are
about statements rather than behaviour.

- **R63 — the guard must accept exactly the forms the resolver accepts.**
  `path_config._current_user_name()` resolves `USER → USERNAME → the literal`, but
  the guard tested `USER` alone, so a host with only `USERNAME` set was refused
  even though resolution would have produced a valid per-user directory — the
  launcher was **stricter than the resolver**, and `README.md` still documented
  the `USER`-else-`USERNAME` order. Ruled: accept `${USER:-${USERNAME:-}}` and
  export whichever supplied the value (R64), or at minimum name `USERNAME` in the
  refusal message. Implemented as the former, with the refusal naming both.
- **R64 — the guard could be defused by a shell variable that never reached the
  environment.** The guard read a **shell variable** — `${USER:-}` sees one whether
  or not it is exported — while the `exec`'d `torch.distributed.run` inherits only
  the **environment** and `path_config._current_user_name()` reads
  `os.getenv("USER")`. Any route that makes those two disagree defeats the check,
  and on a host where `USER` is not already in the environment (a minimal
  container, which is exactly the k8s case) it does so silently:
  * a **sourced** invocation — `USER=alice; source bin/train/kineager_k8s.sh` — runs
    the guard in the caller's own shell, where the unexported variable is visible,
    so the guard passed; `exec python3 -m torch.distributed.run` then handed the
    trainer an environment without `USER`, and the run resolved to the shared
    `/pfs_beijing/kineager/kineager_runs`;
  * a `BASH_ENV`/startup file that sets `USER` without exporting it has the same
    effect for a normal `bash bin/train/kineager_k8s.sh` invocation.

  (A bare `USER=alice; bash bin/train/kineager_k8s.sh`, by contrast, is **not** a
  defeating form: the child bash inherits only the environment, so the pre-fix
  guard saw nothing and refused. An earlier draft of this entry offered it as the
  repro; the mechanism above is the accurate one — R68.)

  The guard therefore *certified* a run that was about to write to a shared
  directory, which made it worse than useless. Ruled: `export` the validated value
  so the value that was checked is the value the trainer receives, pinned by an
  assertion.
- **R65 — an unfilled `USER` placeholder passed the guard.** The manifest's value
  is `"<REPLACE_ME: your username on the shared filesystem>"`, which is
  non-empty, so an operator who forgot to replace it passed the guard and the run
  created `/pfs_beijing/<REPLACE_ME: your username on the shared filesystem>/kineager_runs`.
  The manifest's "an unedited apply fails loudly" property does not extend to this
  value, because any string is a valid Kubernetes env value. Ruled: also reject a
  `USER` matching `^<REPLACE_ME`, with a message that says the placeholder is
  still in place.
- **R66 — the audit's range label was challenged; re-measurement upheld it, and
  both views are now stated.** The review read the `8607086..2c54e9d` stat
  (README `6/3` inside `1144 insertions(+), 3 deletions(-)`) as proof that three
  README deletions fall inside the audit's branch range, which would make the
  deletion set span five files and contradict the whole-branch totals. Measured at
  the round-1 commit, with the range named so the number cannot go stale:

  | range | files | insertions | deletions | README.md |
  |---|---|---|---|---|
  | `3497a5e..161c1ea` (branch vs base) | 28 | **8137** | **856** | 68 / 0 |
  | `3497a5e..161c1ea`, whitespace-ignored (`-w`) | 28 | 7558 | 277 | — |
  | `8607086..161c1ea` (this task's commits) | 7 | 1423 | 3 | 6 / 3 |

  All 856 deletions sit in exactly four files (`main_infer_predictor` 63,
  `main_train_predictor_multigpu` 519, `models/encoders` 100,
  `models/predictor` 174 — unchanged since the round-0 measurement; only the
  insertion count moves as documentation is added). The three lines the R61 commit
  changed were themselves part of Task 20's added README block, so they are
  deletions only relative to `8607086`, not to the branch base — against
  `3497a5e` the README delta stays purely additive (65/0 at `8607086`, 68/0 now),
  and those three lines are documentation in the affinity README section (no
  kinetics relevance). An earlier draft of this entry quoted **+7858/−856** and
  labelled it "the final HEAD": 7858 was correct at `2c54e9d` and stopped being
  correct the moment round 1 committed, which is the same label-vs-number defect
  this ruling is about (R67). The report prints both ranges side by side for the
  same reason.
- **R67 — the appendix repeated the very label-vs-number defect R66 was about.**
  The R66 entry said "Re-measured at the final HEAD … **+7858/−856**": 7858 was the
  round-0 total at `2c54e9d`, and round 1 (which nets +279) had already made it
  wrong, so the appendix and the report stated different numbers for the same
  measurement while both called it final. Ruled: put the current numbers in, with
  the **range named explicitly** (`3497a5e..161c1ea`) rather than a moving label
  such as "final HEAD", so the number and its label stay consistent as later
  commits land. The deletion set (856, four files), `README.md` 68/0 and every
  kinetics conclusion are unaffected — and the fix doubles as the demonstration:
  only insertions move when documentation is added, which is exactly why a range
  label must name its range.
- **R68 — R64's stated reproduction did not reproduce as written.** The launcher
  comment and the R64 entry both offered `USER=alice; bash bin/train/kineager_k8s.sh …`
  as the form that defused the pre-fix guard. It is not: the child bash inherits
  only the environment, so the pre-fix guard saw no `USER` and **refused** that
  literal form. The hazard R64 describes is real and the `export` is the right fix;
  what was wrong is the repro. The unexported variable bites when the guard's shell
  inherits it by a route other than the environment — a **sourced** invocation
  (`USER=alice; source bin/train/kineager_k8s.sh`, where the guard runs in the
  caller's own shell and the `exec`'d trainer does not), or a `BASH_ENV`/startup
  file that sets `USER` without exporting it. Ruled: restate the mechanism
  accurately in both places — the guard read a shell variable while the trainer
  reads the environment, so any route that makes the two disagree defeats the
  check — and keep the export and its tests unchanged.

---

## 9. Supersessions, corrections and dismissed concerns

Read this before acting on any entry above that a later one amended. The table
below contains only genuine **earlier → later** links; rulings that were revised
in place are listed separately, because a row whose two columns name the same
ruling is not a link.

| Earlier | Later | What changed |
|---|---|---|
| R2 | **R36** | R2's constructor list was missing the checkpointing `mode`; without R36 the `--grad_checkpoint_backbone` flag was dead. |
| R30 | **R32** | R30's `unpack_batch` slot mapping was backwards; the real binding puts the label in the row-id slot and the affinity-type string in the labels slot. |
| R31 | **R47**, **R51** | R31 placed the boundary in `train_epoch`; R47 gated the schedule/EMA counters on it, and R51 sized the LR plan in optimizer steps so the schedule can actually finish. |
| R40 | **R42**, **R43** | R42 dropped R40's general rule because it broke partial-head inference checkpoints; R43 extended what remained to test `active_heads` as well as `head_specs`. |
| R47 | **R54**, **R55** | R54 hoisted a `base_lr` pin that R47's gate had moved; R55 recorded that R47's bitwise default-path equivalence rests on a deleted transcript, not a committed artifact. |
| R49 | **R61** | R49's user fallback became unsafe once R57 made `train_data_dir` a write target; R61 requires `USER` rather than allowing the shared fallback. |
| R61 | **R63**, **R64**, **R65** | R61's guard was too weak in three ways, all found by review: it tested `USER` only while the resolver also accepts `USERNAME` (R63); a shell variable that was never `export`ed satisfied it yet never reached the trainer's environment (R64); and the manifest's non-empty `<REPLACE_ME: …>` placeholder passed it (R65). |

### Revised in place (not supersessions)

These rulings were corrected or refined within their own entry; there is no
"later ruling" to follow, so read the entry as it now stands.

* **R28** — the controller's premise (that the kcat expert's protein LoRA never
  trained) was refuted by reading the source (`train_backbone=True` at
  `main_train_kcat_expert.py:128`); the ruling's *action* narrowed to
  `torch.is_grad_enabled() and any(p.requires_grad …)`.
* **R37** — the action (explicit `(B, 2)` labels) stands; the rationale was
  wrong. numpy **raises** on `(B,)` → `(B, 2)`; the silently-accepted shapes are
  `(B, 1)` and a scalar.
* **R46** — does not correct the implementation: it corrects the *brief's own
  test*, which contradicted the brief's next test.
* **R53** — dismisses the cross-version-resume concern rather than amending a
  ruling: accumulation is new, so every pre-existing checkpoint ran at
  `accum == 1`, where the counters coincide by construction.
* **R56 (item 6)** — the implementer's `${1+"$@"}` is better than either form the
  ruling offered, because `"${@:-}"` expands to one empty word that argparse
  rejects.

### The controller errors this session, in one place

Five factual premises were refuted by subagents with primary evidence, verified by
the controller, and corrected in place rather than quietly applied: R28's
kcat-expert claim, the vacuous `no_grad` test formulation specified for the R28
fix, R30's `unpack_batch` slot mapping, R40's validation scope, and R37's numpy
broadcasting premise. A sixth (R56 item 6) was improved rather than merely fixed.
They are listed because the ruling history is only trustworthy if its own
corrections are visible.
