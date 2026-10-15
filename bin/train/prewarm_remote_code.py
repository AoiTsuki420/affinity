"""Prewarm the HuggingFace remote-code module cache (single process).

transformers copies `trust_remote_code` files into ~/.cache/huggingface/modules
non-atomically. When several ranks of the same container start together, one can
import a half-copied module (seen on 2026-10-15 with two local ranks:
"module transformers_modules...configuration_molformer has no attribute
'MolformerConfig'"). Importing the configs once, single-process, before torchrun
starts makes every later import hit the cache directly.

Used by the Kubeflow Pipelines route (one pod runs 8 ranks) and by standalone
multi-rank runs; k8s one-GPU-per-pod launches do not need it.
"""
from transformers import AutoConfig, AutoTokenizer

MOL = ("/pfs_beijing/xinran_2026/models/molformer/"
       "models--ibm-research--MoLFormer-XL-both-10pct/snapshots/"
       "7b12d946c181a37f6012b9dc3b002275de070314")
ESMC = "/pfs_beijing/xinran_2026/models/ESMC-6B2026.9.15"
REV = "7b12d946c181a37f6012b9dc3b002275de070314"

AutoConfig.from_pretrained(MOL, trust_remote_code=True, revision=REV)
AutoTokenizer.from_pretrained(MOL, trust_remote_code=True, revision=REV)
AutoConfig.from_pretrained(ESMC, trust_remote_code=True)
print("[prewarm] remote-code modules cached")
