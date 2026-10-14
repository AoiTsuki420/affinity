# KinEAGER -- PDBbind v2020 affinity image (feat/pdbbind-affinity branch).
#
# Base: xinran_2026/esmc20260915:v1 -- a snapshot of the esmc-precompute-new
# conda env that already carries the load-bearing pieces of this branch:
# Python 3.12, torch 2.11 (cu126), transformers 4.57.6, esm 3.4.1
# (EsmcTokenizer verified) and rdkit/scipy/sklearn/pandas/matplotlib.
# Only the packages the affinity code adds on top are installed here.
#
# The image carries the repo at the path path_config.py's `k8s` profile expects
# (`project_root = /workspace/kineager`), so a Kubeflow pod can run
# `bash bin/train/kineager_k8s.sh ...` with only `workingDir: /workspace/kineager`.
#
# Backbones (ESMC-6B, MoLFormer) are NOT baked in: the runbook requires local
# snapshot directories, mounted from the shared filesystem at run time.

FROM harbor.biomap-int.com/xinran_2026/esmc20260915:v1

# Working pip index for this build and any later in-container install
# (the esmc snapshot carries no pip configuration).
RUN printf '[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple\n' \
        > /etc/pip.conf

# Missing from the base, relative to this branch:
#   peft, sentencepiece, protobuf, wandb  -- in requirements.txt, not installed
#   pytest                                -- cluster smoke run `python -m pytest ... tests/`
#   biopython                             -- esm dependency the snapshot lacks
RUN pip install --no-cache-dir \
        peft \
        sentencepiece \
        protobuf \
        wandb \
        pytest \
        biopython

# /etc/profile resets PATH in login shells, and the Kubeflow manifest launches
# the container as `bash -lc ...` -- without this, that shell would drop the
# conda env and pick the system python (3.10, no torch) instead. profile.d runs
# after the reset, so re-assert the env (and CUDA) first here.
RUN printf 'export PATH="/opt/miniconda3/envs/esmc-precompute-new/bin:/usr/local/cuda/bin:$PATH"\n' \
        > /etc/profile.d/zz-kineager-path.sh

# Repo root == path_config.py k8s profile project_root == manifest workingDir.
COPY . /workspace/kineager
WORKDIR /workspace/kineager

CMD ["/bin/bash"]
