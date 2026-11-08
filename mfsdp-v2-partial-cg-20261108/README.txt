Megatron FSDP v2 partial CUDA Graph validation evidence

Implementation: https://github.com/WEIXILIN0307/Megatron-LM/commit/cf473f207e4c09e3a108457969352b92b7efb86c
Target: NVIDIA/Megatron-LM dev at 3d9609fcc0a96e252323af359d61dc37d1549528.

Current PR-head results are recorded separately in publication/current-head-validation.json
and hardening/pr-dev-*; their source files and intervening upstream changes are hashed.
The feature diff is identical to the previously validated simplified patch series.

Versioned full-size results are under production/hardening-*-dropout* and
hardening/acceptance-analysis.json. They use the pre-simplification runtime on
base 76498e47585be45be9eb03d8ee050f7d010b33ca, whose recorded runtime files are
preserved under versioned-source/pre-simplification/. The training helpers,
public model/data provenance, commands, original logs, scalar events, full-state
comparison reports, RNG payload hashes and compressed profile are included.
The full-size training is complete-architecture random initialization, not
pretrained-weight conversion. Full-size training has not been rerun at the PR head.

Earlier long-run and timing references in the archived validation summary retain
their original source revisions and are not current-head performance claims.
Raw model/Adam checkpoints and dataset/model download files are not included.
No upstream CI or full-pretraining convergence claim is made.

SHA256SUMS.json records every file delivered in this evidence directory.
