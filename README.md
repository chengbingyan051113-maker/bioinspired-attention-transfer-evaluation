# Bioinspired attention transfer evaluation

The evaluation matrix and evidence-qualified principles are the main contribution.
This package includes N3, original N3 component controls, shared EfficientNet-B0
branches, unchanged loader/epoch/validation functions, conditional paired image
bootstrap, post hoc referral Grad-CAM and completed attention-budget controls.
The manuscript reports 96 training executions, including nine follow-up controls
and six same-seed baseline replays; replays add no independent seeds.

Clinical images, identifiers, individual predictions, private split manifests,
device identities and checkpoints are excluded. The public adapter is not the
locked server replay package. Exact historical replication also requires
authorized data, original manifests, runtime and audit records.
The runnable model configurations focus on the N3 study; earlier fixed-operator
and design-space comparisons are supplied as aggregate effects.

## Environment and checks

Historical runtime: Python 3.10.16, PyTorch 1.13.1+cu117, TorchVision
0.14.1+cu117, NumPy 1.24.3, pandas 2.0.3 and Pillow 10.4.0.
Install compatible PyTorch separately. Scripts never download data or weights.

    pip install -r requirements.txt
    python tests/check_package.py
    python src/budget_ledger.py --output outputs/budget

CPU checks cover syntax, exact function hashes and weighted-AUROC ties. GPU
execution is not tested locally. The private Stage31 package mandates full
server byte, stream, identity, gradient, parameter and MAC checks before training.

## Training adapter

Supply four explicit manifests using docs/input_schema.md, local ImageNet weights,
one variant and one seed. Existing output directories are refused.

    CUDA_VISIBLE_DEVICES=0 python train_public.py --variant N3_adaptive_s4 --seed 2026 --data-dir data --imagenet-weights data/efficientnet_b0_imagenet1k_v1.pth --output outputs/seed2026/N3_adaptive_s4

Validation selects checkpoints and thresholds. Its reported performance is
development performance. The adapter does not certify patient/source separation,
create missing private splits or resume an interrupted trial.

## Conditional paired analysis

    python src/paired_bootstrap.py --n3-runs LOCAL_N3_RUNS --component-root LOCAL_COMPONENT_ROOT --matched-root LOCAL_BUDGET_ROOT --output outputs/paired_bootstrap

LOCAL_BUDGET_ROOT is the completed Stage31 result directory containing runs/.
Optional --n2-root and --design-root include earlier archived comparisons.
Models, thresholds and seeds stay fixed; common image multiplicities are sampled
within five grade strata. Intervals do not correct selection optimism, retraining
uncertainty or unknown patient clustering. They are not clinical validation or
evidence of equivalence or failure of biological mechanisms.

The aggregate CSV contains 27 comparisons in both modalities and four scopes
(the fixed three-seed mean plus each fitted seed), totaling 864 effect/interval
rows. Twenty-three contrasts were independently replayed locally. Four original
component contrasts were computed on the server with the same checked script;
their additional raw predictions were not independently resampled locally.
The budget return supplied 36 prediction CSVs, including 18 existing references:
all 160 returned effect/interval rows were reproduced, and 64 duplicate baseline
rows were removed before the 96 new rows were appended. The aggregate provenance
covers 180 file entries (156 locally replayed and 24 server-only original files).
New checkpoint binaries were not returned; local image inference and GPU preflight
were not repeated. See docs/internal_analysis_scope.json and
docs/budget_results_summary.json for review scope and interpretation limits.

## Budget scope

New attention controls have 687 active parameters per branch. Analytical MACs
and scalar-FLOP estimates differ from N3 by less than 1% under the declared
convention, without inert parameters or detached dummy work. Recurrent topology
changes support, late pooling changes response statistics and magnitude/squared
magnitude is a compound unsigned basis. Matching does not establish equal
effective capacity, latency or physiological realism.

## External high-specificity ROC analysis

The new CPU tool evaluates referral sensitivity at each fitted model's lowest feasible cutoff with empirical specificity at least 90%, and integrates the referable ROC over FPR 0 to 0.10. Tied scores stay together. It reports both raw pAUC and McClish-standardized pAUC. Three-seed means average metrics; they do not ensemble probabilities.

    python src/external_roc.py --input LOCAL_UNIT_PREDICTIONS --output outputs/external_roc --replicates 2000
    python tests/check_external_roc.py

See docs/external_roc.md for input layout, estimands and interpretation limits. Aggregate point estimates from the existing 48 prediction files are in docs/external_roc_summary.csv. No clinical predictions are distributed.

Cutoffs use the same external labels as evaluation. This is an exploratory ROC comparison, not independent deployment calibration. The original frozen-threshold results and model advancement decisions are unchanged.

## Citation and license

Complete CITATION.cff.example with real authors and rename it CITATION.cff.
The authors must confirm the code license. The Zenodo example is a template,
not published metadata. The target repository is https://github.com/chengbingyan051113-maker/bioinspired-attention-transfer-evaluation.
The package does not assign an archival DOI.
Add a version-specific DOI only after an actual archive exists. Do not upload
private server bundles, images, weights or individual predictions to this repo.
