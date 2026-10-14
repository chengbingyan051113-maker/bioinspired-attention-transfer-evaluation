# External ROC analysis

The input is a local collection of already aggregated unit predictions:

    ROOT/external/COHORT/seedSEED/MODEL/unit_predictions.csv

Cohorts are `deepdrid_CFP`, `deepdrid_UWF`, `messidor2_CFP`, and `eyepacs_CFP`.
Seeds are 2026, 2027, and 2028. Models are `M0R`, `M1_attention`, `N2_local_contrast`, and `N3_adaptive_s4`.
All 48 files are required. Obtain predictions through authorized local inference; they are not distributed here.

Each CSV needs `unit_id`, `primary_eligible`, `reference_grade`, and `referable_score`.
The latter is a score in [0,1]. Grades are 0 to 4. Eligibility is 0/1 or false/true.
If present, grouping fields `unit_type`, `image_count`, `gradable_images`, `aggregation`, and `grouping_provenance` must match across models and seeds.
CSV membership is paired by unit ID; duplicates and mismatched labels are rejected.
The tool does not construct patient groups or establish independence.

## Computation

- Referable disease means grade at least 2; grade-2 sensitivity uses exactly grade 2 as its denominator.
- A unit is referred when its score is at least the cutoff.
- The cutoff is the next representable value above the negative-score boundary allowed by floor(0.10 times the negative count), with a numerical guard. Ties cannot be split.
- Raw pAUC is the piecewise-linear empirical ROC area over FPR 0 to 0.10, with boundary interpolation.
- McClish-standardized pAUC is 0.5 times [1 + (raw area minus alpha squared/2)/(alpha minus alpha squared/2)], alpha=0.10. The diagonal is 0.5; perfect discrimination is 1.
- Bootstrap resamples whole units within maximum-grade strata using common multiplicities across all models and fixed optimization seeds. Every cutoff is reestimated. Reported effects average the three within-seed differences.
- Intervals are pointwise unadjusted percentile 95%. Seeds are not resampled, and probabilities are not ensembled.

Use `--replicates 0` for point estimates only; use at least 2000 for intervals. Result directories must not already exist.
All sensitivity and specificity results use proportions; multiply sensitivity effects by 100 to express percentage points. Areas remain in area units.

## Reproduction scope

This public implementation was checked against all 48 supplied fitted-model point estimates and 16 aggregate means, along with tied-score and integer-resampling fixtures. The full 2000-replicate historical interval output was not re-run in the 14 October code update. Multinomial multiplicity sampling has the same stratified bootstrap distribution as directly sampling indices but need not reproduce a historical finite Monte Carlo sequence byte for byte.

The external labels set the cutoffs, so this is a post hoc ROC comparison. It neither calibrates a deployment threshold independently nor guarantees future specificity. Grade counts and models stay fixed; selection, retraining and unresolved source/patient dependence remain outside these intervals. Original frozen-threshold results remain a separate analysis.

Reference: McClish DK. Analyzing a portion of the ROC curve. Medical Decision Making. 1989;9:190-195. https://doi.org/10.1177/0272989X8900900307
