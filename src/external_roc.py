"""Exploratory external ROC analysis. No model inference or threshold deployment."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (2026, 2027, 2028)
COHORTS = ('deepdrid_CFP', 'deepdrid_UWF', 'messidor2_CFP', 'eyepacs_CFP')
MODELS = ('M0R', 'M1_attention', 'N2_local_contrast', 'N3_adaptive_s4')
METRICS = ('grade2_sensitivity', 'referable_sensitivity', 'specificity',
           'raw_pauc', 'standardized_pauc')


class FittedROC:
    def __init__(self, scores, grades):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.grades = np.asarray(grades, dtype=np.int64)
        if self.scores.ndim != 1 or self.grades.shape != self.scores.shape:
            raise ValueError('Scores and grades must be paired vectors')
        if not np.all(np.isfinite(self.scores)) or np.any((self.scores < 0) | (self.scores > 1)):
            raise ValueError('Scores must be finite probabilities')
        if np.any((self.grades < 0) | (self.grades > 4)):
            raise ValueError('Grades must lie in 0 to 4')
        self.order = np.argsort(-self.scores, kind='stable')
        ordered = self.scores[self.order]
        self.starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
        self.negative_order = np.flatnonzero(self.grades < 2)
        self.negative_order = self.negative_order[np.argsort(self.scores[self.negative_order], kind='stable')]

    def evaluate(self, weights=None, specificity=.90, max_fpr=.10):
        if not 0 < specificity <= 1 or not 0 < max_fpr <= 1:
            raise ValueError('Invalid ROC region or specificity')
        w = np.ones(len(self.scores), dtype=np.int64) if weights is None else np.asarray(weights)
        if w.shape != self.scores.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError('Invalid unit multiplicities')
        if not np.all(w == np.floor(w)):
            raise ValueError('Bootstrap multiplicities must be integers')
        nneg = int(w[self.grades < 2].sum())
        npos = int(w[self.grades >= 2].sum())
        ng2 = int(w[self.grades == 2].sum())
        if min(nneg, npos, ng2) <= 0:
            raise ValueError('Missing endpoint denominator')
        budget = int(math.floor((1 - specificity) * nneg + 1e-10))
        cumulative = np.cumsum(w[self.negative_order])
        boundary_index = np.searchsorted(cumulative, nneg - budget - 1, side='right')
        cutoff = float(np.nextafter(self.scores[self.negative_order[boundary_index]], np.inf))
        referred = self.scores >= cutoff
        grade2 = (w * referred)[self.grades == 2].sum() / ng2
        sensitivity = (w * referred)[self.grades >= 2].sum() / npos
        actual_sp = (w * (~referred))[self.grades < 2].sum() / nneg
        ordered_w = w[self.order]
        positive = self.grades[self.order] >= 2
        tp = np.add.reduceat(ordered_w * positive, self.starts)
        fp = np.add.reduceat(ordered_w * (~positive), self.starts)
        keep = tp + fp > 0
        tpr = np.r_[0., np.cumsum(tp[keep]) / npos]
        fpr = np.r_[0., np.cumsum(fp[keep]) / nneg]
        if max_fpr < 1:
            stop = np.searchsorted(fpr, max_fpr, side='right')
            value = np.interp(max_fpr, fpr[stop-1:stop+1], tpr[stop-1:stop+1])
            fpr = np.r_[fpr[:stop], max_fpr]
            tpr = np.r_[tpr[:stop], value]
        raw = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
        standardized = .5 * (1 + (raw - max_fpr**2 / 2) / (max_fpr - max_fpr**2 / 2))
        return {'cutoff': cutoff, 'grade2_sensitivity': float(grade2),
                'referable_sensitivity': float(sensitivity), 'specificity': float(actual_sp),
                'raw_pauc': raw, 'standardized_pauc': float(standardized)}


def load_records(root):
    root = Path(root)
    records, sources = {}, []
    for cohort in COHORTS:
        reference = None
        for model in MODELS:
            for seed in SEEDS:
                path = root / 'external' / cohort / ('seed' + str(seed)) / model / 'unit_predictions.csv'
                raw = path.read_bytes()
                with path.open(newline='', encoding='utf-8-sig') as f:
                    all_rows = list(csv.DictReader(f))
                if not all_rows: raise ValueError('Empty prediction file')
                ids = [r['unit_id'] for r in all_rows]
                if len(ids) != len(set(ids)): raise ValueError('Duplicate unit identifiers')
                rows = []
                for row in all_rows:
                    flag = row['primary_eligible'].lower()
                    if flag not in ('0', '1', 'false', 'true'): raise ValueError('Invalid eligibility flag')
                    if flag in ('1', 'true'): rows.append(row)
                rows.sort(key=lambda r: r['unit_id'])
                paired = [(r['unit_id'], int(r['reference_grade']), r.get('unit_type', ''),
                           r.get('image_count', ''), r.get('aggregation', ''),
                           r.get('gradable_images', ''), r.get('grouping_provenance', '')) for r in rows]
                if reference is None: reference = paired
                elif reference != paired: raise ValueError('Unpaired membership, grades or grouping metadata')
                records[cohort, model, seed] = FittedROC(
                    [float(r['referable_score']) for r in rows], [int(r['reference_grade']) for r in rows])
                sources.append({'path': path.relative_to(root).as_posix(),
                                'sha256': hashlib.sha256(raw).hexdigest(), 'primary_units': len(rows)})
    return records, sources


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(root, output, replicates=2000, rng_seed=20261013):
    if replicates != 0 and replicates < 2000:
        raise ValueError('Use zero for points only or at least 2000 replicates')
    output = Path(output)
    if output.exists(): raise FileExistsError('Refusing to overwrite analysis results')
    records, sources = load_records(root)
    output.mkdir(parents=True)
    rng = np.random.default_rng(rng_seed)
    estimates, effects = [], []
    for cohort in COHORTS:
        grades = records[cohort, MODELS[0], SEEDS[0]].grades
        points = {model: np.array([[records[cohort, model, s].evaluate()[m] for m in METRICS]
                                  for s in SEEDS]) for model in MODELS}
        samples = {model: np.empty((replicates, len(METRICS))) for model in MODELS}
        strata = [np.flatnonzero(grades == g) for g in range(5)]
        for b in range(replicates):
            weights = np.zeros(len(grades), dtype=np.int64)
            for idx in strata:
                if len(idx): weights[idx] = rng.multinomial(len(idx), np.full(len(idx), 1/len(idx)))
            for model in MODELS:
                values = [records[cohort, model, s].evaluate(weights) for s in SEEDS]
                samples[model][b] = [np.mean([v[m] for v in values]) for m in METRICS]
        for model in MODELS:
            for j, metric in enumerate(METRICS):
                row = {'cohort': cohort, 'model': model, 'metric': metric,
                       'estimate': float(points[model][:, j].mean()), 'ci95_low': '', 'ci95_high': ''}
                if replicates:
                    low, high = np.quantile(samples[model][:, j], [.025, .975], method='linear')
                    row.update(ci95_low=float(low), ci95_high=float(high))
                estimates.append(row)
        for control in MODELS[:-1]:
            for j, metric in enumerate(METRICS):
                row = {'cohort': cohort, 'candidate': MODELS[-1], 'control': control,
                       'metric': metric, 'effect': float((points[MODELS[-1]][:, j] - points[control][:, j]).mean()),
                       'ci95_low': '', 'ci95_high': ''}
                if replicates:
                    delta = samples[MODELS[-1]][:, j] - samples[control][:, j]
                    low, high = np.quantile(delta, [.025, .975], method='linear')
                    row.update(ci95_low=float(low), ci95_high=float(high))
                effects.append(row)
        print(cohort + ': completed', flush=True)
    write_csv(output / 'model_estimates.csv', estimates)
    write_csv(output / 'paired_effects.csv', effects)
    protocol = {'target_specificity': .90, 'max_fpr': .10, 'bootstrap_replicates': replicates,
                'bootstrap_rng_seed': rng_seed, 'optimization_seeds': list(SEEDS),
                'unit': 'Released primary patient/subject/examination group',
                'resampling': 'Whole units within maximum-grade strata; common multiplicities across all models/seeds',
                'cutoff_reestimated': True, 'probability_ensembling': False,
                'sensitivity_units': 'Proportions; multiply by 100 for percentage points',
                'pauc_units': 'Area units; not sensitivity percentages',
                'interval': 'Pointwise unadjusted percentile 95%; fixed fitted models and grade counts',
                'source_files': sources,
                'limitations': ['Same-label threshold fitting is exploratory, not deployment calibration',
                               'No correction for selection, retraining or unresolved patient dependence',
                               'No independent cross-source patient separation is established']}
    (output / 'analysis_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    analyze(args.input, args.output, args.replicates)
