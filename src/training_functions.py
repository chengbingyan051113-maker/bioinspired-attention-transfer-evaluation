"""Exact original loader, epoch and validation function bodies."""
import hashlib
import numpy as np

def make_loader(torch, baseline, frame, spec, training, seed):
    return torch.utils.data.DataLoader(
        baseline.MMRDRMultiTaskDataset(frame, spec['image_size'], training),
        batch_size=spec['batch_size_per_modality'], shuffle=training,
        num_workers=spec['num_workers'], pin_memory=True, persistent_workers=False,
        worker_init_fn=baseline.seed_worker,
        generator=torch.Generator().manual_seed(seed))

def train_epoch(torch, baseline, model, frames, optimizer, criteria, device, spec, seed, epoch):
    epoch_seed = seed + epoch * 1000
    baseline.set_seed(epoch_seed)
    loaders = {m: make_loader(torch, baseline, frames[m]['train'], spec, True, epoch_seed + i * 10)
               for i, m in enumerate(('CFP', 'UWF'))}
    iterators = {m: iter(loader) for m, loader in loaders.items()}
    streams, first_batches = {m: [] for m in loaders}, {}
    model.train()
    totals = {'total': 0.0, 'grade': 0.0, 'lesion': 0.0}
    steps = max(len(loader) for loader in loaders.values())
    max_grad_norm = 0.0
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        grade_losses, lesion_losses = [], []
        for modality in ('CFP', 'UWF'):
            batch = next(iterators[modality], None)
            if batch is None:
                continue
            ids = list(batch['sample_id'])
            streams[modality].extend(ids)
            if modality not in first_batches:
                first_batches[modality] = {'sample_ids': ids,
                    'augmented_image_tensor_sha256': hashlib.sha256(batch['image'].numpy().tobytes()).hexdigest()}
            batch = baseline.move_batch(batch, device)
            prediction = model.forward_modality(modality, batch['image'])
            grade_losses.append(baseline.normalized_weighted_grade_loss(
                prediction['grade_logits'], batch['grade'], criteria['grade_' + modality],
                spec['label_smoothing'], spec['grade2_multiplier']))
            lesion_losses.append(baseline.masked_lesion_loss(prediction['lesion_logits'],
                batch['lesion'], batch['lesion_mask'], criteria['lesion_' + modality]))
        grade_loss = torch.stack(grade_losses).mean()
        lesion_loss = torch.stack(lesion_losses).mean()
        loss = grade_loss + spec['lesion_loss_weight'] * lesion_loss
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Non-finite FP32 loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip_norm'],
                                            error_if_nonfinite=True)
        max_grad_norm = max(max_grad_norm, float(norm))
        if epoch == 1 and step == 0:
            if not float(norm) > 0:
                raise RuntimeError('First full-batch step has no active gradient')
            if model.variant != 'M0R':
                active = torch.stack([p.grad.abs().sum() for name, p in model.named_parameters()
                    if '.stage_attention.' in name and p.grad is not None]).sum()
                if not float(active) > 0:
                    raise RuntimeError('Attention first full-batch step has no gradient')
        optimizer.step()
        for key, value in [('total', loss), ('grade', grade_loss), ('lesion', lesion_loss)]:
            totals[key] += float(value.detach())
        if (step + 1) % 100 == 0 or step + 1 == steps:
            print(f'TRAIN epoch={epoch} step={step + 1}/{steps} loss={float(loss.detach()):.6f}', flush=True)
    for modality, ids in streams.items():
        if len(ids) != len(frames[modality]['train']) or len(set(ids)) != len(ids) or set(ids) != set(frames[modality]['train'].sample_id):
            raise RuntimeError('Training stream did not consume each sample exactly once')
    del iterators, loaders
    return {key: value / steps for key, value in totals.items()}, {
        'epoch_seed': epoch_seed, 'steps': steps, 'max_unclipped_gradient_norm': max_grad_norm,
        'first_batches': first_batches,
        'ordered_training_stream_sha256': {m: hashlib.sha256(('\n'.join(ids) + '\n').encode()).hexdigest()
                                           for m, ids in streams.items()}}

def evaluate(torch, baseline, model, modality, loader, frame, device, spec, seed):
    metrics, predictions = baseline.evaluate_modality(
        model, modality, loader, device, spec['minimum_specificity'], seed)
    if list(predictions.sample_id) != list(frame.sample_id):
        raise RuntimeError('Validation membership/order changed')
    probability = predictions[[f'prob_grade_{i}' for i in range(5)]].to_numpy(float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1, atol=1e-5):
        raise RuntimeError('Invalid validation probabilities')
    # The unchanged legacy UWF helper uses +inf for its reject-all candidate.
    # Serialize an equivalent finite candidate only in that boundary case.
    threshold = metrics['selected_threshold_specificity_0_90']
    if modality == 'UWF' and threshold == np.inf:
        finite = np.nextafter(float(probability[:, 2:].sum(axis=1).max()), np.inf)
        metrics.update(baseline.screening_metrics_from_grade_probabilities(
            predictions.grade_true.to_numpy(), probability, spec['minimum_specificity'], fixed_threshold=finite))
        metrics['threshold_source'] = 'internal_validation_finite_reject_all'
    for key in ('referable_auc', 'sensitivity_at_specificity_0_90', 'specificity_at_selected_threshold',
                'selected_threshold_specificity_0_90'):
        if not np.isfinite(metrics[key]):
            raise RuntimeError('Non-finite required validation metric: ' + key)
    if metrics['specificity_at_selected_threshold'] + 1e-12 < spec['minimum_specificity']:
        raise RuntimeError('Validation operating point violates specificity constraint')
    metrics['interpretation'] = 'apparent internal development validation; selected checkpoint/threshold'
    return metrics, predictions
