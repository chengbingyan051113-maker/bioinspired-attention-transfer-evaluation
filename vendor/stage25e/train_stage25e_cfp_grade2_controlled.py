#!/usr/bin/env python3
"""Stage25E controlled CFP grade-2 internal-development trainer.

B1 changes the checkpoint/threshold selection target from UWF to CFP. B2 keeps
that policy and applies one prespecified 2x multiplier to CFP grade-2 examples.
The released test splits and all external datasets are deliberately unavailable
to this program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from stage22_screening_core import (
    LESIONS,
    fixed_split,
    lesion_metrics,
    sample_id_hash,
    screening_metrics_from_grade_probabilities,
)
from stage25e_screening_core import (
    cfp_checkpoint_selection_key,
    cfp_screening_metrics,
    is_better_cfp_selection,
)


VARIANTS = ("B1_cfp_selection_control", "B2_cfp_grade2_x2")
MODEL_VARIANT = "matched_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfp-manifest", required=True)
    parser.add_argument("--uwf-manifest", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=20260926)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lesion-weight", type=float, default=0.30)
    parser.add_argument("--ordinal-weight", type=float, default=0.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--adapter-bottleneck", type=int, default=160)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-specificity", type=float, default=0.90)
    parser.add_argument("--phase", choices=("smoke", "full_internal"), required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--full-authorization")
    return parser.parse_args()


def validate_contract(args: argparse.Namespace) -> None:
    if args.seed not in {2026, 2027, 2028}:
        raise ValueError("optimization seed must be one of 2026, 2027, 2028")
    if args.phase == "smoke":
        smoke_expected = {
            "seed": 2026,
            "split_seed": 20260926,
            "epochs": 1,
            "batch_size": 16,
            "image_size": 224,
            "num_workers": 4,
            "validation_fraction": 0.15,
            "encoder_lr": 2e-5,
            "head_lr": 3e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "lesion_weight": 0.30,
            "patience": 5,
        }
        for name, value in smoke_expected.items():
            if abs(float(getattr(args, name)) - float(value)) > 1e-12:
                raise ValueError(f"smoke {name} is locked to {value}")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be between 0 and 0.5")
    if abs(args.minimum_specificity - 0.90) > 1e-12:
        raise ValueError("minimum-specificity is locked to 0.90")
    if args.ordinal_weight != 0.0 or args.consistency_weight != 0.0:
        raise ValueError("Stage25E preserves the matched baseline without ordinal/consistency loss")
    if args.phase == "full_internal":
        if not args.full_authorization:
            raise ValueError("full_internal requires --full-authorization")
        expected = {
            "split_seed": 20260926,
            "epochs": 25,
            "batch_size": 16,
            "image_size": 384,
            "num_workers": 4,
            "validation_fraction": 0.15,
            "encoder_lr": 2e-5,
            "head_lr": 3e-4,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "lesion_weight": 0.30,
            "patience": 5,
        }
        for name, value in expected.items():
            if abs(float(getattr(args, name)) - float(value)) > 1e-12:
                raise ValueError(f"full_internal {name} is locked to {value}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol_lock(args: argparse.Namespace) -> dict:
    path = Path(args.protocol_lock).resolve()
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("stage") != "stage25e_cfp_grade2_controlled_internal_v1":
        raise ValueError("unexpected Stage25E protocol lock")
    factor = float(lock["variants"][args.variant]["cfp_grade2_multiplier"])
    expected = 1.0 if args.variant.startswith("B1_") else 2.0
    if factor != expected:
        raise ValueError("variant grade-2 multiplier does not match protocol")
    return lock


def verify_full_authorization(args: argparse.Namespace, protocol_lock: dict) -> str | None:
    if args.phase != "full_internal":
        return None
    path = Path(args.full_authorization).resolve()
    authorization = json.loads(path.read_text(encoding="utf-8"))
    if authorization.get("stage") != protocol_lock.get("stage"):
        raise ValueError("full authorization stage mismatch")
    if authorization.get("full_internal_authorized") is not True:
        raise ValueError("full_internal is not authorized")
    protocol_sha = file_sha256(Path(args.protocol_lock).resolve())
    if authorization.get("protocol_lock_sha256") != protocol_sha:
        raise ValueError("full authorization protocol SHA256 mismatch")
    return file_sha256(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MMRDRMultiTaskDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int, training: bool):
        self.frame = frame.reset_index(drop=True)
        operations: list = [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        ]
        if training:
            operations.extend(
                [
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ColorJitter(
                        brightness=0.10,
                        contrast=0.10,
                        saturation=0.08,
                        hue=0.02,
                    ),
                ]
            )
        operations.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.transform = transforms.Compose(operations)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        lesions = np.asarray([row[f"lesion_{name}"] for name in LESIONS], dtype=np.float32)
        masks = np.asarray([row[f"lesion_{name}_mask"] for name in LESIONS], dtype=np.float32)
        return {
            "image": tensor,
            "grade": torch.tensor(int(row["grade"]), dtype=torch.long),
            "lesion": torch.from_numpy(np.nan_to_num(lesions, nan=0.0)),
            "lesion_mask": torch.from_numpy(masks),
            "sample_id": str(row["sample_id"]),
        }


class ZeroInitializedResidualAdapter(nn.Module):
    def __init__(self, feature_dim: int, bottleneck: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, feature_dim)
        nn.init.kaiming_uniform_(self.down.weight, a=np.sqrt(5.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(self.norm(feature))))
        return feature + residual


class SharedMonotonicOrdinalAxis(nn.Module):
    """One shared severity direction with strictly ordered cut-points."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.severity = nn.Linear(feature_dim, 1)
        self.cutpoint_base = nn.Parameter(torch.tensor(-1.5))
        inverse_softplus_one = float(np.log(np.exp(1.0) - 1.0))
        self.cutpoint_gap_raw = nn.Parameter(
            torch.full((3,), inverse_softplus_one)
        )

    def cutpoints(self) -> torch.Tensor:
        gaps = nn.functional.softplus(self.cutpoint_gap_raw) + 1e-4
        return torch.cat(
            [
                self.cutpoint_base.reshape(1),
                self.cutpoint_base + torch.cumsum(gaps, dim=0),
            ]
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        severity = self.severity(feature)
        return severity - self.cutpoints().unsqueeze(0)


class ModalityBranch(nn.Module):
    def __init__(
        self,
        projection_dim: int,
        adapter_bottleneck: int,
        use_adapter: bool,
    ):
        super().__init__()
        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.feature_dim = self.encoder.classifier[1].in_features
        self.encoder.classifier = nn.Identity()
        self.adapter = (
            ZeroInitializedResidualAdapter(self.feature_dim, adapter_bottleneck)
            if use_adapter
            else nn.Identity()
        )
        self.grade_head = nn.Sequential(nn.Dropout(0.25), nn.Linear(self.feature_dim, 5))
        self.lesion_head = nn.Sequential(nn.Dropout(0.20), nn.Linear(self.feature_dim, len(LESIONS)))
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(512, projection_dim),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_feature = self.encoder(image)
        feature = self.adapter(raw_feature)
        return {
            "feature": feature,
            "grade_logits": self.grade_head(feature),
            "lesion_logits": self.lesion_head(feature),
            "embedding": nn.functional.normalize(self.projection(feature), dim=1),
        }


class Stage20CModel(nn.Module):
    def __init__(self, variant: str, projection_dim: int, adapter_bottleneck: int):
        super().__init__()
        use_adapter = variant != "matched_baseline"
        use_ordinal = variant in {"adapter_ordinal", "task_preserving_full"}
        use_consistency = variant == "task_preserving_full"
        self.variant = variant
        self.use_adapter = use_adapter
        self.use_ordinal = use_ordinal
        self.use_consistency = use_consistency
        self.cfp = ModalityBranch(projection_dim, adapter_bottleneck, use_adapter)
        self.uwf = ModalityBranch(projection_dim, adapter_bottleneck, use_adapter)
        feature_dim = self.cfp.feature_dim
        self.shared_ordinal_axis = SharedMonotonicOrdinalAxis(feature_dim) if use_ordinal else None
        if use_consistency:
            self.lesion_risk_raw_weights = nn.Parameter(torch.zeros(len(LESIONS)))
            self.lesion_risk_bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("lesion_risk_raw_weights", None)
            self.register_parameter("lesion_risk_bias", None)

    def branch(self, modality: str) -> ModalityBranch:
        return self.cfp if modality == "CFP" else self.uwf

    def forward_modality(self, modality: str, image: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.branch(modality)(image)
        if self.shared_ordinal_axis is not None:
            output["ordinal_logits"] = self.shared_ordinal_axis(output["feature"])
        return output

    def lesion_risk_logits(
        self,
        lesion_logits: torch.Tensor,
        lesion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_consistency:
            raise RuntimeError("lesion-risk head requested for a model without consistency")
        probability = torch.sigmoid(lesion_logits)
        weights = nn.functional.softplus(self.lesion_risk_raw_weights) + 1e-6
        weighted_mask = lesion_mask * weights.unsqueeze(0)
        denominator = weighted_mask.sum(dim=1).clamp_min(1e-6)
        centered_risk = (weighted_mask * (probability - 0.5)).sum(dim=1) / denominator
        logits = self.lesion_risk_bias + 4.0 * centered_risk
        valid = lesion_mask.sum(dim=1) > 0
        return logits, valid


def validate_manifest(frame: pd.DataFrame, modality: str) -> dict:
    if set(frame["modality"].astype(str)) != {modality}:
        raise RuntimeError(f"{modality} manifest modality mismatch")
    if set(frame["task_name"].astype(str)) != {"dr_5class"}:
        raise RuntimeError(f"{modality} is not the five-level DR task")
    required = {
        "sample_id",
        "image_path",
        "official_split",
        "grade",
        *{f"lesion_{name}" for name in LESIONS},
        *{f"lesion_{name}_mask" for name in LESIONS},
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{modality} manifest missing columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise RuntimeError(f"{modality} has duplicate sample IDs")
    if set(frame["official_split"].astype(str)) != {"train", "test"}:
        raise RuntimeError(f"{modality} official split is invalid")
    if not set(pd.to_numeric(frame["grade"], errors="raise").astype(int)).issubset(set(range(5))):
        raise RuntimeError(f"{modality} contains invalid DR grades")
    masked_elements = 0
    for name in LESIONS:
        target = pd.to_numeric(frame[f"lesion_{name}"], errors="coerce")
        mask = pd.to_numeric(frame[f"lesion_{name}_mask"], errors="raise").astype(int)
        if not set(mask.unique()).issubset({0, 1}):
            raise RuntimeError(f"{modality}/{name} has non-binary masks")
        invalid_labeled = (mask == 1) & ~target.isin([0.0, 1.0])
        if invalid_labeled.any():
            raise RuntimeError(f"{modality}/{name} labels non-binary lesion values")
        masked_elements += int((mask == 0).sum())
    return {
        "rows": int(len(frame)),
        "masked_lesion_elements": masked_elements,
        "manifest_sample_id_sha256": sample_id_hash(frame["sample_id"].astype(str)),
    }


def make_loader(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    training: bool,
    loader_seed: int,
) -> DataLoader:
    return DataLoader(
        MMRDRMultiTaskDataset(frame, args.image_size, training),
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(loader_seed),
    )


def grade_weights(frame: pd.DataFrame) -> torch.Tensor:
    counts = frame["grade"].value_counts().reindex(range(5), fill_value=0).to_numpy(float)
    if np.any(counts == 0):
        raise RuntimeError(f"training split lacks a DR grade: {counts.tolist()}")
    weights = len(frame) / (5.0 * counts)
    weights = np.clip(weights / weights.mean(), 0.25, 4.0)
    return torch.tensor(weights, dtype=torch.float32)


def normalized_weighted_grade_loss(
    logits: torch.Tensor,
    grade: torch.Tensor,
    class_weight: torch.Tensor,
    label_smoothing: float,
    grade2_multiplier: float,
) -> torch.Tensor:
    """Preserve native weighted-mean scale while changing only grade-2 influence."""
    values = nn.functional.cross_entropy(
        logits,
        grade,
        weight=class_weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    multiplier = torch.where(
        grade == 2,
        torch.as_tensor(grade2_multiplier, device=grade.device, dtype=values.dtype),
        torch.ones((), device=grade.device, dtype=values.dtype),
    )
    denominator = (class_weight[grade].to(values.dtype) * multiplier).sum().clamp_min(1e-12)
    return (values * multiplier).sum() / denominator


def verify_loss_equivalence(device: torch.device, label_smoothing: float) -> float:
    """Require multiplier=1 to reproduce the Stage22 weighted CE reduction."""
    logits = torch.tensor(
        [
            [0.2, -0.1, 0.4, 0.0, -0.3],
            [-0.2, 0.3, 0.1, 0.6, -0.5],
            [0.5, 0.2, -0.4, 0.1, -0.1],
            [0.1, -0.2, 0.8, -0.3, 0.0],
            [-0.4, 0.2, 0.1, 0.3, 0.7],
        ],
        dtype=torch.float32,
        device=device,
    )
    grade = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
    weight = torch.tensor([0.7, 1.1, 1.8, 0.9, 1.4], dtype=torch.float32, device=device)
    native = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)(logits, grade)
    controlled = normalized_weighted_grade_loss(
        logits, grade, weight, label_smoothing, grade2_multiplier=1.0
    )
    difference = float(torch.abs(native - controlled).item())
    if difference > 2e-6:
        raise RuntimeError(
            f"controlled loss does not reproduce native weighted CE: difference={difference}"
        )
    return difference


def ordinal_pos_weights(frame: pd.DataFrame) -> torch.Tensor:
    grade = frame["grade"].to_numpy(int)
    weights = []
    for threshold in range(4):
        positive = float(np.sum(grade > threshold))
        negative = float(np.sum(grade <= threshold))
        weights.append(np.clip(negative / max(positive, 1.0), 0.25, 8.0))
    return torch.tensor(weights, dtype=torch.float32)


def lesion_pos_weights(frame: pd.DataFrame) -> torch.Tensor:
    values = []
    for name in LESIONS:
        target = pd.to_numeric(frame[f"lesion_{name}"], errors="coerce")
        mask = pd.to_numeric(frame[f"lesion_{name}_mask"], errors="coerce").fillna(0).astype(bool)
        valid = target[mask]
        positive = float((valid == 1).sum())
        negative = float((valid == 0).sum())
        values.append(np.clip(negative / max(positive, 1.0), 0.5, 20.0))
    return torch.tensor(values, dtype=torch.float32)


def masked_lesion_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    values = nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def ordinal_targets(grade: torch.Tensor) -> torch.Tensor:
    thresholds = torch.arange(4, device=grade.device).unsqueeze(0)
    return (grade.unsqueeze(1) > thresholds).to(dtype=torch.float32)


def grade_referable_probability(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 2:].sum(dim=1)


def move_batch(batch: dict | None, device: torch.device) -> dict | None:
    if batch is None:
        return None
    return {
        "image": batch["image"].to(device, non_blocking=True),
        "grade": batch["grade"].to(device, non_blocking=True),
        "lesion": batch["lesion"].to(device, non_blocking=True),
        "lesion_mask": batch["lesion_mask"].to(device, non_blocking=True),
    }


def train_epoch(
    model: Stage20CModel,
    cfp_loader: DataLoader,
    uwf_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    criteria: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    iterators = {"CFP": iter(cfp_loader), "UWF": iter(uwf_loader)}
    steps = max(len(cfp_loader), len(uwf_loader))
    totals = {"total": 0.0, "grade": 0.0, "lesion": 0.0, "ordinal": 0.0, "consistency": 0.0}
    for _ in range(steps):
        batches: dict[str, dict | None] = {}
        for modality in ("CFP", "UWF"):
            try:
                batches[modality] = next(iterators[modality])
            except StopIteration:
                batches[modality] = None
            batches[modality] = move_batch(batches[modality], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            grade_losses: list[torch.Tensor] = []
            lesion_losses: list[torch.Tensor] = []
            ordinal_losses: list[torch.Tensor] = []
            consistency_losses: list[torch.Tensor] = []
            for modality in ("CFP", "UWF"):
                batch = batches[modality]
                if batch is None:
                    continue
                output = model.forward_modality(modality, batch["image"])
                multiplier = args.cfp_grade2_multiplier if modality == "CFP" else 1.0
                grade_losses.append(
                    normalized_weighted_grade_loss(
                        output["grade_logits"],
                        batch["grade"],
                        criteria[f"grade_{modality}"],
                        args.label_smoothing,
                        multiplier,
                    )
                )
                lesion_losses.append(
                    masked_lesion_loss(
                        output["lesion_logits"],
                        batch["lesion"],
                        batch["lesion_mask"],
                        criteria[f"lesion_{modality}"],
                    )
                )
                if model.use_ordinal:
                    ordinal_losses.append(
                        nn.functional.binary_cross_entropy_with_logits(
                            output["ordinal_logits"],
                            ordinal_targets(batch["grade"]),
                            pos_weight=criteria[f"ordinal_{modality}"],
                        )
                    )
                if model.use_consistency:
                    lesion_risk_logits, valid = model.lesion_risk_logits(
                        output["lesion_logits"], batch["lesion_mask"]
                    )
                    referable_target = (batch["grade"] >= 2).to(torch.float32)
                    supervised_risk = nn.functional.binary_cross_entropy_with_logits(
                        lesion_risk_logits[valid], referable_target[valid]
                    )
                    grade_probability = grade_referable_probability(output["grade_logits"])
                    agreement = nn.functional.mse_loss(
                        torch.sigmoid(lesion_risk_logits[valid]), grade_probability[valid]
                    )
                    consistency_losses.append(supervised_risk + agreement)
            grade_loss = torch.stack(grade_losses).mean()
            lesion_loss = torch.stack(lesion_losses).mean()
            ordinal_loss = (
                torch.stack(ordinal_losses).mean() if ordinal_losses else grade_loss * 0.0
            )
            consistency_loss = (
                torch.stack(consistency_losses).mean() if consistency_losses else grade_loss * 0.0
            )
            total_loss = (
                grade_loss
                + args.lesion_weight * lesion_loss
                + args.ordinal_weight * ordinal_loss
                + args.consistency_weight * consistency_loss
            )
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals["total"] += float(total_loss.item())
        totals["grade"] += float(grade_loss.item())
        totals["lesion"] += float(lesion_loss.item())
        totals["ordinal"] += float(ordinal_loss.item())
        totals["consistency"] += float(consistency_loss.item())
    return {name: value / steps for name, value in totals.items()}


@torch.no_grad()
def evaluate_modality(
    model: Stage20CModel,
    modality: str,
    loader: DataLoader,
    device: torch.device,
    minimum_specificity: float,
    seed: int,
    fixed_threshold: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    sample_ids: list[str] = []
    grades: list[np.ndarray] = []
    grade_probabilities: list[np.ndarray] = []
    lesions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lesion_probabilities: list[np.ndarray] = []
    ordinal_probabilities: list[np.ndarray] = []
    for batch in loader:
        output = model.forward_modality(modality, batch["image"].to(device, non_blocking=True))
        sample_ids.extend(batch["sample_id"])
        grades.append(batch["grade"].numpy())
        grade_probabilities.append(torch.softmax(output["grade_logits"], dim=1).cpu().numpy())
        lesions.append(batch["lesion"].numpy())
        masks.append(batch["lesion_mask"].numpy())
        lesion_probabilities.append(torch.sigmoid(output["lesion_logits"]).cpu().numpy())
        if model.use_ordinal:
            ordinal_probabilities.append(torch.sigmoid(output["ordinal_logits"]).cpu().numpy())
    grade = np.concatenate(grades)
    grade_probability = np.concatenate(grade_probabilities)
    lesion = np.concatenate(lesions)
    mask = np.concatenate(masks)
    lesion_probability = np.concatenate(lesion_probabilities)
    if modality == "CFP":
        metrics = cfp_screening_metrics(
            grade,
            grade_probability,
            minimum_specificity=minimum_specificity,
            fixed_threshold=fixed_threshold,
        )
    else:
        metrics = screening_metrics_from_grade_probabilities(
            grade,
            grade_probability,
            minimum_specificity=minimum_specificity,
            fixed_threshold=fixed_threshold,
        )
    metrics.update(lesion_metrics(lesion, mask, lesion_probability))
    metrics.update({"modality": modality, "seed": seed})
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "grade_true": grade,
            "grade_pred": grade_probability.argmax(axis=1),
            "modality": modality,
            "seed": seed,
        }
    )
    for index in range(5):
        predictions[f"prob_grade_{index}"] = grade_probability[:, index]
    for index, name in enumerate(LESIONS):
        predictions[f"lesion_{name}_true"] = lesion[:, index]
        predictions[f"lesion_{name}_mask"] = mask[:, index]
        predictions[f"lesion_{name}_prob"] = lesion_probability[:, index]
    if ordinal_probabilities:
        ordinal_probability = np.concatenate(ordinal_probabilities)
        for threshold in range(4):
            predictions[f"prob_grade_gt_{threshold}"] = ordinal_probability[:, threshold]
        monotonic_violation = np.maximum(np.diff(ordinal_probability, axis=1), 0.0)
        metrics["ordinal_monotonic_violation_mean"] = float(monotonic_violation.mean())
    return metrics, predictions


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load the checkpoint written locally by this training process.

    Torch 1.13 exposes ``weights_only`` but its restricted unpickler cannot
    read every ordinary dictionary produced by ``torch.save``.  The file is
    created by this script in the same run, so explicitly disabling the
    restricted loader is safe here.  The TypeError fallback covers older
    Torch releases that do not accept the keyword.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    validate_contract(args)
    protocol_lock = load_protocol_lock(args)
    full_authorization_sha256 = verify_full_authorization(args, protocol_lock)
    args.cfp_grade2_multiplier = float(
        protocol_lock["variants"][args.variant]["cfp_grade2_multiplier"]
    )
    set_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_paths = {
        "CFP": Path(args.cfp_manifest).resolve(),
        "UWF": Path(args.uwf_manifest).resolve(),
    }
    manifest_file_sha256 = {
        modality: file_sha256(path) for modality, path in manifest_paths.items()
    }
    for modality, path in manifest_paths.items():
        expected = protocol_lock["inputs"]["manifests"][modality]
        if manifest_file_sha256[modality] != expected["sha256"]:
            raise RuntimeError(f"{modality} manifest SHA256 mismatch: {path}")
    cfp = pd.read_csv(manifest_paths["CFP"], low_memory=False)
    uwf = pd.read_csv(manifest_paths["UWF"], low_memory=False)
    manifest_audit = {
        "CFP": validate_manifest(cfp, "CFP"),
        "UWF": validate_manifest(uwf, "UWF"),
    }
    split_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {
        "CFP": fixed_split(cfp, args.validation_fraction, args.split_seed),
        "UWF": fixed_split(uwf, args.validation_fraction, args.split_seed),
    }
    split_audit: dict[str, object] = {
        "optimization_seed": args.seed,
        "immutable_split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "patient_ids_available": False,
        "split_unit": "image",
        "released_test_loader_constructed": False,
        "released_test_evaluated": False,
        "external_data_loaded": False,
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_audit": manifest_audit,
        "modalities": {},
    }
    for modality, (train, validation, test) in split_frames.items():
        split_audit["modalities"][modality] = {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "train_sample_ids_sha256": sample_id_hash(train["sample_id"]),
            "validation_sample_ids_sha256": sample_id_hash(validation["sample_id"]),
            "test_sample_ids_sha256": sample_id_hash(test["sample_id"]),
            "overlap": 0,
            "train_grade_distribution": {
                str(key): int(value) for key, value in train["grade"].value_counts().sort_index().items()
            },
            "validation_grade_distribution": {
                str(key): int(value) for key, value in validation["grade"].value_counts().sort_index().items()
            },
        }
        expected = protocol_lock["inputs"]["split_sample_id_sha256"][modality]
        observed = split_audit["modalities"][modality]
        for split_name in ("train", "validation", "test"):
            if observed[f"{split_name}_sample_ids_sha256"] != expected[split_name]:
                raise RuntimeError(f"{modality} {split_name} sample ID hash mismatch")
    save_json(output / "split_audit.json", split_audit)

    loaders: dict[str, dict[str, DataLoader]] = {}
    for offset, modality in enumerate(("CFP", "UWF")):
        train, validation, _test_never_loaded = split_frames[modality]
        loaders[modality] = {
            "train": make_loader(train, args, True, args.seed + offset * 10),
            "validation": make_loader(validation, args, False, args.split_seed + offset * 10 + 1),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage25E training requires CUDA")
    loss_equivalence_difference = verify_loss_equivalence(device, args.label_smoothing)
    model = Stage20CModel(MODEL_VARIANT, args.projection_dim, args.adapter_bottleneck).to(device)
    parameter_audit = {
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "adapter_parameters": int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if ".adapter." in name)
        ),
        "ordinal_parameters": int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if "shared_ordinal_axis" in name)
        ),
        "consistency_parameters": int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if "lesion_risk" in name)
        ),
    }
    save_json(output / "parameter_audit.json", parameter_audit)

    criteria: dict[str, torch.Tensor | nn.Module] = {}
    for modality in ("CFP", "UWF"):
        train = split_frames[modality][0]
        criteria[f"grade_{modality}"] = grade_weights(train).to(device)
        criteria[f"lesion_{modality}"] = lesion_pos_weights(train).to(device)
        criteria[f"ordinal_{modality}"] = ordinal_pos_weights(train).to(device)
    loss_audit = {
        "native_weighted_cross_entropy_equivalence_absolute_difference": loss_equivalence_difference,
        "normalization": "sum(per_example_loss * sample_multiplier) / sum(target_class_weight * sample_multiplier)",
        "label_smoothing": args.label_smoothing,
        "lesion_weight": args.lesion_weight,
        "cfp_grade2_multiplier": args.cfp_grade2_multiplier,
        "uwf_grade2_multiplier": 1.0,
        "grade_class_weights": {
            modality: [float(value) for value in criteria[f"grade_{modality}"].detach().cpu()]
            for modality in ("CFP", "UWF")
        },
        "only_intended_loss_difference_between_B1_and_B2": "CFP grade==2 sample multiplier",
    }
    save_json(output / "loss_audit.json", loss_audit)

    encoder_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for branch in (model.cfp, model.uwf):
        encoder_parameters.extend(branch.encoder.parameters())
        head_parameters.extend(branch.adapter.parameters())
        head_parameters.extend(branch.grade_head.parameters())
        head_parameters.extend(branch.lesion_head.parameters())
        head_parameters.extend(branch.projection.parameters())
    if model.shared_ordinal_axis is not None:
        head_parameters.extend(model.shared_ordinal_axis.parameters())
    if model.lesion_risk_raw_weights is not None:
        head_parameters.extend([model.lesion_risk_raw_weights, model.lesion_risk_bias])
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    print("=" * 92)
    print("Stage25E controlled CFP grade-2 internal development")
    print("Variant/config:", args.variant, args.config_id)
    print("Optimization/split seed:", args.seed, args.split_seed)
    print("Released test loader/evaluation: False False")
    print("CFP grade-2 multiplier:", args.cfp_grade2_multiplier)
    print("Selection: CFP grade-2 sensitivity at validation specificity >=", args.minimum_specificity)
    for modality in ("CFP", "UWF"):
        parts = split_frames[modality]
        print(f"{modality} train/validation/test: {len(parts[0])} {len(parts[1])} {len(parts[2])}")

    checkpoint_path = output / "best_model.pt"
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_values = train_epoch(
            model,
            loaders["CFP"]["train"],
            loaders["UWF"]["train"],
            optimizer,
            scaler,
            criteria,
            device,
            args,
        )
        cfp_metrics, _ = evaluate_modality(
            model, "CFP", loaders["CFP"]["validation"], device, args.minimum_specificity, args.seed
        )
        uwf_metrics, _ = evaluate_modality(
            model, "UWF", loaders["UWF"]["validation"], device, args.minimum_specificity, args.seed
        )
        current_key = cfp_checkpoint_selection_key(cfp_metrics)
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_values.items()},
            "cfp_val_grade2_sensitivity_at_specificity_0_90": cfp_metrics[
                "grade2_sensitivity_at_selected_threshold"
            ],
            "cfp_val_sensitivity_at_specificity_0_90": cfp_metrics[
                "sensitivity_at_specificity_0_90"
            ],
            "cfp_val_specificity": cfp_metrics["specificity_at_selected_threshold"],
            "cfp_val_referable_auc": cfp_metrics["referable_auc"],
            "cfp_val_referable_auprc": cfp_metrics["referable_auprc"],
            "cfp_val_qwk": cfp_metrics["quadratic_weighted_kappa"],
            "cfp_val_brier": cfp_metrics["referable_brier"],
            "uwf_val_sensitivity_at_specificity_0_90": uwf_metrics["sensitivity_at_specificity_0_90"],
            "uwf_val_specificity": uwf_metrics["specificity_at_selected_threshold"],
            "uwf_val_referable_auc": uwf_metrics["referable_auc"],
            "uwf_val_referable_auprc": uwf_metrics["referable_auprc"],
            "uwf_val_qwk": uwf_metrics["quadratic_weighted_kappa"],
            "uwf_val_grade_mae": uwf_metrics["grade_mae"],
            "uwf_val_brier": uwf_metrics["referable_brier"],
            "uwf_val_lesion_macro_auc": uwf_metrics["lesion_macro_auc"],
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | total={train_values['total']:.4f} | "
            f"grade={train_values['grade']:.4f} | lesion={train_values['lesion']:.4f} | "
            f"ordinal={train_values['ordinal']:.4f} | consistency={train_values['consistency']:.4f} | "
            f"CFP G2 sens@spec90={cfp_metrics['grade2_sensitivity_at_selected_threshold']:.4f} | "
            f"CFP overall sens={cfp_metrics['sensitivity_at_specificity_0_90']:.4f} | "
            f"CFP AUC={cfp_metrics['referable_auc']:.4f} | "
            f"UWF AUC={uwf_metrics['referable_auc']:.4f} | time={row['seconds']:.1f}s",
            flush=True,
        )
        if is_better_cfp_selection(current_key, best_key):
            best_key = current_key
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "variant": args.variant,
                    "architecture_variant": MODEL_VARIANT,
                    "config_id": args.config_id,
                    "optimization_seed": args.seed,
                    "split_seed": args.split_seed,
                    "best_epoch": best_epoch,
                    "selection_key": list(best_key),
                    "selection_key_names": [
                        "cfp_grade2_sensitivity_at_spec_ge_0_90",
                        "cfp_referable_sensitivity_at_spec_ge_0_90",
                        "cfp_referable_auc",
                    ],
                    "cfp_grade2_multiplier": args.cfp_grade2_multiplier,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            print("Early stopping", flush=True)
            break

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])
    validation_hashes: dict[str, str] = {}
    for modality in ("CFP", "UWF"):
        metrics, predictions = evaluate_modality(
            model,
            modality,
            loaders[modality]["validation"],
            device,
            args.minimum_specificity,
            args.seed,
        )
        metrics.update(
            {
                "variant": args.variant,
                "config_id": args.config_id,
                "best_epoch": best_epoch,
                "optimization_seed": args.seed,
                "immutable_split_seed": args.split_seed,
                "selection_role": "CFP_primary_internal_development" if modality == "CFP" else "UWF_secondary_forgetting_audit",
                "cfp_grade2_multiplier": args.cfp_grade2_multiplier,
            }
        )
        predictions["variant"] = args.variant
        predictions["config_id"] = args.config_id
        validation_hashes[modality] = sample_id_hash(predictions["sample_id"])
        save_json(output / f"validation_metrics_{modality}.json", metrics)
        predictions.to_csv(output / f"validation_predictions_{modality}.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")
    save_json(output / "training_config.json", vars(args))
    protocol_audit = {
        "schema_version": 1,
        "stage": "stage25e_cfp_grade2_controlled_internal_v1",
        "variant": args.variant,
        "architecture_variant": MODEL_VARIANT,
        "config_id": args.config_id,
        "optimization_seed": args.seed,
        "immutable_split_seed": args.split_seed,
        "uwf_validation_sample_ids_sha256": validation_hashes["UWF"],
        "cfp_validation_sample_ids_sha256": validation_hashes["CFP"],
        "primary_selection_used_only_cfp_validation": True,
        "uwf_used_for_selection": False,
        "cfp_grade2_multiplier": args.cfp_grade2_multiplier,
        "released_test_loader_constructed": False,
        "released_test_evaluated": False,
        "external_data_loaded": False,
        "patient_identifiers_available": False,
        "inference_unit": "image",
        "uwf_invalid_ih_value_2_handling": "manifest target NaN and mask 0; excluded from lesion loss and metrics",
        "protocol_lock_sha256": file_sha256(Path(args.protocol_lock).resolve()),
        "full_authorization_sha256": full_authorization_sha256,
        "protocol_integrity_passed": True,
        "claim_boundary": "Internal image-level development only; no released test or external evaluation.",
    }
    save_json(output / "protocol_audit.json", protocol_audit)
    (output / "completed.ok").write_text("STAGE25E_RUN_COMPLETE=1\n", encoding="utf-8")
    print("=" * 92)
    print("Stage25E internal run completed")
    print("BEST_EPOCH=", best_epoch)
    print("BEST_SELECTION_KEY=", best_key)
    print("RELEASED_TEST_EVALUATED=0")
    print("OUTPUT=", output)
    print("STAGE25E_RUN_COMPLETE=1")


if __name__ == "__main__":
    main()
