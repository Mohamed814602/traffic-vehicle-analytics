"""
Custom Ultralytics trainer that uses a WeightedRandomSampler instead of
uniform random shuffling -- rare-class images get sampled more often
per epoch, as a genuine alternative/complement to physically duplicating
files on disk (oversample_rare_classes.py). Uses the ORIGINAL unmodified
image folder; nothing is written to disk.

This required subclassing DetectionTrainer.get_dataloader(), since
Ultralytics' public train() API has no argument for a custom sampler --
verified by reading build_dataloader()'s actual source, not assumed.

Multi-GPU (--device 0,1) requires a genuinely different sampler
(DistributedWeightedSampler below), not a plain WeightedRandomSampler --
without partitioning, every GPU would independently draw from the ENTIRE
dataset, silently breaking the usual distributed-training guarantee that
each rank sees a distinct shard per step. Confirmed via Ultralytics' own
source that it calls `sampler.set_epoch(epoch)` each epoch whenever
RANK != -1, so any custom multi-GPU sampler must implement that method.

Usage:
    from ultralytics import YOLO
    from weighted_sampler_trainer import WeightedDetectionTrainer

    model = YOLO("yolo11s.pt")
    model.train(trainer=WeightedDetectionTrainer, data="data.yaml", ...)
"""
import math
import os

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first


def compute_image_weights(dataset, num_classes: int) -> np.ndarray:
    """
    One weight per image = inverse frequency of the RAREST class present
    in that image (max of per-class inverse frequencies, not sum/product)
    -- same principle as oversample_rare_classes.py's "use max, not
    multiply" rule, so an image with multiple rare classes doesn't get an
    extreme, runaway weight relative to everything else.

    Background-only images (no boxes at all) get the lowest weight
    (equal to the most common class's inverse frequency).
    """
    class_counts = np.zeros(num_classes, dtype=np.int64)
    per_image_classes = []
    for label in dataset.labels:
        cls = np.asarray(label["cls"]).reshape(-1).astype(int)
        present = set(cls.tolist())
        per_image_classes.append(present)
        for c in present:
            if 0 <= c < num_classes:
                class_counts[c] += 1

    class_counts = np.maximum(class_counts, 1)  # avoid division by zero
    inv_freq = 1.0 / class_counts

    weights = np.empty(len(dataset.labels), dtype=np.float64)
    for i, present in enumerate(per_image_classes):
        valid = [inv_freq[c] for c in present if 0 <= c < num_classes]
        weights[i] = max(valid) if valid else inv_freq.min()
    return weights


class DistributedWeightedSampler(Sampler):
    """
    Weighted sampling that ALSO correctly partitions across GPUs.
    All ranks generate the SAME full weighted draw using a shared,
    epoch-dependent seed, then each rank takes its own interleaved slice
    -- deterministic and non-overlapping across ranks with no cross-
    process communication needed.

    epoch_multiplier: draws epoch_multiplier * len(weights) total samples
    per epoch (before partitioning across ranks), not just len(weights).
    Matters for a fair comparison against file-based oversampling, which
    inflates the actual dataset size (e.g. 2.24x in this project) --
    without this, a WeightedRandomSampler-based run sees meaningfully
    FEWER total images per epoch than an equivalent oversampled run at
    the same nominal epoch number, understating its real progress.
    """

    def __init__(self, weights, num_replicas: int, rank: int, seed: int = 0, epoch_multiplier: float = 1.0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        total_draws = max(1, round(len(weights) * epoch_multiplier))
        self.num_samples_per_replica = math.ceil(total_draws / num_replicas)
        self.total_size = self.num_samples_per_replica * num_replicas

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples_per_replica


class WeightedDetectionTrainer(DetectionTrainer):
    """Drop-in replacement for Ultralytics' DetectionTrainer that uses
    weighted sampling for the TRAIN loader only -- validation always uses
    the standard (unweighted) loader, so evaluation metrics stay honest.

    Class attribute EPOCH_MULTIPLIER controls how many samples are drawn
    per epoch, as a multiple of the original dataset size (default 1.0 =
    same as before). Set it before calling model.train(trainer=...) to
    match an equivalent file-based oversampling run's inflated size, e.g.:
        WeightedDetectionTrainer.EPOCH_MULTIPLIER = 2.24
    """

    EPOCH_MULTIPLIER = 1.0

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        if mode == "val":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        num_classes = len(self.data["names"])
        weights = compute_image_weights(dataset, num_classes)
        num_draws = max(1, round(len(weights) * self.EPOCH_MULTIPLIER))

        if rank == -1:
            sampler = WeightedRandomSampler(weights, num_samples=num_draws, replacement=True)
        else:
            world_size = torch.cuda.device_count() or 1
            sampler = DistributedWeightedSampler(
                weights, num_replicas=world_size, rank=rank, epoch_multiplier=self.EPOCH_MULTIPLIER
            )

        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()
        nw = min(os.cpu_count() // max(nd, 1), self.args.workers)

        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,  # sampler determines order, not shuffle=True
            num_workers=nw,
            sampler=sampler,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
        )
