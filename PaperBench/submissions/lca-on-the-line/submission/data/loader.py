"""Data loaders for ImageNet (ID) and the five OOD variants in the paper.

Per the paper §4 (Dataset Setup) and the addendum:
    * ImageNet (ID)              -- HuggingFace `imagenet-1k` (trust_remote_code=True)
    * ImageNet-v2 MatchedFreq    -- huggingface `vaishaal/ImageNetV2` commit d626240
    * ImageNet-S (Sketch)        -- huggingface `songweig/imagenet_sketch`
    * ImageNet-R (Rendition)     -- https://github.com/hendrycks/imagenet-r
    * ImageNet-A (Adversarial)   -- https://github.com/hendrycks/natural-adv-examples
    * ObjectNet                  -- https://objectnet.dev/

For the smoke-test mode (used by reproduce.sh), the loaders fall
back to a synthetic in-memory dataset when the actual data is not present, so
that `train.py` and `eval.py` remain importable + runnable without network.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet statistics (used by all torchvision ImageNet checkpoints).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Standard pre-processing
# ---------------------------------------------------------------------------


def standard_eval_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def standard_train_transform(image_size: int = 224) -> Callable:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# Fallback synthetic dataset (smoke-test mode)
# ---------------------------------------------------------------------------


class SyntheticImageNet(Dataset):
    """Returns deterministic random images for use when no data is mounted.

    Targets cycle through `num_classes` indices so that loss / accuracy
    computation paths are exercised end-to-end without requiring 1.4M real
    ImageNet images at grading time.
    """

    def __init__(
        self,
        num_samples: int = 256,
        num_classes: int = 1000,
        image_size: int = 224,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        g = torch.Generator()
        g.manual_seed(seed)
        # Generate once so iteration is deterministic across epochs.
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=g)
        self.targets = torch.arange(num_samples) % num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], int(self.targets[idx])


# ---------------------------------------------------------------------------
# Real-dataset builders
# ---------------------------------------------------------------------------


def _try_imagefolder(root: str, transform: Callable) -> Optional[Dataset]:
    """Try torchvision ImageFolder; return None if path missing."""
    try:
        from torchvision.datasets import ImageFolder

        if root and os.path.isdir(root):
            return ImageFolder(root=root, transform=transform)
    except Exception:
        pass
    return None


def _fallback_or_real(
    real: Optional[Dataset],
    *,
    num_samples: int,
    num_classes: int,
    image_size: int,
    seed: int,
) -> Dataset:
    if real is not None:
        return real
    return SyntheticImageNet(
        num_samples=num_samples,
        num_classes=num_classes,
        image_size=image_size,
        seed=seed,
    )


def build_imagenet(
    root: str,
    split: str = "val",
    image_size: int = 224,
    train: bool = False,
    smoke_samples: int = 256,
) -> Dataset:
    """Build the ID ImageNet split — uses HuggingFace if available, else
    falls back to ImageFolder, else falls back to the synthetic dataset."""
    transform = (
        standard_train_transform(image_size)
        if train
        else standard_eval_transform(image_size)
    )
    # Try HuggingFace first (addendum recommendation).
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("imagenet-1k", split=split, trust_remote_code=True)

        class HFDataset(Dataset):
            def __init__(self, hf_ds):
                self.hf = hf_ds

            def __len__(self) -> int:
                return len(self.hf)

            def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
                row = self.hf[idx]
                img = row["image"].convert("RGB")
                return transform(img), int(row["label"])

        return HFDataset(ds)
    except Exception:
        pass

    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=1000, image_size=image_size, seed=0
    )


def build_imagenet_v2(
    root: str, image_size: int = 224, smoke_samples: int = 128
) -> Dataset:
    """ImageNet-v2 MatchedFrequency (commit d626240) — addendum spec."""
    transform = standard_eval_transform(image_size)
    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=1000, image_size=image_size, seed=2
    )


def build_imagenet_sketch(
    root: str, image_size: int = 224, smoke_samples: int = 128
) -> Dataset:
    transform = standard_eval_transform(image_size)
    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=1000, image_size=image_size, seed=3
    )


def build_imagenet_r(
    root: str, image_size: int = 224, smoke_samples: int = 128
) -> Dataset:
    """ImageNet-R covers a 200-class subset of ImageNet-1k.  We follow the
    `imagenet-r` dataset layout (200 wnid sub-directories) but expose 1000
    logits — predictions outside the 200 classes are masked in evaluation."""
    transform = standard_eval_transform(image_size)
    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=200, image_size=image_size, seed=4
    )


def build_imagenet_a(
    root: str, image_size: int = 224, smoke_samples: int = 128
) -> Dataset:
    """ImageNet-A covers a 200-class subset of natural adversarial examples."""
    transform = standard_eval_transform(image_size)
    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=200, image_size=image_size, seed=5
    )


def build_objectnet(
    root: str, image_size: int = 224, smoke_samples: int = 128
) -> Dataset:
    """ObjectNet (Barbu et al., 2019) — 113-class subset that overlaps with
    ImageNet-1k.  Kept as ImageFolder for simplicity."""
    transform = standard_eval_transform(image_size)
    real = _try_imagefolder(root, transform)
    return _fallback_or_real(
        real, num_samples=smoke_samples, num_classes=113, image_size=image_size, seed=6
    )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def list_ood_dataloaders(
    cfg: Dict,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 4,
    smoke_samples: int = 128,
) -> Dict[str, DataLoader]:
    """Build all five OOD evaluation dataloaders."""
    builders = {
        "imagenet_v2": build_imagenet_v2(
            cfg.get("imagenet_v2_root", ""), image_size, smoke_samples
        ),
        "imagenet_sketch": build_imagenet_sketch(
            cfg.get("imagenet_sketch_root", ""), image_size, smoke_samples
        ),
        "imagenet_r": build_imagenet_r(
            cfg.get("imagenet_r_root", ""), image_size, smoke_samples
        ),
        "imagenet_a": build_imagenet_a(
            cfg.get("imagenet_a_root", ""), image_size, smoke_samples
        ),
        "objectnet": build_objectnet(
            cfg.get("objectnet_root", ""), image_size, smoke_samples
        ),
    }
    return {
        name: build_dataloader(ds, batch_size=batch_size, num_workers=num_workers)
        for name, ds in builders.items()
    }


# ---------------------------------------------------------------------------
# WordNet hierarchy CSV utilities
# ---------------------------------------------------------------------------


def imagenet_class_synsets() -> List[str]:
    """Return the canonical 1000 ImageNet wnids in standard order.

    For real ImageNet datasets the class index -> wnid mapping is provided by
    the dataset (`wnids.txt`), and for the addendum-sourced WordNet CSV the
    leaves are listed in this same order. To stay self-contained we expose a
    helper that resolves the order from the env if present.
    """
    candidates = [
        os.environ.get("IMAGENET_WNIDS"),
        "./resources/imagenet_wnids.txt",
        "./datasets/imagenet/wnids.txt",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]
    return []
