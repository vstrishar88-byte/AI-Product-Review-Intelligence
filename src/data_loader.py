from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

DATASET_CANDIDATES = (
    "mteb/amazon_polarity",
)
DEFAULT_SUBSET_SIZE = 20000


def load_amazon_polarity_subset(sample_size: int = DEFAULT_SUBSET_SIZE, seed: int = 42):
    """Load a reproducible subset of the Amazon Polarity dataset.

    Uses a small sample that is suitable for local laptop training and keeps the
    setup realistic without training on the full multi-million-row dataset.
    """
    last_error = None
    for dataset_name in DATASET_CANDIDATES:
        try:
            dataset = load_dataset(dataset_name, split="train")
            dataset = dataset.shuffle(seed=seed)
            subset = dataset.select(range(min(sample_size, len(dataset))))
            return subset
        except Exception as exc:  # pragma: no cover - compatibility fallback for HF dataset IDs
            last_error = exc

    raise RuntimeError(
        "Failed to load the Amazon Polarity dataset from the available Hugging Face IDs."
    ) from last_error


def ensure_data_dir(path: str | Path | None = None):
    target = Path(path) if path else Path(__file__).resolve().parent.parent / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target
