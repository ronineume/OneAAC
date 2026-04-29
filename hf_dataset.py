"""HuggingFace datasets adapter for PCVRHyFormer.

Loads data via ``from datasets import load_dataset`` and converts it into
the same tensor layout that ``PCVRHyFormerRankingTrainer`` expects.

Usage::

    from hf_dataset import get_hf_data
    train_loader, valid_loader, dataset = get_hf_data(
        dataset_name="TAAC2026/data_sample_1000",
        batch_size=32,
    )
"""

import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from typing import Dict, List, Tuple, Optional, Any

from dataset import FeatureSchema, NUM_TIME_BUCKETS, BUCKET_BOUNDARIES


def _max_val_in_dataset(hf_dataset, col_name: str) -> int:
    """Infer vocab size (max value + 1) from a HF dataset column."""
    col = hf_dataset[col_name]
    max_val = 0
    for v in col:
        if isinstance(v, list):
            for x in v:
                if x is not None and x > 0:
                    max_val = max(max_val, x)
        else:
            if v is not None and v > 0:
                max_val = max(max_val, v)
    return int(max_val)


def _max_len_in_dataset(hf_dataset, col_name: str) -> int:
    """Find maximum list length in a HF dataset column."""
    col = hf_dataset[col_name]
    max_len = 0
    for v in col:
        if isinstance(v, list):
            max_len = max(max_len, len(v))
    return max_len


def _is_timestamp_column(hf_dataset, col_name: str) -> bool:
    """Heuristic: a timestamp column contains large values (> 1e9)."""
    col = hf_dataset[col_name]
    for v in col:
        if isinstance(v, list) and v:
            for x in v:
                if x is not None and x > 0:
                    return x > 1_000_000_000
        elif v is not None and v > 0:
            return v > 1_000_000_000
    return False


class HFDataset(Dataset):
    """PyTorch Dataset wrapper for a HuggingFace ``datasets.Dataset``.

    Automatically infers feature schema (vocab sizes, dimensions, sequence
    domains) from the raw column layout and pre-converts everything to numpy
    so that ``__getitem__`` is a cheap tensor-wrap.
    """

    def __init__(
        self,
        hf_dataset,
        seq_max_lens: Optional[Dict[str, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
        schema_source: Optional["HFDataset"] = None,
    ) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset
        self.seq_max_lens = seq_max_lens or {}
        self.clip_vocab = clip_vocab
        self.is_training = is_training

        # --- Schema inference / copy ---
        if schema_source is not None:
            self._copy_schema_from(schema_source)
        else:
            self._infer_schema()
        # --- Preprocess to numpy ---
        self._preprocess()

        logging.info(
            f"HFDataset: {len(hf_dataset)} rows, "
            f"user_int_dim={self.user_int_schema.total_dim}, "
            f"item_int_dim={self.item_int_schema.total_dim}, "
            f"user_dense_dim={self.user_dense_schema.total_dim}, "
            f"item_dense_dim={self.item_dense_schema.total_dim}, "
            f"seq_domains={list(self.seq_domains.keys())}"
        )

    def _copy_schema_from(self, source: "HFDataset") -> None:
        """Copy all schema-related attributes from another HFDataset.

        This guarantees that train / validation splits share the exact same
        feature dimensions and vocab sizes, preventing index errors when the
        validation subset happens to have shorter lists or smaller max values.
        """
        self._user_int_cols = source._user_int_cols
        self._item_int_cols = source._item_int_cols
        self._user_dense_cols = source._user_dense_cols
        self._item_dense_cols = source._item_dense_cols
        self.seq_domains = source.seq_domains
        self.user_int_schema = source.user_int_schema
        self.user_int_vocab_sizes = source.user_int_vocab_sizes
        self.item_int_schema = source.item_int_schema
        self.item_int_vocab_sizes = source.item_int_vocab_sizes
        self.user_dense_schema = source.user_dense_schema
        self.item_dense_schema = source.item_dense_schema
        self.seq_domain_vocab_sizes = source.seq_domain_vocab_sizes
        self._seq_maxlen = source._seq_maxlen

    # ------------------------------------------------------------------ #
    # Schema inference
    # ------------------------------------------------------------------ #

    def _infer_schema(self) -> None:
        cols = self.hf_dataset.column_names
        sample = self.hf_dataset[0]

        # ---- user_int: user_int_feats_{fid} ----
        self._user_int_cols: List[Tuple[str, int, int, int]] = []
        user_int_names = sorted([c for c in cols if c.startswith("user_int_feats_")])
        for c in user_int_names:
            fid = int(c.split("_")[-1])
            is_list = isinstance(sample[c], list)
            max_len = _max_len_in_dataset(self.hf_dataset, c) if is_list else 1
            vocab = _max_val_in_dataset(self.hf_dataset, c) + 1
            self._user_int_cols.append((c, fid, max_len, vocab))

        # ---- item_int ----
        self._item_int_cols: List[Tuple[str, int, int, int]] = []
        item_int_names = sorted([c for c in cols if c.startswith("item_int_feats_")])
        for c in item_int_names:
            fid = int(c.split("_")[-1])
            is_list = isinstance(sample[c], list)
            max_len = _max_len_in_dataset(self.hf_dataset, c) if is_list else 1
            vocab = _max_val_in_dataset(self.hf_dataset, c) + 1
            self._item_int_cols.append((c, fid, max_len, vocab))

        # ---- user_dense ----
        self._user_dense_cols: List[Tuple[str, int, int]] = []
        user_dense_names = sorted([c for c in cols if c.startswith("user_dense_feats_")])
        for c in user_dense_names:
            fid = int(c.split("_")[-1])
            max_len = _max_len_in_dataset(self.hf_dataset, c)
            self._user_dense_cols.append((c, fid, max_len))

        # ---- item_dense (may be empty in this dataset) ----
        self._item_dense_cols: List[Tuple[str, int, int]] = []
        item_dense_names = sorted([
            c for c in cols
            if c.startswith("item_dense_feats_") and not c.startswith("user_dense_feats_")
        ])
        for c in item_dense_names:
            fid = int(c.split("_")[-1])
            max_len = _max_len_in_dataset(self.hf_dataset, c)
            self._item_dense_cols.append((c, fid, max_len))

        # ---- sequence domains ----
        self.seq_domains: Dict[str, Dict[str, Any]] = {}
        domain_prefixes = set()
        for c in cols:
            if "_seq_" in c:
                prefix = c.rsplit("_", 1)[0]
                domain_prefixes.add(prefix)

        for prefix in sorted(domain_prefixes):
            domain = prefix
            domain_cols = sorted([c for c in cols if c.startswith(prefix + "_")])

            ts_col = None
            ts_fid = None
            for c in domain_cols:
                if _is_timestamp_column(self.hf_dataset, c):
                    ts_col = c
                    ts_fid = int(c.split("_")[-1])
                    break

            sideinfo_cols = [c for c in domain_cols if c != ts_col]
            self.seq_domains[domain] = {
                "cols": domain_cols,
                "sideinfo_cols": sideinfo_cols,
                "ts_col": ts_col,
                "ts_fid": ts_fid,
            }

        # ---- Build FeatureSchema objects ----
        self.user_int_schema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for _, fid, max_len, vocab in self._user_int_cols:
            self.user_int_schema.add(fid, max_len)
            self.user_int_vocab_sizes.extend([vocab] * max_len)

        self.item_int_schema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for _, fid, max_len, vocab in self._item_int_cols:
            self.item_int_schema.add(fid, max_len)
            self.item_int_vocab_sizes.extend([vocab] * max_len)

        self.user_dense_schema = FeatureSchema()
        for _, fid, max_len in self._user_dense_cols:
            self.user_dense_schema.add(fid, max_len)

        self.item_dense_schema = FeatureSchema()
        for _, fid, max_len in self._item_dense_cols:
            self.item_dense_schema.add(fid, max_len)

        # ---- Sequence vocab sizes (sideinfo only) ----
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        for domain, info in self.seq_domains.items():
            vs_list = []
            for c in info["sideinfo_cols"]:
                vs = _max_val_in_dataset(self.hf_dataset, c) + 1
                vs_list.append(vs)
            self.seq_domain_vocab_sizes[domain] = vs_list

        # ---- Per-domain sequence max length ----
        self._seq_maxlen: Dict[str, int] = {}
        for domain, info in self.seq_domains.items():
            max_len = 0
            for c in info["cols"]:
                ml = _max_len_in_dataset(self.hf_dataset, c)
                max_len = max(max_len, ml)
            self._seq_maxlen[domain] = self.seq_max_lens.get(domain, max_len)

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess(self) -> None:
        N = len(self.hf_dataset)

        # Non-sequence buffers
        self.user_int = np.zeros((N, self.user_int_schema.total_dim), dtype=np.int64)
        self.item_int = np.zeros((N, self.item_int_schema.total_dim), dtype=np.int64)
        self.user_dense = np.zeros((N, self.user_dense_schema.total_dim), dtype=np.float32)
        self.item_dense = np.zeros((N, self.item_dense_schema.total_dim), dtype=np.float32)

        self.timestamps = np.zeros(N, dtype=np.int64)
        self.labels = np.zeros(N, dtype=np.int64)
        self.user_ids: List[Any] = []

        # Sequence buffers
        self.seq_data: Dict[str, np.ndarray] = {}
        self.seq_lens: Dict[str, np.ndarray] = {}
        self.seq_time_buckets: Dict[str, np.ndarray] = {}

        for domain, info in self.seq_domains.items():
            max_len = self._seq_maxlen[domain]
            n_feats = len(info["sideinfo_cols"])
            self.seq_data[domain] = np.zeros((N, n_feats, max_len), dtype=np.int64)
            self.seq_lens[domain] = np.zeros(N, dtype=np.int64)
            self.seq_time_buckets[domain] = np.zeros((N, max_len), dtype=np.int64)

        # Fill buffers row by row
        for i in range(N):
            row = self.hf_dataset[i]

            self.timestamps[i] = row.get("timestamp", 0)
            if self.is_training:
                self.labels[i] = 1 if row.get("label_type", 0) == 2 else 0
            self.user_ids.append(row.get("user_id", 0))

            # -- user_int --
            offset = 0
            for c, fid, max_len, vocab in self._user_int_cols:
                v = row[c]
                if isinstance(v, list):
                    use_len = min(len(v), max_len)
                    arr = np.array(v[:use_len], dtype=np.int64)
                    arr[arr <= 0] = 0
                    if self.clip_vocab and vocab > 0:
                        arr = np.clip(arr, 0, vocab - 1)
                    self.user_int[i, offset : offset + use_len] = arr
                else:
                    val = v if v is not None and v > 0 else 0
                    if self.clip_vocab and vocab > 0:
                        val = min(val, vocab - 1)
                    self.user_int[i, offset] = val
                offset += max_len

            # -- item_int --
            offset = 0
            for c, fid, max_len, vocab in self._item_int_cols:
                v = row[c]
                if isinstance(v, list):
                    use_len = min(len(v), max_len)
                    arr = np.array(v[:use_len], dtype=np.int64)
                    arr[arr <= 0] = 0
                    if self.clip_vocab and vocab > 0:
                        arr = np.clip(arr, 0, vocab - 1)
                    self.item_int[i, offset : offset + use_len] = arr
                else:
                    val = v if v is not None and v > 0 else 0
                    if self.clip_vocab and vocab > 0:
                        val = min(val, vocab - 1)
                    self.item_int[i, offset] = val
                offset += max_len

            # -- user_dense --
            offset = 0
            for c, fid, max_len in self._user_dense_cols:
                v = row[c]
                if isinstance(v, list):
                    use_len = min(len(v), max_len)
                    arr = np.array(v[:use_len], dtype=np.float32)
                    self.user_dense[i, offset : offset + use_len] = arr
                else:
                    self.user_dense[i, offset] = float(v) if v is not None else 0.0
                offset += max_len

            # -- item_dense --
            offset = 0
            for c, fid, max_len in self._item_dense_cols:
                v = row[c]
                if isinstance(v, list):
                    use_len = min(len(v), max_len)
                    arr = np.array(v[:use_len], dtype=np.float32)
                    self.item_dense[i, offset : offset + use_len] = arr
                else:
                    self.item_dense[i, offset] = float(v) if v is not None else 0.0
                offset += max_len

            # -- sequence features --
            for domain, info in self.seq_domains.items():
                max_len = self._seq_maxlen[domain]
                ts_col = info["ts_col"]
                sideinfo_cols = info["sideinfo_cols"]

                # Sequence length from first column (all columns have same length)
                first_col = info["cols"][0]
                v_first = row[first_col]
                seq_len = len(v_first) if isinstance(v_first, list) else 1
                use_len = min(seq_len, max_len)
                self.seq_lens[domain][i] = use_len

                # Fill sideinfo
                for feat_idx, c in enumerate(sideinfo_cols):
                    v = row[c]
                    if isinstance(v, list):
                        arr = np.array(v[:use_len], dtype=np.int64)
                        arr[arr <= 0] = 0
                        self.seq_data[domain][i, feat_idx, :use_len] = arr

                # Time buckets
                if ts_col:
                    ts_vals = row.get(ts_col, [])
                    if isinstance(ts_vals, list):
                        ts_padded = np.zeros(max_len, dtype=np.int64)
                        ts_padded[:use_len] = ts_vals[:use_len]
                        time_diff = np.maximum(self.timestamps[i] - ts_padded, 0)
                        raw_buckets = np.clip(
                            np.searchsorted(BUCKET_BOUNDARIES, time_diff),
                            0, len(BUCKET_BOUNDARIES) - 1,
                        )
                        buckets = raw_buckets + 1
                        buckets[ts_padded == 0] = 0
                        self.seq_time_buckets[domain][i] = buckets

    # ------------------------------------------------------------------ #
    # PyTorch Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "user_int_feats": torch.from_numpy(self.user_int[idx]),
            "user_dense_feats": torch.from_numpy(self.user_dense[idx]),
            "item_int_feats": torch.from_numpy(self.item_int[idx]),
            "item_dense_feats": torch.from_numpy(self.item_dense[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "timestamp": torch.tensor(self.timestamps[idx], dtype=torch.long),
            "user_id": self.user_ids[idx],
            "_seq_domains": list(self.seq_domains.keys()),
        }
        for domain in self.seq_domains:
            item[domain] = torch.from_numpy(self.seq_data[domain][idx])
            item[f"{domain}_len"] = torch.tensor(self.seq_lens[domain][idx], dtype=torch.long)
            item[f"{domain}_time_bucket"] = torch.from_numpy(self.seq_time_buckets[domain][idx])
        return item


from torch.utils.data._utils.collate import default_collate


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate that keeps ``_seq_domains`` as a plain list instead of
    stacking it into a list-of-lists.
    """
    domains = batch[0]["_seq_domains"]
    # Temporarily remove _seq_domains so default_collate handles only tensors
    for b in batch:
        del b["_seq_domains"]
    result = default_collate(batch)
    result["_seq_domains"] = domains
    return result


def get_hf_data(
    dataset_name: str = "TAAC2026/data_sample_1000",
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    seed: int = 42,
    seq_max_lens: Optional[Dict[str, int]] = None,
    clip_vocab: bool = True,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, HFDataset]:
    """Create train / valid DataLoaders from a HuggingFace dataset.

    Returns:
        ``(train_loader, valid_loader, train_dataset)``.
        The third element exposes the inferred schema needed to build the model.
    """
    # Detect local Arrow dataset (saved via save_to_disk) vs. remote HF hub
    if os.path.isdir(dataset_name):
        logging.info(f"Loading local dataset from disk: {dataset_name}")
        ds = load_from_disk(dataset_name)
        if "train" in ds:
            train_hf = ds["train"]
        else:
            raise ValueError(f"Local dataset {dataset_name} missing 'train' split. Found: {list(ds.keys())}")
        if "valid" in ds:
            # Already split: use as-is
            train_split = ds["train"]
            valid_split = ds["valid"]
            train_dataset = HFDataset(
                train_split,
                seq_max_lens=seq_max_lens,
                clip_vocab=clip_vocab,
                is_training=True,
            )
            valid_dataset = HFDataset(
                valid_split,
                seq_max_lens=seq_max_lens,
                clip_vocab=clip_vocab,
                is_training=True,
                schema_source=train_dataset,
            )
            use_cuda = torch.cuda.is_available()
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=use_cuda,
                collate_fn=_collate_fn,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=use_cuda,
                collate_fn=_collate_fn,
            )
            logging.info(
                f"HF train: {len(train_dataset)} rows, valid: {len(valid_dataset)} rows, "
                f"batch_size={batch_size}"
            )
            return train_loader, valid_loader, train_dataset
    else:
        logging.info(f"Loading remote dataset: {dataset_name}")
        ds = load_dataset(dataset_name)
        train_hf = ds["train"]

    # Train / validation split
    split = train_hf.train_test_split(test_size=valid_ratio, seed=seed)
    train_split = split["train"]
    valid_split = split["test"]

    train_dataset = HFDataset(
        train_split,
        seq_max_lens=seq_max_lens,
        clip_vocab=clip_vocab,
        is_training=True,
    )
    valid_dataset = HFDataset(
        valid_split,
        seq_max_lens=seq_max_lens,
        clip_vocab=clip_vocab,
        is_training=True,
        schema_source=train_dataset,
    )

    use_cuda = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        collate_fn=_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        collate_fn=_collate_fn,
    )

    logging.info(
        f"HF train: {len(train_dataset)} rows, valid: {len(valid_dataset)} rows, "
        f"batch_size={batch_size}"
    )

    return train_loader, valid_loader, train_dataset
