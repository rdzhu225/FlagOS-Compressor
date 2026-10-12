"""Calibration text loading without a hard dependency on ``datasets``."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import torch

from flagos_compressor.core.policy import CalibrationPolicy


def _read_local_text(path: Path, text_column: str) -> list[str]:
    """Read one of the documented local calibration schemas.

    ``.txt``/``.text`` uses one sample per non-empty line. ``.jsonl`` uses one
    JSON string or ``{text_column: string}`` object per line. ``.json`` accepts
    a top-level list with the same string/object records, or an object whose
    ``data`` or ``text_column`` value contains that list.
    """
    suffix = path.suffix.lower()
    if suffix in {".txt", ".text"}:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".jsonl":
        records = [
            json.loads(line)
            # Unicode line/paragraph separators are valid inside JSON strings;
            # only newline characters delimit JSONL records.
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
    elif suffix == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = records.get("data", records.get(text_column, records))
    else:
        raise ValueError(f"Unsupported calibration file type: {path.suffix}")
    if isinstance(records, str):
        records = [records]
    if not isinstance(records, list):
        records = []
    texts = [
        record if isinstance(record, str) else record.get(text_column)
        for record in records
        if isinstance(record, (str, dict))
    ]
    return [text for text in texts if isinstance(text, str) and text.strip()]


def load_calibration_texts(policy: CalibrationPolicy) -> list[str]:
    if isinstance(policy.data, tuple):
        texts = list(policy.data)
    elif isinstance(policy.data, str):
        path = Path(policy.data)
        if path.exists():
            texts = _read_local_text(path, policy.text_column)
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face dataset calibration requires the optional "
                    "'datasets' package; alternatively pass a local txt/json/jsonl file"
                ) from exc
            dataset = load_dataset(policy.data, split=policy.split)
            if hasattr(dataset, "shuffle"):
                dataset = dataset.shuffle(seed=policy.seed)
            texts = []
            candidate_limit = max(policy.samples * 10, policy.samples)
            for row in dataset:
                text = row.get(policy.text_column)
                if isinstance(text, str):
                    texts.append(text)
                    if len(texts) >= candidate_limit:
                        break
    else:
        raise ValueError(
            "GPTQ/AWQ/AutoRound requires calibration.data or --calibration-data"
        )
    if not texts:
        raise ValueError("Calibration source did not contain any usable text")
    random.Random(policy.seed).shuffle(texts)
    return texts


def build_calibration_batches(tokenizer: Any, policy: CalibrationPolicy) -> list[dict[str, torch.Tensor]]:
    """Tokenize and pack examples into fixed-length calibration blocks.

    AutoAWQ evaluates attention modules on all captured examples at once, so
    their sequence dimensions must agree. Like AutoAWQ's calibration helper,
    we concatenate short documents and split full blocks; if the corpus is
    smaller than one block we retain a single short block rather than failing.
    """
    texts = load_calibration_texts(policy)
    token_stream: list[int] = []
    accepted = 0
    for value in texts:
        encoded = tokenizer.encode(
            value,
            add_special_tokens=True,
            truncation=False,
        )
        if not encoded or len(encoded) > policy.sequence_length:
            continue
        token_stream.extend(int(token) for token in encoded)
        accepted += 1
        if accepted >= policy.samples:
            break
    if not token_stream:
        raise ValueError(
            "Calibration source has no non-empty examples at or below "
            f"sequence_length={policy.sequence_length}"
        )

    sequence_length = policy.sequence_length
    chunks = [
        token_stream[start : start + sequence_length]
        for start in range(0, len(token_stream) - sequence_length + 1, sequence_length)
    ]
    if not chunks:
        chunks = [token_stream]
    batches: list[dict[str, torch.Tensor]] = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long)
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        )
    return batches


__all__ = ["build_calibration_batches", "load_calibration_texts"]
