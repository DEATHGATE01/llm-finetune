from __future__ import annotations

import re
from typing import Dict, List


DatasetItem = Dict[str, str]


def normalize_text(value: str) -> str:
	"""Normalize text to stabilize duplicate detection."""
	return re.sub(r"\s+", " ", value.strip().lower())


def is_valid_record(record: DatasetItem, min_output_words: int = 12) -> bool:
	"""Keep only records that are complete and informative."""
	instruction = record.get("instruction", "").strip()
	output = record.get("output", "").strip()

	if not instruction or not output:
		return False

	if len(instruction) < 8:
		return False

	if len(output.split()) < min_output_words:
		return False

	return True


def deduplicate_records(records: List[DatasetItem]) -> List[DatasetItem]:
	"""Remove near-identical records using normalized instruction-output keys."""
	seen = set()
	unique_records: List[DatasetItem] = []

	for item in records:
		key = (
			normalize_text(item.get("instruction", "")),
			normalize_text(item.get("output", "")),
		)
		if key in seen:
			continue
		seen.add(key)
		unique_records.append(item)

	return unique_records


def clean_dataset(records: List[DatasetItem], min_output_words: int = 12) -> List[DatasetItem]:
	"""Run quality filters and deduplication for phase-1 data."""
	filtered = [r for r in records if is_valid_record(r, min_output_words=min_output_words)]
	return deduplicate_records(filtered)

