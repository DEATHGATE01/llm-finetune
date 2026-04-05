from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


DatasetItem = Dict[str, str]


def to_instruction_schema(records: List[DatasetItem]) -> List[DatasetItem]:
	"""Format records to the assignment schema: instruction, input, output."""
	formatted: List[DatasetItem] = []

	for item in records:
		formatted.append(
			{
				"instruction": item["instruction"].strip(),
				"input": item.get("input", "").strip(),
				"output": item["output"].strip(),
			}
		)

	return formatted


def sort_dataset(records: List[DatasetItem]) -> List[DatasetItem]:
	"""Sort for stable diffs and repeatable experiments."""
	return sorted(records, key=lambda x: (x["instruction"].lower(), x["output"].lower()))


def build_instruction_text(record: DatasetItem) -> str:
	"""Convert a record into a plain instruction-tuning text format."""
	instruction = record["instruction"].strip()
	input_text = record.get("input", "").strip()
	output_text = record["output"].strip()

	if input_text:
		return (
			"<|instruction|>\n"
			f"{instruction}\n\n"
			"<|input|>\n"
			f"{input_text}\n\n"
			"<|response|>\n"
			f"{output_text}"
		)

	return (
		"<|instruction|>\n"
		f"{instruction}\n\n"
		"<|response|>\n"
		f"{output_text}"
	)


def to_instruction_records(records: List[DatasetItem]) -> List[Dict[str, str]]:
	"""Create instruction-style training rows with a single text field."""
	return [{"text": build_instruction_text(item)} for item in records]


def to_chat_records(records: List[DatasetItem]) -> List[Dict[str, List[Dict[str, str]]]]:
	"""Create chat-style rows compatible with modern supervised fine-tuning workflows."""
	chat_rows: List[Dict[str, List[Dict[str, str]]]] = []

	for item in records:
		user_content = item["instruction"].strip()
		input_text = item.get("input", "").strip()
		if input_text:
			user_content = f"{user_content}\n\nAdditional context:\n{input_text}"

		chat_rows.append(
			{
				"messages": [
					{"role": "user", "content": user_content},
					{"role": "assistant", "content": item["output"].strip()},
				]
			}
		)

	return chat_rows


def train_val_split(records: List[DatasetItem], val_ratio: float, seed: int) -> Dict[str, List[DatasetItem]]:
	"""Split records into train/validation sets with deterministic shuffle."""
	if not 0 < val_ratio < 1:
		raise ValueError("val_ratio must be between 0 and 1")

	shuffled = records[:]
	rng = random.Random(seed)
	rng.shuffle(shuffled)

	val_size = max(1, int(len(shuffled) * val_ratio))
	val_records = shuffled[:val_size]
	train_records = shuffled[val_size:]

	return {"train": train_records, "val": val_records}


def save_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for row in rows:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset(path: Path) -> List[DatasetItem]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Dataset must be a JSON array")
	return to_instruction_schema(data)


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
	root = project_root()
	parser = argparse.ArgumentParser(description="Phase 2 dataset formatter")
	parser.add_argument(
		"--input",
		type=Path,
		default=root / "data" / "final_dataset.json",
		help="Path to cleaned phase-1 dataset JSON",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=root / "data" / "processed",
		help="Directory for processed phase-2 files",
	)
	parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser


def run_phase2(input_path: Path, output_dir: Path, val_ratio: float, seed: int) -> Dict[str, int]:
	records = sort_dataset(load_dataset(input_path))
	splits = train_val_split(records, val_ratio=val_ratio, seed=seed)

	train_instruction = to_instruction_records(splits["train"])
	val_instruction = to_instruction_records(splits["val"])
	train_chat = to_chat_records(splits["train"])
	val_chat = to_chat_records(splits["val"])

	save_jsonl(output_dir / "train_instruct.jsonl", train_instruction)
	save_jsonl(output_dir / "val_instruct.jsonl", val_instruction)
	save_jsonl(output_dir / "train_chat.jsonl", train_chat)
	save_jsonl(output_dir / "val_chat.jsonl", val_chat)

	return {
		"total": len(records),
		"train": len(splits["train"]),
		"val": len(splits["val"]),
		"train_instruction": len(train_instruction),
		"val_instruction": len(val_instruction),
		"train_chat": len(train_chat),
		"val_chat": len(val_chat),
	}


def main() -> None:
	args = build_arg_parser().parse_args()
	stats = run_phase2(
		input_path=args.input,
		output_dir=args.output_dir,
		val_ratio=args.val_ratio,
		seed=args.seed,
	)

	print("Phase 2 formatting complete")
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()

