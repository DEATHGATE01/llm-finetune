from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.metrics import aggregate_scores, evaluate_pair
from src.utils.config import get_paths


def load_predictions(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError(f"Expected a JSON list in {path}")
	return data


def evaluate_predictions(rows: List[Dict[str, str]]) -> Dict[str, object]:
	per_item = []
	for row in rows:
		reference = row.get("reference", "")
		prediction = row.get("prediction", "")
		score = evaluate_pair(reference, prediction)
		per_item.append(
			{
				"instruction": row.get("instruction", ""),
				"metrics": score,
			}
		)

	summary = aggregate_scores([entry["metrics"] for entry in per_item])
	return {
		"count": len(per_item),
		"summary": summary,
		"details": per_item,
	}


def summarize_delta(base_summary: Dict[str, float], tuned_summary: Dict[str, float]) -> Dict[str, float]:
	keys = ["bleu2", "rouge_l_f1", "exact_match"]
	return {
		key: round(tuned_summary.get(key, 0.0) - base_summary.get(key, 0.0), 6)
		for key in keys
	}


def run_compare(base_file: Path, tuned_file: Path, report_file: Path) -> Dict[str, object]:
	base_rows = load_predictions(base_file)
	tuned_rows = load_predictions(tuned_file)

	if len(base_rows) != len(tuned_rows):
		raise ValueError("Base and fine-tuned output counts do not match")

	base_report = evaluate_predictions(base_rows)
	tuned_report = evaluate_predictions(tuned_rows)
	delta = summarize_delta(base_report["summary"], tuned_report["summary"])

	report = {
		"base": {
			"file": str(base_file),
			"count": base_report["count"],
			"summary": base_report["summary"],
		},
		"finetuned": {
			"file": str(tuned_file),
			"count": tuned_report["count"],
			"summary": tuned_report["summary"],
		},
		"delta_finetuned_minus_base": delta,
	}

	report_file.parent.mkdir(parents=True, exist_ok=True)
	with report_file.open("w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	return report


def main() -> None:
	paths = get_paths()
	parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model outputs")
	parser.add_argument(
		"--base-file",
		type=Path,
		default=paths.results_dir / "base_outputs" / "predictions.json",
	)
	parser.add_argument(
		"--finetuned-file",
		type=Path,
		default=paths.results_dir / "finetuned_outputs" / "predictions.json",
	)
	parser.add_argument(
		"--report-file",
		type=Path,
		default=paths.results_dir / "metrics" / "comparison_report.json",
	)
	args = parser.parse_args()

	report = run_compare(args.base_file, args.finetuned_file, args.report_file)

	print("Evaluation comparison complete")
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()

