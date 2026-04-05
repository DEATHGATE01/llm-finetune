from __future__ import annotations

import argparse
import json

from src.model.trainer import run_training
from src.utils.config import get_model_config, get_training_config


def main() -> None:
	parser = argparse.ArgumentParser(description="Phase 5 training entrypoint")
	parser.add_argument("--dry-run", action="store_true", help="Validate training config and dataset only")
	parser.add_argument("--max-train-samples", type=int, default=32)
	parser.add_argument("--max-eval-samples", type=int, default=16)
	args = parser.parse_args()

	model_cfg = get_model_config()
	train_cfg = get_training_config()

	if args.dry_run:
		summary = {
			"dry_run": True,
			"model_name": model_cfg.model_name,
			"train_file": str(train_cfg.train_file),
			"val_file": str(train_cfg.val_file),
			"output_dir": str(train_cfg.output_dir),
			"batch_size": train_cfg.per_device_train_batch_size,
			"gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
			"num_train_epochs": train_cfg.num_train_epochs,
			"learning_rate": train_cfg.learning_rate,
			"max_train_samples": args.max_train_samples,
			"max_eval_samples": args.max_eval_samples,
		}
		print("Phase 5 training smoke test complete")
		print(json.dumps(summary, indent=2))
		return

	try:
		stats = run_training(max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples)
	except RuntimeError as exc:
		print(f"Training aborted: {exc}")
		raise SystemExit(2) from exc

	print("Phase 5 training complete")
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()

