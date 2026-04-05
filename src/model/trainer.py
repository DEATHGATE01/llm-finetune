from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from peft import get_peft_model
import torch
from transformers import TrainingArguments

from src.model.load_model import build_quant_config, load_model
from src.model.lora_config import build_lora_config, get_lora_runtime_config
from src.preprocessing.tokenizer import load_tokenizer
from src.utils.config import get_model_config, get_training_config


def attach_lora_adapters(model, lora_config):
	"""Wrap base model with PEFT LoRA adapters."""
	requested = sorted(list(lora_config.target_modules or []))
	available_module_names = [name for name, _ in model.named_modules()]

	# First attempt uses user or default target modules.
	try:
		return get_peft_model(model, lora_config)
	except ValueError as exc:
		# Fallback for models like GPT-2 where attention projection names differ.
		fallback_candidates = ["c_attn", "query_key_value", "Wqkv", "qkv_proj"]
		fallback_targets = [m for m in fallback_candidates if any(n.endswith(m) for n in available_module_names)]

		if not fallback_targets:
			raise ValueError(
				"LoRA target modules not found. "
				f"Requested: {requested}. "
				"No known fallback targets were detected in this model."
			) from exc

		lora_config.target_modules = fallback_targets
		print(
			"Warning: Falling back to LoRA target modules "
			f"{fallback_targets} because requested modules {requested} were not found."
		)
		return get_peft_model(model, lora_config)


def trainable_stats(model) -> dict:
	trainable = 0
	total = 0
	for param in model.parameters():
		total += param.numel()
		if param.requires_grad:
			trainable += param.numel()

	percent = (100.0 * trainable / total) if total else 0.0
	return {
		"trainable_params": int(trainable),
		"total_params": int(total),
		"trainable_percent": round(percent, 6),
	}


def load_instruction_jsonl(file_path: Path, max_samples: Optional[int] = None) -> Dataset:
	ds = Dataset.from_json(str(file_path))
	if max_samples is not None and max_samples > 0:
		ds = ds.select(range(min(max_samples, len(ds))))
	return ds


def validate_training_environment(model_cfg) -> None:
	"""Fail fast when the requested model/runtime combination is not realistic locally."""
	model_name = model_cfg.model_name.lower()
	if "mistral" in model_name and not model_cfg.use_4bit and not torch.cuda.is_available():
		raise RuntimeError(
			"Local CPU training with Mistral 7B in full precision is not practical in this environment. "
			"Use USE_4BIT=true on a supported CUDA environment, or switch MODEL_NAME to a smaller model "
			"for local validation (for example sshleifer/tiny-gpt2)."
		)


def build_training_arguments(training_cfg, output_dir: Optional[Path] = None) -> TrainingArguments:
	resolved_output = output_dir or training_cfg.output_dir
	resolved_output.mkdir(parents=True, exist_ok=True)
	training_cfg.logging_dir.mkdir(parents=True, exist_ok=True)

	has_cuda = torch.cuda.is_available()
	use_fp16 = has_cuda
	use_bf16 = False
	if has_cuda:
		use_bf16 = torch.cuda.is_bf16_supported()
		if use_bf16:
			use_fp16 = False

	return TrainingArguments(
		output_dir=str(resolved_output),
		per_device_train_batch_size=training_cfg.per_device_train_batch_size,
		per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
		gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
		num_train_epochs=training_cfg.num_train_epochs,
		learning_rate=training_cfg.learning_rate,
		weight_decay=training_cfg.weight_decay,
		logging_steps=training_cfg.logging_steps,
		eval_steps=training_cfg.eval_steps,
		save_steps=training_cfg.save_steps,
		warmup_ratio=training_cfg.warmup_ratio,
		logging_dir=str(training_cfg.logging_dir),
		evaluation_strategy="steps",
		save_strategy="steps",
		report_to=[],
		fp16=use_fp16,
		bf16=use_bf16,
	)


def build_sft_trainer(model, tokenizer, train_dataset: Dataset, eval_dataset: Optional[Dataset], training_args: TrainingArguments):
	# Import lazily so dry-runs can work even before every training dependency is fully ready.
	from trl import SFTTrainer
	from src.utils.config import get_model_config

	model_cfg = get_model_config()
	requested_max = int(model_cfg.max_length)
	tok_max = getattr(tokenizer, "model_max_length", requested_max)
	if tok_max is None or tok_max <= 0 or tok_max > 1_000_000:
		safe_max_seq_length = requested_max
	else:
		safe_max_seq_length = min(int(tok_max), requested_max)

	return SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		args=training_args,
		dataset_text_field="text",
		max_seq_length=safe_max_seq_length,
	)


def run_training(max_train_samples: Optional[int] = None, max_eval_samples: Optional[int] = None) -> dict:
	model_cfg = get_model_config()
	training_cfg = get_training_config()
	lora_runtime = get_lora_runtime_config()
	lora_cfg = build_lora_config(lora_runtime)

	validate_training_environment(model_cfg)

	tokenizer = load_tokenizer(model_cfg.model_name)
	quant_cfg = build_quant_config(model_cfg)
	effective_device_map = model_cfg.device_map if quant_cfg is not None else None
	base_model = load_model(
		model_name=model_cfg.model_name,
		quant_config=quant_cfg,
		trust_remote_code=model_cfg.trust_remote_code,
		device_map=effective_device_map,
	)
	peft_model = attach_lora_adapters(base_model, lora_cfg)

	train_dataset = load_instruction_jsonl(training_cfg.train_file, max_samples=max_train_samples)
	eval_dataset = load_instruction_jsonl(training_cfg.val_file, max_samples=max_eval_samples)
	training_args = build_training_arguments(training_cfg)
	trainer = build_sft_trainer(
		model=peft_model,
		tokenizer=tokenizer,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		training_args=training_args,
	)

	train_result = trainer.train()
	trainer.save_model(str(training_cfg.output_dir))

	stats = trainable_stats(peft_model)
	stats.update(
		{
			"model_name": model_cfg.model_name,
			"train_samples": len(train_dataset),
			"eval_samples": len(eval_dataset),
			"train_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
			"train_loss": float(train_result.metrics.get("train_loss", 0.0)),
			"output_dir": str(training_cfg.output_dir),
		}
	)
	return stats


def main() -> None:
	parser = argparse.ArgumentParser(description="Model trainer helper")
	parser.add_argument("--dry-run", action="store_true", help="Skip base model load")
	parser.add_argument("--max-train-samples", type=int, default=16)
	parser.add_argument("--max-eval-samples", type=int, default=8)
	args = parser.parse_args()

	model_cfg = get_model_config()
	training_cfg = get_training_config()
	lora_runtime = get_lora_runtime_config()
	lora_cfg = build_lora_config(lora_runtime)

	if not args.dry_run:
		validate_training_environment(model_cfg)

	if args.dry_run:
		train_ds = load_instruction_jsonl(training_cfg.train_file, max_samples=args.max_train_samples)
		eval_ds = load_instruction_jsonl(training_cfg.val_file, max_samples=args.max_eval_samples)
		training_args = build_training_arguments(training_cfg)

		print("Trainer dry-run smoke test complete")
		print(
			json.dumps(
				{
					"dry_run": True,
					"model_name": model_cfg.model_name,
					"lora_r": lora_cfg.r,
					"lora_alpha": lora_cfg.lora_alpha,
					"lora_dropout": lora_cfg.lora_dropout,
					"target_modules": sorted(list(lora_cfg.target_modules or [])),
					"train_samples": len(train_ds),
					"eval_samples": len(eval_ds),
					"batch_size": training_args.per_device_train_batch_size,
					"gradient_accumulation_steps": training_args.gradient_accumulation_steps,
					"num_train_epochs": training_args.num_train_epochs,
					"learning_rate": training_args.learning_rate,
				},
				indent=2,
			)
		)
		return

	stats = run_training(max_train_samples=args.max_train_samples, max_eval_samples=args.max_eval_samples)
	stats["lora_r"] = lora_cfg.r
	stats["target_modules"] = sorted(list(lora_cfg.target_modules or []))

	print("Training run complete")
	print(json.dumps(stats, indent=2))


if __name__ == "__main__":
	main()

