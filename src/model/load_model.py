from __future__ import annotations

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.config import get_model_config


DTYPE_MAP = {
	"float16": torch.float16,
	"bfloat16": torch.bfloat16,
	"float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
	if name not in DTYPE_MAP:
		raise ValueError(f"Unsupported dtype: {name}. Use one of {list(DTYPE_MAP)}")
	return DTYPE_MAP[name]


def build_quant_config(config) -> Optional[BitsAndBytesConfig]:
	if not config.use_4bit:
		return None

	allow_fallback = os.getenv("ALLOW_4BIT_FALLBACK", "true").lower() == "true"

	try:
		version("bitsandbytes")
	except PackageNotFoundError as exc:
		if allow_fallback:
			print(
				"Warning: bitsandbytes not found. Falling back to non-4bit loading. "
				"Set USE_4BIT=false to silence this warning."
			)
			return None
		raise RuntimeError(
			"bitsandbytes is required for 4-bit loading. Install it or set USE_4BIT=false."
		) from exc

	return BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_quant_type=config.bnb_4bit_quant_type,
		bnb_4bit_compute_dtype=resolve_dtype(config.bnb_4bit_compute_dtype),
		bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
	)


def load_tokenizer(model_name: str):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "right"
	return tokenizer


def load_model(
	model_name: str,
	quant_config: Optional[BitsAndBytesConfig],
	trust_remote_code: bool,
	device_map: Optional[str],
):
	kwargs = {
		"trust_remote_code": trust_remote_code,
	}
	if quant_config is not None:
		kwargs["quantization_config"] = quant_config
	if device_map is not None:
		kwargs["device_map"] = device_map

	return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def build_summary(config, dry_run: bool) -> dict:
	return {
		"model_name": config.model_name,
		"use_4bit": config.use_4bit,
		"bnb_4bit_quant_type": config.bnb_4bit_quant_type,
		"bnb_4bit_compute_dtype": config.bnb_4bit_compute_dtype,
		"bnb_4bit_use_double_quant": config.bnb_4bit_use_double_quant,
		"device_map": config.device_map,
		"dry_run": dry_run,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Phase 3 model loader")
	parser.add_argument("--model-name", type=str, default=None, help="Override model name")
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Validate configuration only, skip quantization and model loading",
	)
	parser.add_argument(
		"--check-tokenizer",
		action="store_true",
		help="When used with --dry-run, also try tokenizer download/load",
	)
	args = parser.parse_args()

	config = get_model_config()
	if args.model_name:
		config = type(config)(
			model_name=args.model_name,
			trust_remote_code=config.trust_remote_code,
			use_4bit=config.use_4bit,
			bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
			bnb_4bit_quant_type=config.bnb_4bit_quant_type,
			bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
			device_map=config.device_map,
			max_length=config.max_length,
		)

	if args.dry_run:
		summary = build_summary(config, dry_run=True)

		if args.check_tokenizer:
			tokenizer = load_tokenizer(config.model_name)
			summary["tokenizer_checked"] = True
			summary["tokenizer_pad_token"] = tokenizer.pad_token
			summary["tokenizer_eos_token"] = tokenizer.eos_token
		else:
			summary["tokenizer_checked"] = False

		print("Phase 3 model setup smoke test complete")
		print(json.dumps(summary, indent=2))
		return

	quant_config = build_quant_config(config)
	tokenizer = load_tokenizer(config.model_name)

	model = load_model(
		model_name=config.model_name,
		quant_config=quant_config,
		trust_remote_code=config.trust_remote_code,
		device_map=config.device_map,
	)

	summary = build_summary(config, dry_run=False)
	summary["num_parameters"] = int(sum(p.numel() for p in model.parameters()))
	summary["tokenizer_pad_token"] = tokenizer.pad_token
	summary["tokenizer_eos_token"] = tokenizer.eos_token
	print("Phase 3 model setup complete")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()

