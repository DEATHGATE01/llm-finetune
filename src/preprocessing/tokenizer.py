from __future__ import annotations

import argparse
import json
from typing import Dict, List

from transformers import AutoTokenizer

from src.utils.config import get_model_config


def load_tokenizer(model_name: str):
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	return tokenizer


def tokenize_batch(
	tokenizer,
	texts: List[str],
	max_length: int,
) -> Dict[str, List[List[int]]]:
	return tokenizer(
		texts,
		truncation=True,
		max_length=max_length,
		padding=True,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Tokenizer smoke test")
	parser.add_argument("--model-name", type=str, default=None, help="Override model name")
	parser.add_argument("--text", type=str, default="Explain REST API in simple terms.")
	args = parser.parse_args()

	config = get_model_config()
	model_name = args.model_name or config.model_name
	tokenizer = load_tokenizer(model_name)
	encoded = tokenize_batch(tokenizer, [args.text], max_length=config.max_length)

	summary = {
		"model_name": model_name,
		"pad_token": tokenizer.pad_token,
		"eos_token": tokenizer.eos_token,
		"input_length": len(encoded["input_ids"][0]),
		"max_length": config.max_length,
	}
	print("Tokenizer smoke test complete")
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
	main()

