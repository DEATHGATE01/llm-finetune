from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.inference import mock_generate
from src.model.load_model import load_model, load_tokenizer
from src.utils.config import get_model_config, get_paths


def generate_with_model(model_name: str, prompt: str, max_new_tokens: int) -> str:
	import torch

	tokenizer = load_tokenizer(model_name)
	model = load_model(
		model_name=model_name,
		quant_config=None,
		trust_remote_code=False,
		device_map="auto",
	)
	encoded = tokenizer(prompt, return_tensors="pt")
	encoded = {k: v.to(model.device) for k, v in encoded.items()}

	with torch.no_grad():
		generated = model.generate(
			**encoded,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=tokenizer.eos_token_id,
		)

	decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
	return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


def run_demo(prompt: str, base_model: str, finetuned_model: str, use_mock: bool, max_new_tokens: int) -> dict:
	if use_mock:
		base_response = mock_generate(prompt, mode="base")
		finetuned_response = mock_generate(prompt, mode="finetuned")
	else:
		base_response = generate_with_model(base_model, prompt, max_new_tokens=max_new_tokens)
		finetuned_response = generate_with_model(finetuned_model, prompt, max_new_tokens=max_new_tokens)

	return {
		"prompt": prompt,
		"base_model": base_model,
		"finetuned_model": finetuned_model,
		"base_response": base_response,
		"finetuned_response": finetuned_response,
	}


def main() -> None:
	paths = get_paths()
	model_cfg = get_model_config()

	parser = argparse.ArgumentParser(description="Phase 7 demo: base vs fine-tuned response")
	parser.add_argument("--prompt", type=str, default="Explain hashing in simple terms.")
	parser.add_argument("--base-model", type=str, default=model_cfg.model_name)
	parser.add_argument(
		"--finetuned-model",
		type=str,
		default=str(paths.results_dir / "finetuned_model"),
	)
	parser.add_argument("--mock", action="store_true")
	parser.add_argument("--max-new-tokens", type=int, default=140)
	parser.add_argument("--save", type=Path, default=None, help="Optional JSON output path")
	args = parser.parse_args()

	result = run_demo(
		prompt=args.prompt,
		base_model=args.base_model,
		finetuned_model=args.finetuned_model,
		use_mock=args.mock,
		max_new_tokens=args.max_new_tokens,
	)

	print("Demo output")
	print(json.dumps(result, ensure_ascii=False, indent=2))

	if args.save is not None:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		with args.save.open("w", encoding="utf-8") as f:
			json.dump(result, f, ensure_ascii=False, indent=2)
		print(f"Saved demo output to: {args.save}")


if __name__ == "__main__":
	main()

