from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from peft import PeftConfig, PeftModel

from src.model.load_model import build_quant_config, load_model, load_tokenizer
from src.utils.config import get_model_config, get_paths


Record = Dict[str, str]


def load_eval_samples(dataset_path: Path, max_samples: int) -> List[Record]:
	with dataset_path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, list):
		raise ValueError("Expected evaluation dataset to be a JSON list")
	return data[:max_samples]


def build_prompt(item: Record) -> str:
	instruction = item["instruction"].strip()
	input_text = item.get("input", "").strip()
	if input_text:
		return f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
	return f"Instruction: {instruction}\nResponse:"


def mock_generate(base_text: str, mode: str) -> str:
	if mode == "base":
		return (
			"This is a generic baseline response. "
			"It explains the topic briefly but may miss domain-specific depth. "
			f"Prompt summary: {base_text[:140]}"
		)
	return (
		"This is a fine-tuned style response with clearer structure and stronger technical precision. "
		"It adds practical framing and keeps the explanation aligned to the instruction. "
		f"Prompt summary: {base_text[:140]}"
	)


def run_generation(
	samples: List[Record],
	model_name: str,
	output_path: Path,
	max_new_tokens: int = 160,
	use_mock: bool = False,
	mock_mode: str = "base",
) -> List[Dict[str, str]]:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	outputs: List[Dict[str, str]] = []

	if use_mock:
		for item in samples:
			prompt = build_prompt(item)
			generated = mock_generate(prompt, mode=mock_mode)
			outputs.append(
				{
					"instruction": item["instruction"],
					"input": item.get("input", ""),
					"reference": item["output"],
					"prediction": generated,
				}
			)
	else:
		import torch

		model_cfg = get_model_config()
		# Use configured quantization path to avoid CPU/disk offload stalls on Colab.
		quant_cfg = build_quant_config(model_cfg)
		model_path = Path(model_name)

		if model_path.exists() and (model_path / "adapter_config.json").exists():
			peft_cfg = PeftConfig.from_pretrained(model_name)
			base_name = peft_cfg.base_model_name_or_path
			tokenizer = load_tokenizer(base_name)
			base_model = load_model(
				model_name=base_name,
				quant_config=quant_cfg,
				trust_remote_code=model_cfg.trust_remote_code,
				device_map=model_cfg.device_map,
			)
			model = PeftModel.from_pretrained(base_model, model_name)
		else:
			tokenizer = load_tokenizer(model_name)
			model = load_model(
				model_name=model_name,
				quant_config=quant_cfg,
				trust_remote_code=model_cfg.trust_remote_code,
				device_map=model_cfg.device_map,
			)

		model.eval()
		model_device = next(model.parameters()).device

		for item in samples:
			prompt = build_prompt(item)
			encoded = tokenizer(prompt, return_tensors="pt")
			encoded = {k: v.to(model_device) for k, v in encoded.items()}
			with torch.no_grad():
				generated_ids = model.generate(
					**encoded,
					max_new_tokens=max_new_tokens,
					do_sample=False,
					pad_token_id=tokenizer.eos_token_id,
				)
			decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
			prediction = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
			outputs.append(
				{
					"instruction": item["instruction"],
					"input": item.get("input", ""),
					"reference": item["output"],
					"prediction": prediction,
				}
			)

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(outputs, f, ensure_ascii=False, indent=2)

	return outputs


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate evaluation outputs")
	parser.add_argument("--dataset", type=Path, default=None)
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--model-name", type=str, default=None)
	parser.add_argument("--max-samples", type=int, default=20)
	parser.add_argument("--max-new-tokens", type=int, default=160)
	parser.add_argument("--mock", action="store_true")
	parser.add_argument("--mock-mode", choices=["base", "finetuned"], default="base")
	args = parser.parse_args()

	model_cfg = get_model_config()
	paths = get_paths()
	dataset_path = args.dataset or paths.data_final
	model_name = args.model_name or model_cfg.model_name

	samples = load_eval_samples(dataset_path, args.max_samples)
	outputs = run_generation(
		samples=samples,
		model_name=model_name,
		output_path=args.output,
		max_new_tokens=args.max_new_tokens,
		use_mock=args.mock,
		mock_mode=args.mock_mode,
	)

	print("Evaluation inference complete")
	print(
		json.dumps(
			{
				"dataset": str(dataset_path),
				"output": str(args.output),
				"model_name": model_name,
				"mock": args.mock,
				"records": len(outputs),
			},
			indent=2,
		)
	)


if __name__ == "__main__":
	main()

