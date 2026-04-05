from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

from peft import LoraConfig, TaskType


@dataclass(frozen=True)
class LoraRuntimeConfig:
	r: int
	lora_alpha: int
	target_modules: List[str]
	lora_dropout: float
	bias: str
	task_type: str


def get_lora_runtime_config() -> LoraRuntimeConfig:
	target_modules = os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj")
	modules = [m.strip() for m in target_modules.split(",") if m.strip()]

	return LoraRuntimeConfig(
		r=int(os.getenv("LORA_R", "16")),
		lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
		target_modules=modules,
		lora_dropout=float(os.getenv("LORA_DROPOUT", "0.1")),
		bias=os.getenv("LORA_BIAS", "none"),
		task_type=os.getenv("LORA_TASK_TYPE", "CAUSAL_LM"),
	)


def build_lora_config(runtime: LoraRuntimeConfig) -> LoraConfig:
	task_type = TaskType.CAUSAL_LM if runtime.task_type == "CAUSAL_LM" else TaskType.CAUSAL_LM
	return LoraConfig(
		r=runtime.r,
		lora_alpha=runtime.lora_alpha,
		target_modules=runtime.target_modules,
		lora_dropout=runtime.lora_dropout,
		bias=runtime.bias,
		task_type=task_type,
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Phase 4 LoRA config smoke test")
	parser.add_argument("--print-json", action="store_true", help="Print LoRA config details")
	args = parser.parse_args()

	runtime = get_lora_runtime_config()
	lora_cfg = build_lora_config(runtime)

	summary = {
		"r": lora_cfg.r,
		"lora_alpha": lora_cfg.lora_alpha,
		"target_modules": sorted(list(lora_cfg.target_modules or [])),
		"lora_dropout": lora_cfg.lora_dropout,
		"bias": lora_cfg.bias,
		"task_type": str(lora_cfg.task_type),
	}

	print("Phase 4 LoRA config smoke test complete")
	if args.print_json:
		print(json.dumps(summary, indent=2))
	else:
		print(summary)


if __name__ == "__main__":
	main()

