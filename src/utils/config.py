from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
	root: Path
	data_final: Path
	data_processed: Path
	results_dir: Path


@dataclass(frozen=True)
class ModelConfig:
	model_name: str
	trust_remote_code: bool
	use_4bit: bool
	bnb_4bit_compute_dtype: str
	bnb_4bit_quant_type: str
	bnb_4bit_use_double_quant: bool
	device_map: str
	max_length: int


@dataclass(frozen=True)
class TrainingConfig:
	train_file: Path
	val_file: Path
	output_dir: Path
	logging_dir: Path
	per_device_train_batch_size: int
	per_device_eval_batch_size: int
	gradient_accumulation_steps: int
	num_train_epochs: float
	learning_rate: float
	weight_decay: float
	logging_steps: int
	eval_steps: int
	save_steps: int
	warmup_ratio: float


def project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def get_paths() -> ProjectPaths:
	root = project_root()
	return ProjectPaths(
		root=root,
		data_final=root / "data" / "final_dataset.json",
		data_processed=root / "data" / "processed",
		results_dir=root / "results",
	)


def get_model_config() -> ModelConfig:
	return ModelConfig(
		model_name=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
		trust_remote_code=os.getenv("TRUST_REMOTE_CODE", "false").lower() == "true",
		use_4bit=os.getenv("USE_4BIT", "true").lower() == "true",
		bnb_4bit_compute_dtype=os.getenv("BNB_4BIT_COMPUTE_DTYPE", "float16"),
		bnb_4bit_quant_type=os.getenv("BNB_4BIT_QUANT_TYPE", "nf4"),
		bnb_4bit_use_double_quant=os.getenv("BNB_4BIT_USE_DOUBLE_QUANT", "true").lower() == "true",
		device_map=os.getenv("DEVICE_MAP", "auto"),
		max_length=int(os.getenv("MAX_LENGTH", "1024")),
	)


def get_training_config() -> TrainingConfig:
	paths = get_paths()
	output_dir = Path(os.getenv("TRAIN_OUTPUT_DIR", str(paths.results_dir / "finetuned_model")))
	logging_dir = Path(os.getenv("TRAIN_LOGGING_DIR", str(paths.results_dir / "logs")))

	return TrainingConfig(
		train_file=Path(os.getenv("TRAIN_FILE", str(paths.data_processed / "train_instruct.jsonl"))),
		val_file=Path(os.getenv("VAL_FILE", str(paths.data_processed / "val_instruct.jsonl"))),
		output_dir=output_dir,
		logging_dir=logging_dir,
		per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "2")),
		per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "2")),
		gradient_accumulation_steps=int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4")),
		num_train_epochs=float(os.getenv("NUM_TRAIN_EPOCHS", "3")),
		learning_rate=float(os.getenv("LEARNING_RATE", "2e-4")),
		weight_decay=float(os.getenv("WEIGHT_DECAY", "0.0")),
		logging_steps=int(os.getenv("LOGGING_STEPS", "10")),
		eval_steps=int(os.getenv("EVAL_STEPS", "50")),
		save_steps=int(os.getenv("SAVE_STEPS", "100")),
		warmup_ratio=float(os.getenv("WARMUP_RATIO", "0.03")),
	)

