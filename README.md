# Fine-Tune LLM

This project builds an instruction-following assistant using a custom synthetic dataset and LoRA / QLoRA fine-tuning.

## What is included

- Dataset generation, cleaning, and formatting pipeline
- Training pipeline with LoRA / QLoRA support
- Evaluation and comparison tooling
- Demo app for base vs fine-tuned outputs
- Colab-ready notebook for running the real Mistral training path

## Colab notebook

Use [notebooks/colab_training.ipynb](notebooks/colab_training.ipynb) when you want to run the full Mistral + QLoRA workflow on a GPU runtime.

### Colab flow

1. Open the notebook in Colab.
2. Switch runtime to GPU.
3. Install dependencies.
4. Mount Google Drive and point `PROJECT_ROOT` to your repository folder.
5. Run the training cell with `USE_4BIT=true`.
6. Generate base and fine-tuned predictions.
7. Compare metrics and save outputs to Drive.

### Recommended Colab settings

- `USE_4BIT=true`
- `MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2`
- `MAX_LENGTH=1024`

## Local validation on this machine

Your current laptop is good for smoke tests and small-model validation, not full Mistral training.

Use this locally when you want a quick proof of life:

```powershell
$env:USE_4BIT="false"
$env:MODEL_NAME="sshleifer/tiny-gpt2"
python.exe -m src.training.train --max-train-samples 16 --max-eval-samples 8
```

## Real training on Colab

Run this from the repository root inside Colab:

```bash
export USE_4BIT=true
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
export MAX_LENGTH=1024
python -m src.training.train --max-train-samples 128 --max-eval-samples 32
```

## Output locations

- Clean dataset: [data/final_dataset.json](data/final_dataset.json)
- Processed train/val files: [data/processed](data/processed)
- Fine-tuned model: [results/finetuned_model](results/finetuned_model)
- Comparison report: [results/metrics/comparison_report.json](results/metrics/comparison_report.json)

## Notes

- The notebook and code are set up to keep code human-written and easy to follow.
- If Colab memory is tight, keep `MAX_LENGTH` at 1024 and use 4-bit loading.
- For local machines without a CUDA GPU, use the tiny-model smoke path only.
