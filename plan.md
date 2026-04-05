Generative AI & LLMs
Assignment-2: Fine-Tuning a Large Language Model Using Custom Dataset
Objective: The objective of this assignment is to understand how βine-tuning improves the

performance of Large Language Models for specialized tasks by adapting them to domain-
speciβic datasets.

Problem Statement: Pretrained LLMs are trained on large general-purpose datasets and
may not perform well on domain-speciβic tasks. Fine-tuning allows models to learn
specialized behaviour by training them on custom datasets.
Create or curate a dataset and βine-tune an open-source LLM for a speciβic task.
• Domain-speciβic question answering
• Instruction-following assistant
• Code generation assistant
• Domain-speciβic summarization
• Customer support chatbot
• Structured data to text generation
Dataset Requirements: Construct or collect a dataset containing at least 500–2000
examples. The dataset must contain an input prompt and expected output.
Possible dataset sources:
 Synthetic dataset generated using LLMs
 Public datasets
 Custom curated datasets
Dataset must include:
 Input prompt
 Expected output
Model Options: Fine-tune models such as:
 LLaMA-based models
 Mistral
 Gemma
 GPT-style open models
 Other open-source LLMs
Recommended Approaches:
 LoRA / QLoRA
 PEFT βine-tuning
 HuggingFace Transformers
Deliverables
• Complete source code including dataset preprocessing and training pipeline
• Dataset used for βine-tuning with documentation
• Experimental report (8–10 pages) explaining methodology and results
Report must include - Problem deβinition, Dataset creation methodology, Model
architecture and βine-tuning method, Training conβiguration, Evaluation results
• Comparison between base model and βine-tuned model - metrics may include - Accuracy,
BLEU / ROUGE, Human evaluation, Task-speciβic metrics
• Example outputs demonstrating performance improvement (screenshots, etc.)

_________________________________________________________________________________________________________________

# 🔷 1. Project Direction (Pick This Carefully)

## ✅ Recommended: **Instruction-Following Technical Assistant (Best Choice)**

**Why this wins:**

* You can **generate dataset synthetically** (fast + scalable)
* Easy to evaluate (instruction → response correctness)
* Works well with LoRA/QLoRA
* Strong viva explanation (alignment, instruction tuning)

### Example Task:

> Input: “Explain REST API in simple terms”
> Output: “A REST API is…”

---

## 🔥 Alternative (If you want edge):

**Code Assistant (Python-focused)**

* Input: Problem description
* Output: Code + explanation
  ⚠️ Slightly harder to evaluate properly

---

# 🔷 2. System Architecture

```text
Dataset (Prompt → Response)
        ↓
Tokenization
        ↓
Base LLM (Frozen)
        ↓
LoRA Adapters (Trainable)
        ↓
Fine-tuned Model
        ↓
Evaluation (Base vs Tuned)
```

---

# 🔷 3. Tech Stack (Deliberate Choices)

### Model (pick 1 primary + 1 optional comparison)

* **Mistral 7B** (best balance)
* OR **Gemma 2B/7B** (lighter, easier to train)

### Fine-tuning Method

* **QLoRA (must use)** → memory efficient + expected in assignment
* PEFT (HuggingFace)

### Libraries

* `transformers`
* `peft`
* `datasets`
* `bitsandbytes`
* `trl` (for SFTTrainer)

---

# 🔷 4. Directory Structure (Clean + Modular)

```bash
fine-tune-llm/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── final_dataset.json
│
├── src/
│   ├── dataset/
│   │   ├── generator.py        # synthetic generation
│   │   ├── cleaner.py
│   │   └── formatter.py
│   │
│   ├── preprocessing/
│   │   └── tokenizer.py
│   │
│   ├── model/
│   │   ├── load_model.py
│   │   ├── lora_config.py
│   │   └── trainer.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── compare.py
│   │   └── inference.py
│   │
│   └── utils/
│       └── config.py
│
├── notebooks/
│   └── dataset_generation.ipynb
│
├── app/
│   └── demo.py
│
├── results/
│   ├── base_outputs/
│   ├── finetuned_outputs/
│   └── metrics/
│
├── report/
│   └── report.pdf
│
├── requirements.txt
└── README.md
```

---

# 🔷 5. Phase-Wise Plan (Execution Strategy)

## ✅ Phase 1 — Dataset Creation (MOST IMPORTANT)

Target: **1000 examples (safe middle)**

### Structure:

```json
{
  "instruction": "Explain binary search",
  "input": "",
  "output": "Binary search is..."
}
```

### Methods:

1. **Synthetic generation (recommended)**

   * Use GPT or open LLM
   * Prompt:

     ```
     Generate 100 instruction-response pairs for computer science topics.
     ```
2. Mix categories:

   * Programming
   * APIs
   * DBMS
   * OS basics

⚠️ Avoid low-quality repetitive data — evaluators check this.

---

## ✅ Phase 2 — Data Formatting

Convert to training format:

```text
<|instruction|>
Explain REST API

<|response|>
A REST API is...
```

OR chat format (better for modern LLMs):

```json
{
  "messages": [
    {"role": "user", "content": "Explain REST API"},
    {"role": "assistant", "content": "A REST API is..."}
  ]
}
```

---

## ✅ Phase 3 — Model Setup

Load model in **4-bit (QLoRA)**:

* Reduces VRAM usage
* Required for laptops/Colab

---

## ✅ Phase 4 — LoRA Configuration

```python
r=16
lora_alpha=32
target_modules=["q_proj", "v_proj"]
dropout=0.1
```

---

## ✅ Phase 5 — Training

Key parameters:

```python
batch_size = 2
gradient_accumulation = 4
epochs = 3
learning_rate = 2e-4
```

Trainer:

* `SFTTrainer` (TRL)

---

## ✅ Phase 6 — Evaluation (Critical for Marks)

### Compare:

**Base Model vs Fine-tuned Model**

Metrics:

| Metric        | Use                     |
| ------------- | ----------------------- |
| BLEU          | text similarity         |
| ROUGE         | summarization quality   |
| Human Eval    | best scoring factor     |
| Task accuracy | instruction correctness |

---

## ✅ Phase 7 — Demo

Simple CLI or Streamlit:

Input:

> “Explain hashing”

Output:

* Base model response
* Fine-tuned response

---

# 🔷 6. Required Comparison (MANDATORY)

### Config A (Baseline)

* Base Mistral (no fine-tuning)

### Config B (Fine-tuned)

* Mistral + QLoRA + custom dataset

---

# 🔷 7. What Evaluators Will Test in Viva

Be ready for:

### ❓ Why QLoRA?

* memory efficient (4-bit quantization)
* enables training large models on small GPUs

### ❓ Why LoRA?

* reduces trainable params
* faster + cheaper

### ❓ Why your dataset is good?

* diverse topics
* clean formatting
* non-redundant

---

# 🔷 8. Report Structure (Strict)

1. Problem Definition
2. Dataset Creation Methodology
3. Model + Fine-tuning Method (LoRA/QLoRA)
4. Training Setup
5. Results (Base vs Tuned)
6. Observations
7. Conclusion

---

# 🔷 9. Critical Mistakes to Avoid

* ❌ Using <200 samples → weak
* ❌ No comparison → low marks
* ❌ No explanation of LoRA → viva fail
* ❌ No dataset justification → major deduction
