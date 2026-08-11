# Qwen 2.5 Fine-Tuning Preparation

This folder prepares the local pipeline for future Qwen 2.5 fine-tuning. It does **not** launch training.

## Structure

- `configs/`
  - `training_config.yaml`: main training configuration
  - `lora_config.yaml`: LoRA preset
  - `qlora_config.yaml`: QLoRA preset
  - `inference_config.yaml`: local inference defaults
- `datasets/`
  - `train.jsonl`
  - `validation.jsonl`
  - `test.jsonl`
- `models/`: local model or adapter directories live here later
- `prompts/system_prompt.txt`: reusable assistant system prompt
- `scripts/`
  - `validate_dataset.py`
  - `split_dataset.py`
  - `prepare_training.py`
  - `inference.py`
  - `benchmark.py`
- `requirements.txt`

## Dependencies

Install the preparation-time dependencies before using the inference or benchmark utilities:

- `torch`
- `transformers`
- `peft`
- `pyyaml`
- `safetensors`

Optional, if you later add training execution locally:

- `accelerate`
- `datasets`

## Dataset format

The pipeline expects JSONL rows shaped like this:

```json
{
  "id": "conv_123",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "store": "root4pro",
    "topic": "shipping",
    "resolution": "resolved",
    "language": "fr"
  }
}
```

## Validate a dataset

Validate structure, malformed rows, language distribution, topic distribution, and average conversation length:

```powershell
python training/scripts/validate_dataset.py --input training/datasets/train.jsonl --strict
```

The validator checks:

- valid JSONL
- required fields
- valid message blocks
- metadata fields `store`, `topic`, and `resolution`
- counts and distributions
- average conversation length

## Split a dataset

Split a validated dataset into 80/10/10 with a reproducible seed:

```powershell
python training/scripts/split_dataset.py --input training/datasets/training_dataset.jsonl --output-dir training/datasets --seed 42
```

This writes:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `split_summary.json`

## Review-first preparation

If you are starting from a manual review CSV, use `prepare_training.py`:

```powershell
python training/scripts/prepare_training.py --review-csv path\to\fine_tuning_review.csv --output-dir training/datasets
```

The script will:

- compile approved rows into `training_dataset.jsonl`
- validate the compiled JSONL
- split it into train/validation/test
- write `training_preparation_summary.json`

## Training configuration

Training is prepared but not executed. The default configuration lives in `configs/training_config.yaml`.

It contains:

- base model path
- learning rate
- epochs
- batch size
- gradient accumulation steps
- max sequence length
- LoRA parameters
- output directory
- logging configuration

When you later move to a GPU server, the training launcher should simply read this config and start the run.

## System prompt

The reusable assistant prompt is in `prompts/system_prompt.txt`. It instructs the model to:

- answer in the user's language
- understand Tunisian Darija
- stay polite and professional
- follow ROOT4PRO policies
- never invent product information
- rely on RAG for dynamic data

## Inference

Interactive inference works on CPU and loads only local files by default.

```powershell
python training/scripts/inference.py --model-path .\models\qwen2.5-base
```

To load a LoRA adapter later:

```powershell
python training/scripts/inference.py --model-path .\models\qwen2.5-base --adapter-path .\models\qwen2.5-lora
```

If the adapter path is omitted, the script loads the base model only.

The script does not download models automatically. It fails fast if the paths do not exist.

## Benchmarking

Compare a base model against a fine-tuned model after you have both local paths available:

```powershell
python training/scripts/benchmark.py --dataset training/datasets/test.jsonl --base-model-path .\models\qwen2.5-base --finetuned-model-path .\models\qwen2.5-finetuned --output-dir training/benchmark
```

The benchmark writes JSON and CSV outputs that include:

- average response length
- latency
- language consistency
- policy compliance
- side-by-side responses for qualitative review

## Later training launch

Once the GPU server is ready, training should be launched externally using the validated split files and the YAML config. This repository only prepares the data, prompts, configs, and inference/benchmark tooling.
