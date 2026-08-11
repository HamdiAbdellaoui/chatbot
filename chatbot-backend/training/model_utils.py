"""Model loading and generation helpers for local Qwen 2.5 inference."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBundle:
    tokenizer: Any
    model: Any
    model_path: str
    adapter_path: str | None
    device: str


def resolve_local_model_path(path_value: str | None, *, env_name: str, default_value: str | None = None) -> str:
    candidate = (path_value or os.getenv(env_name) or default_value or "").strip()
    if not candidate:
        raise ValueError(f"Missing model path. Provide --model-path or set {env_name}.")
    if Path(candidate).exists():
        return candidate
    raise FileNotFoundError(
        f"Model path not found: {candidate}. This pipeline does not download large models automatically. "
        f"Place the model locally or pass an existing directory."
    )


def _import_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing inference dependencies. Install torch, transformers, and peft before using inference/benchmark scripts."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def load_model_bundle(*, model_path: str, adapter_path: str | None = None, trust_remote_code: bool = True) -> ModelBundle:
    torch, AutoModelForCausalLM, AutoTokenizer = _import_transformers()
    resolved_model = resolve_local_model_path(model_path, env_name="QWEN_BASE_MODEL_PATH")
    resolved_adapter = adapter_path.strip() if isinstance(adapter_path, str) and adapter_path.strip() else None
    if resolved_adapter and not Path(resolved_adapter).exists():
        raise FileNotFoundError(f"Adapter path not found: {resolved_adapter}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model, local_files_only=True, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float32,
        device_map=None,
    )

    if resolved_adapter:
        try:
            from peft import PeftModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("peft is required to load LoRA adapters") from exc
        model = PeftModel.from_pretrained(model, resolved_adapter)

    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model, model_path=resolved_model, adapter_path=resolved_adapter, device="cpu")


def load_system_prompt(prompt_path: str | None) -> str:
    if prompt_path and Path(prompt_path).exists():
        return Path(prompt_path).read_text(encoding="utf-8").strip()
    return (
        "You are ROOT4PRO's customer support assistant. Respond in the user's language, "
        "understand Tunisian Darija, stay polite and professional, follow ROOT4PRO policies, "
        "never invent product information, and rely on RAG for dynamic business details."
    )


def build_chat_messages(*, user_text: str, system_prompt: str, history: Sequence[dict[str, str]] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend([{"role": item["role"], "content": item["content"]} for item in history])
    messages.append({"role": "user", "content": user_text})
    return messages


def render_prompt(tokenizer: Any, messages: Sequence[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    lines: list[str] = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


def generate_text(
    bundle: ModelBundle,
    *,
    messages: Sequence[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    torch, _, _ = _import_transformers()
    prompt = render_prompt(bundle.tokenizer, messages)
    inputs = bundle.tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(bundle.device) for key, value in inputs.items()}

    do_sample = temperature > 0.0
    with torch.inference_mode():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=bundle.tokenizer.eos_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
