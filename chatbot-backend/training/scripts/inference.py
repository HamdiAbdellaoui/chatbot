"""Interactive local inference for Qwen 2.5 with optional LoRA adapters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TRAINING_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.model_utils import build_chat_messages, generate_text, load_model_bundle, load_system_prompt, resolve_local_model_path


def _interactive_loop(bundle, *, system_prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> None:
    history: list[dict[str, str]] = []
    print("Interactive inference ready. Type /exit to quit, /reset to clear history.\n")
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/reset":
            history.clear()
            print("History cleared.\n")
            continue

        messages = build_chat_messages(user_text=user_text, system_prompt=system_prompt, history=history)
        reply = generate_text(bundle, messages=messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        print(f"Assistant> {reply}\n")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive local inference for Qwen 2.5.")
    parser.add_argument("--model-path", default=None, help="Local path to the base model directory")
    parser.add_argument("--adapter-path", default=None, help="Optional local path to a LoRA adapter directory")
    parser.add_argument("--system-prompt-file", default=str(TRAINING_ROOT / "prompts" / "system_prompt.txt"), help="System prompt file")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", default=None, help="Optional one-shot prompt instead of interactive mode")
    args = parser.parse_args()

    if not args.model_path:
        raise SystemExit("Missing --model-path (or set QWEN_BASE_MODEL_PATH). This script never downloads large models automatically.")

    model_path = resolve_local_model_path(args.model_path, env_name="QWEN_BASE_MODEL_PATH")
    bundle = load_model_bundle(model_path=model_path, adapter_path=args.adapter_path)
    system_prompt = load_system_prompt(args.system_prompt_file)

    if args.prompt is not None:
        messages = build_chat_messages(user_text=args.prompt, system_prompt=system_prompt, history=[])
        reply = generate_text(bundle, messages=messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        print(reply)
        return 0

    _interactive_loop(bundle, system_prompt=system_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
