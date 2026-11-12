"""
generate.py — write with a trained model
========================================

    python generate.py --model out/model.pt                        # type the start of a text, it continues
    python generate.py --model out/model.pt --prompt "Тэр өдөр"    # one continuation, then exit

The model only continues text. It has read books, not conversations, so it does not answer
questions yet; that comes later, with fine-tuning (persona.py). Start a sentence and let it
finish.

While it runs:
    text + Enter     the model continues your text
    Enter alone      the model starts a text of its own
    /temp 0.5        sampling temperature: lower is safer and more repetitive, higher is wilder
    /len 150         how many tokens to write
    Ctrl+C           quit

The defaults (temperature 0.7, top-p 0.9, repetition penalty 1.1) are the ones train.py uses
for the sample it prints at the end of training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

from memory import load_model


def continue_text(model, tok, prompt: str, tokens: int, temperature: float, top_p: float,
                  repetition_penalty: float) -> str:
    """The prompt followed by the model's continuation.

    The prompt goes after the document separator, the way every book starts in training, so
    the model reads it as the beginning of a text. If the prompt and the continuation do not
    fit in the context window together, only the end of the prompt is kept."""
    start = tok.encode("<|endoftext|>\n", allow_special=True)
    tokens = max(1, min(tokens, model.config.max_seq_len - len(start) - 1))
    ids = tok.encode(prompt)                                   # typed text: never a control token
    ids = ids[-(model.config.max_seq_len - len(start) - tokens):]
    x = torch.tensor([start + ids], device=next(model.parameters()).device)
    out = model.generate(x, max_new_tokens=tokens, temperature=temperature, top_p=top_p,
                         repetition_penalty=repetition_penalty, eos_token_id=tok.eot_id)
    return tok.decode(out[0].tolist()).strip()


def main(argv: Optional[List[str]] = None) -> List[str]:
    ap = argparse.ArgumentParser(description="Continue text with a trained model.")
    ap.add_argument("--model", type=Path, default=Path("out/model.pt"))
    ap.add_argument("--prompt", help="continue this text once and exit (without it: type prompts one by one)")
    ap.add_argument("--tokens", type=int, default=100, help="how many tokens to write")
    ap.add_argument("--temperature", type=float, default=0.7, help="0 = always take the most likely token")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, help="fix the randomness: the same seed gives the same text")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if not args.model.exists():
        sys.exit(f"no model at {args.model}: train one with train.py, or pass --model")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tok = load_model(args.model)
    model.to(args.device)
    settings = dict(tokens=args.tokens, temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty)

    if args.prompt is not None:
        text = continue_text(model, tok, args.prompt, **settings)
        print(text)
        return [text]

    print(f"{args.model} | {model.num_parameters() / 1e6:.2f}M parameters | vocabulary {tok.vocab_size:,} | "
          f"context {model.config.max_seq_len} tokens")
    print("Type the start of a text and press Enter; the model continues it. Enter alone: it starts on its own.")
    print("It continues text; it cannot answer questions yet. /temp 0.5 and /len 150 change the settings. "
          "Ctrl+C quits.")
    written: List[str] = []
    try:
        while True:
            line = input(f"\n[temperature {settings['temperature']}, {settings['tokens']} tokens] > ").strip()
            if line in ("exit", "quit", "/q"):
                break
            if line.startswith(("/temp", "/len")):
                name, _, value = line.partition(" ")
                try:
                    if name == "/temp":
                        settings["temperature"] = max(0.0, float(value))
                    else:
                        settings["tokens"] = max(1, int(value))
                except ValueError:
                    print("write it like /temp 0.5 or /len 150")
                continue
            text = continue_text(model, tok, line, **settings)
            print(text)
            written.append(text)
    except (KeyboardInterrupt, EOFError):
        print()
    return written


if __name__ == "__main__":
    main()
