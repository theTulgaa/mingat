"""
train.py — train MongolHistorySLM on a small text corpus
========================================================

    python train.py --data data/books.txt                     # the nano preset trains on a CPU
    python train.py --data data/ --preset micro --batch 32 --context 256   # a folder of .txt files; use a GPU
    python train.py --data data/books.txt --resume            # continue an interrupted run
    python train.py --data data/books.txt --init public/model.pt --lr 5e-4   # continue a trained model on new text

It trains a tokenizer (unless out/tokenizer.json exists), trains the model, keeps the weights
with the best validation loss in out/model.pt, and prints a sample.

The recipe is built for small data, where a model can memorize the corpus long before it has
learned the language. Each part was kept because it lowered the loss on unseen text in the
experiments described in the README:

    weight averaging The saved model is an exponential moving average (EMA) of the weights
                     over the last ~1/(1 - ema) steps. It averages out the noise of single
                     steps.
    dropout          Every sublayer output and the attention weights are randomly zeroed, so
                     no single pathway can memorize a passage.
    early stopping   The validation loss falls, then rises once memorization starts. Only the
                     best point is saved, and --patience stops the run after it.
    Muon             The weight matrices of the blocks are updated with Muon; the embedding
                     and the gains stay on AdamW. It beat AdamW in all three comparisons, by
                     1-2% each. --optimizer adamw switches back.

The validation text is held out from training. Without --val, one of every 20 blocks of the
corpus is held out.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from model import IGNORE_INDEX, PRESETS, ModelConfig, MongolHistorySLM, autocast_dtype
from tokenizer import SPECIAL_RE, BPETokenizer


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def read_text(path: Path) -> str:
    """A text file, or a folder of .txt files joined with the document separator."""
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
        if not files:
            sys.exit(f"no .txt files in {path}")
        return "\n<|endoftext|>\n".join(f.read_text(encoding="utf-8-sig") for f in files)
    return path.read_text(encoding="utf-8-sig")


def hold_out(text: str, every: int = 20, n_blocks: Optional[int] = None) -> Tuple[str, str]:
    """Split the text into blocks at line ends; every `every`-th block is validation.

    Blocks are at least 25,000 characters, and there are 40 to 400 of them: a corpus of 1M
    characters or less gives 2 validation slices, a 10M one gives 20, spread over all of its
    books. (A fixed 40 blocks made the validation of the 47-book corpus 2 slices, one of them
    most of the last book.) Where no line ends near a cut, the cut falls on a space, or
    failing that right there."""
    if n_blocks is None:
        n_blocks = min(400, max(40, len(text) // 25_000))
    size, blocks, start = max(1, len(text) // n_blocks), [], 0
    while start < len(text):
        target = min(len(text), start + size)
        end = text.find("\n", target)
        if end == -1 or end - target > size:
            end = text.find(" ", target)
            if end == -1 or end - target > size:
                end = target - 1
        end = min(len(text), end + 1)
        blocks.append(text[start:end])
        start = end
    val = [b for i, b in enumerate(blocks) if i % every == every - 1]
    train = [b for i, b in enumerate(blocks) if i % every != every - 1]
    return "".join(train), "\n<|endoftext|>\n".join(val)


def n_text_bytes(text: str) -> int:
    """UTF-8 bytes of the text, not counting special-token markers."""
    return len(SPECIAL_RE.sub("", text).encode("utf-8"))


def get_batch(data: torch.Tensor, batch: int, context: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random windows. y is x shifted left by one: y[:, t] is the token after x[:, t]."""
    starts = torch.randint(0, len(data) - context - 1, (batch,), generator=gen)
    idx = starts[:, None] + torch.arange(context + 1)
    chunk = data[idx].long()
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model: MongolHistorySLM, data: torch.Tensor, context: int, batch: int, device: str,
             max_tokens: Optional[int] = None) -> float:
    """Mean next-token loss (nats) over the data, in back-to-back windows of `context` tokens.

    With max_tokens, only every k-th window is scored, so that about max_tokens tokens are
    used and they are spread over all of the data. Taking the first max_tokens instead would
    judge the model on the first few documents only."""
    was_training = model.training
    model.eval()
    n = len(data) - 1
    starts = list(range(0, n, context))
    if max_tokens is not None:
        starts = starts[::max(1, math.ceil(len(starts) * context / max_tokens))]
    total, count = 0.0, 0
    for i in range(0, len(starts), batch):
        rows = starts[i:i + batch]
        x = torch.zeros(len(rows), context, dtype=torch.long)
        y = torch.full((len(rows), context), IGNORE_INDEX, dtype=torch.long)
        for r, s in enumerate(rows):
            length = min(context, n - s)
            x[r, :length] = data[s:s + length]
            y[r, :length] = data[s + 1:s + 1 + length]
            count += length
        total += model(x.to(device), y.to(device), reduction="sum")[1].item()
    model.train(was_training)
    return total / count


# ----------------------------------------------------------------------------------------
# Optimization
# ----------------------------------------------------------------------------------------
def lr_scale(step: int, total: int, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then a cosine from 1 down to min_ratio at the last step."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    """Exponential moving average of the weights: shadow <- decay * shadow + (1 - decay) * w.

    Early in training the decay is lower ((1 + n) / (10 + n)), so the average does not stay
    stuck at the random initial weights."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay, self.updates = decay, 0
        self.shadow = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for s, p in zip(self.shadow, model.parameters()):
            s.lerp_(p.detach(), 1 - decay)

    @contextlib.contextmanager
    def applied(self, model: torch.nn.Module):
        """Temporarily put the averaged weights into the model."""
        backup = [p.detach().clone() for p in model.parameters()]
        with torch.no_grad():
            for p, s in zip(model.parameters(), self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(model.parameters(), backup):
                    p.copy_(b)


def newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately the nearest orthogonal matrix to g (U V^T from its SVD), using only
    matrix products. Each step applies the odd polynomial a*s + b*s^3 + c*s^5 to every
    singular value s, which pushes it toward 1 (Jordan et al., 2024). After 5 steps the
    singular values lie between about 0.7 and 1.2, close enough for an update direction.
    Ones that started below roughly 1/500 of the matrix norm are only partly raised."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    tall = x.size(0) > x.size(1)
    if tall:                                  # work on the wide orientation: x @ x.T is smaller
        x = x.T
    x = x / (x.norm() + 1e-7)                 # Frobenius norm 1, so every singular value is <= 1
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    """Muon (Jordan et al., 2024), for the 2-D weight matrices inside the blocks.

    AdamW scales each weight's step separately. Muon looks at the whole matrix: it takes the
    momentum of the gradient and replaces it by the nearest orthogonal matrix, so the update
    has the same size in every direction of the matrix, including directions the gradient
    barely touches. The step is then scaled by sqrt(rows / columns) for tall matrices and
    applied with decoupled weight decay, as in AdamW."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                m = state["momentum"].lerp_(p.grad, 1 - group["momentum"])
                update = newton_schulz(p.grad.lerp(m, group["momentum"]))        # Nesterov momentum
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


MUON_LR_RATIO = 10.0     # Muon's matrices step at 10 x --lr: 0.02 at the default 2e-3, the values tested


def make_optimizers(model: MongolHistorySLM, args) -> List[torch.optim.Optimizer]:
    """Muon for the block matrices with AdamW for the embedding and the RMSNorm gains (they
    are not matrices that transform the residual stream), or AdamW for everything."""
    if args.optimizer == "adamw":
        opts = [model.configure_optimizer(lr=args.lr, weight_decay=args.weight_decay)]
    else:
        matrices = [p for n, p in model.named_parameters() if p.dim() == 2 and n.startswith("layers.")]
        ids = {id(p) for p in matrices}
        rest = [p for p in model.parameters() if id(p) not in ids]
        adam = torch.optim.AdamW(
            [{"params": [p for p in rest if p.dim() >= 2], "weight_decay": args.weight_decay},
             {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}],
            lr=args.lr, betas=(0.9, 0.95))
        opts = [Muon(matrices, lr=MUON_LR_RATIO * args.lr, weight_decay=args.weight_decay), adam]
    for opt in opts:
        for group in opt.param_groups:
            group["peak_lr"] = group["lr"]
    return opts


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train MongolHistorySLM on a small text corpus.")
    ap.add_argument("--data", type=Path, required=True, help="a .txt file or a folder of .txt files")
    ap.add_argument("--val", type=Path, help="validation text (default: hold out 5%% of --data)")
    ap.add_argument("--test", type=Path, help="test text, scored once at the end")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--preset", default="nano", choices=sorted(PRESETS))
    ap.add_argument("--vocab-size", type=int, default=2048)
    ap.add_argument("--tokenizer", type=Path, help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--init", type=Path, help="start from a trained model.pt (its tokenizer and size are kept): "
                                              "continue on new text, e.g. public pretraining first, then your corpus")
    ap.add_argument("--context", type=int, default=128, help="tokens per training window")
    ap.add_argument("--batch", type=int, default=16, help="windows per step")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--epochs", type=float, default=20.0, help="passes over the training tokens")
    ap.add_argument("--steps", type=int, help="optimizer steps (overrides --epochs)")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate (Muon's matrices use 10 times this)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2, help="0.1-0.2 for a few MB; 0 for one pass over a large corpus")
    ap.add_argument("--ema", type=float, default=0.999, help="weight-averaging decay (0 = off)")
    ap.add_argument("--optimizer", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--unique-layers", type=int, help="distinct blocks: fewer parameters, slightly higher loss")
    ap.add_argument("--d-model", type=int, help="override the preset width (the head count stays; head_dim scales)")
    ap.add_argument("--n-layers", type=int, help="override the preset depth")
    ap.add_argument("--eval-interval", type=int, default=100, help="steps between validations")
    ap.add_argument("--eval-tokens", type=int, default=20000, help="validation tokens per check (all at the end)")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many checks without a new best (0 = never)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", action="store_true", help="continue from out/ckpt.pt")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="pause after this many steps in this session (e.g. before a Colab limit); continue with --resume")
    ap.add_argument("--no-ckpt", action="store_true", help="do not write out/ckpt.pt (experiments)")
    ap.add_argument("--sample", default="Монгол", help="prompt for the sample printed at the end")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device, on_gpu = args.device, args.device.startswith("cuda")

    # ---- data and tokenizer --------------------------------------------------------------
    train_text = read_text(args.data)
    if args.val:
        val_text = read_text(args.val)
    else:
        train_text, val_text = hold_out(train_text)
    init = torch.load(args.init, map_location="cpu", weights_only=False) if args.init else None
    tok_path = args.tokenizer or args.out / "tokenizer.json"
    if init is not None:                            # a trained model brings its own tokenizer
        tok = BPETokenizer.from_dict(init["tokenizer"])
    elif tok_path.exists():
        tok = BPETokenizer.load(tok_path)
    else:
        tok = BPETokenizer.train([train_text], args.vocab_size)
        tok.save(tok_path)
    encode = lambda text: torch.tensor(tok.encode(text, allow_special=True), dtype=torch.int32)
    train_ids, val_ids = encode(train_text), encode(val_text)
    test_text = read_text(args.test) if args.test else None
    if len(train_ids) < args.context + 2:
        sys.exit(f"the corpus has only {len(train_ids)} tokens; lower --context")
    if len(val_ids) < 2:
        sys.exit("the validation text is empty; pass --val or use a larger corpus")
    tokens_per_step = args.batch * args.context * args.accum
    steps = args.steps or math.ceil(args.epochs * len(train_ids) / tokens_per_step)

    # ---- model ---------------------------------------------------------------------------
    overrides = dict(vocab_size=tok.vocab_size, dropout=args.dropout,
                     max_seq_len=max(args.context, PRESETS[args.preset]["max_seq_len"]),
                     ce_chunk_size=max(2048, 2 ** 26 // tok.vocab_size))   # ~256 MB of logits per chunk
    if args.unique_layers:
        overrides["n_unique_layers"] = args.unique_layers
    if args.n_layers:
        overrides["n_layers"] = args.n_layers
    if args.d_model:
        heads = PRESETS[args.preset]["n_heads"]
        overrides.update(d_model=args.d_model, head_dim=args.d_model // heads,
                         ffn_hidden=round(args.d_model * 8 / 3 / 8) * 8)
    cfg = ModelConfig.preset(args.preset, **overrides)
    if init is not None:                            # ... and its own size; only the dropout is taken from the flags
        cfg = ModelConfig(**{**init["config"], "dropout": args.dropout})
        if args.context > cfg.max_seq_len:
            sys.exit(f"--context {args.context} is longer than the model's max_seq_len {cfg.max_seq_len}")
    model = MongolHistorySLM(cfg)
    if init is not None:
        model.load_state_dict(init["model"])
    model.to(device)
    opts = make_optimizers(model, args)
    ema = EMA(model, args.ema) if args.ema > 0 else None
    amp_dtype = autocast_dtype() if on_gpu else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=on_gpu and amp_dtype == torch.float16)
    print(f"train {len(train_ids):,} tokens | val {len(val_ids):,} | vocab {tok.vocab_size:,} | "
          f"{model.num_parameters() / 1e6:.2f}M params ({cfg.n_layers} layers, {cfg.n_unique_layers} unique, "
          f"d={cfg.d_model}) | {steps:,} steps = {steps * tokens_per_step / len(train_ids):.1f} epochs | {device}")

    # ---- state that a resume restores ----------------------------------------------------
    step, best, history = 0, {"raw": (float("inf"), 0, None), "ema": (float("inf"), 0, None)}, []
    running, seen = 0.0, 0                          # sum and count of training losses since the last check
    ckpt_path = args.out / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        for opt, state in zip(opts, ck["opts"]):
            opt.load_state_dict(state)
        if ema is not None:
            ema.shadow, ema.updates = ck["ema"], ck["ema_updates"]
        step, best, history = ck["step"], ck["best"], ck["history"]
        running, seen = ck["running"], ck["seen"]
        gen.set_state(ck["gen"])                    # the data order continues where it stopped
        torch.set_rng_state(ck["rng"])              # and so do the dropout masks
        if on_gpu and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        scaler.load_state_dict(ck["scaler"])
        print(f"resumed from step {step}")

    def save_ckpt(step_now: int, running_loss: float, running_count: int) -> None:
        """Everything a resume needs to continue as if the run had never stopped."""
        if not args.no_ckpt:
            torch.save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                        "ema": ema.shadow if ema else None, "ema_updates": ema.updates if ema else 0,
                        "running": running_loss, "seen": running_count,
                        "step": step_now, "best": best, "history": history, "gen": gen.get_state(),
                        "rng": torch.get_rng_state(), "scaler": scaler.state_dict(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if on_gpu else None}, ckpt_path)

    def check(step_now: int, train_loss: Optional[float], elapsed: float) -> bool:
        """Validate the current and the averaged weights; remember the best of each.
        Returns True when the run should stop (patience ran out)."""
        scores = {"raw": evaluate(model, val_ids, args.context, 64, device, args.eval_tokens)}
        if ema is not None:
            with ema.applied(model):
                scores["ema"] = evaluate(model, val_ids, args.context, 64, device, args.eval_tokens)
        for kind, loss in scores.items():
            if loss < best[kind][0]:
                weights = model.parameters() if kind == "raw" else ema.shadow
                best[kind] = (loss, step_now, [w.detach().clone() for w in weights])
        history.append((step_now, train_loss, scores["raw"], scores.get("ema")))
        print(f"step {step_now:>6}/{steps} | epoch {step_now * tokens_per_step / len(train_ids):5.1f} | "
              f"train {'  -  ' if train_loss is None else format(train_loss, '.3f')} | val {scores['raw']:.3f}" +
              (f" | val(avg) {scores['ema']:.3f}" if ema is not None else "") +
              f" | best {min(best['raw'][0], best['ema'][0]):.3f} | "
              f"{step_now * tokens_per_step / max(elapsed, 1e-9):,.0f} tok/s", flush=True)
        save_ckpt(step_now, 0.0, 0)                 # the running training loss restarts after a check
        last_best = max(best["raw"][1], best["ema"][1])
        return args.patience > 0 and (step_now - last_best) >= args.patience * args.eval_interval

    # ---- training loop -------------------------------------------------------------------
    model.train()
    if init is not None and step == 0:
        # Score the model we start from. It is then the best so far, so a continuation that
        # only makes things worse on the validation text ends by saving these weights again.
        check(0, None, 0.0)
    t0, start_step = time.perf_counter(), step
    while step < steps:
        scale = lr_scale(step, steps, args.warmup)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = group["peak_lr"] * scale
        for _ in range(args.accum):
            x, y = get_batch(train_ids, args.batch, args.context, gen)
            with torch.autocast(device_type="cuda" if on_gpu else "cpu", dtype=amp_dtype, enabled=on_gpu):
                _, loss = model(x.to(device), y.to(device))
            scaler.scale(loss / args.accum).backward()
            running, seen = running + loss.item(), seen + 1
        for opt in opts:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            scaler.step(opt)
        scaler.update()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        step += 1
        if step % args.eval_interval == 0 or step == steps:
            elapsed = (time.perf_counter() - t0) * step / max(step - start_step, 1)   # as if never paused
            stop = check(step, running / max(seen, 1), elapsed)
            running, seen = 0.0, 0
            if stop:
                print(f"stopping: no new best for {args.patience} checks")
                break
        if args.stop_after and step - start_step >= args.stop_after and step < steps:
            save_ckpt(step, running, seen)
            print(f"paused at step {step}; run the same command with --resume to continue")
            return {"paused_at": step}
    seconds = time.perf_counter() - t0

    # ---- keep the best weights, score them on all of val (and test) ------------------------
    kind = "ema" if best["ema"][0] <= best["raw"][0] else "raw"
    with torch.no_grad():
        for p, w in zip(model.parameters(), best[kind][2]):
            p.copy_(w)

    def score(ids: torch.Tensor, text: str) -> dict:
        nats = evaluate(model, ids, args.context, 64, device)
        return {"loss": nats, "ppl": math.exp(nats),
                "bits_per_byte": nats * (len(ids) - 1) / (math.log(2) * n_text_bytes(text))}

    result = {
        "params": model.num_parameters(), "vocab": tok.vocab_size, "train_tokens": len(train_ids),
        "steps": step, "epochs": step * tokens_per_step / len(train_ids), "seconds": seconds,
        "best_raw": best["raw"][:2], "best_ema": best["ema"][:2], "kept": kind,
        "val": score(val_ids, val_text),
        "test": score(encode(test_text), test_text) if test_text else None,
        "history": history, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    torch.save({"config": asdict(cfg), "model": model.state_dict(), "tokenizer": tok.to_dict(),
                "val_loss": result["val"]["loss"], "step": best[kind][1]}, args.out / "model.pt")
    (args.out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"\nbest: step {best[kind][1]} ({'averaged' if kind == 'ema' else 'raw'} weights) | "
          f"val loss {result['val']['loss']:.3f} = perplexity {result['val']['ppl']:.1f} = "
          f"{result['val']['bits_per_byte']:.3f} bits/byte" +
          (f" | test loss {result['test']['loss']:.3f}" if test_text else "") +
          f" | saved {args.out / 'model.pt'}")

    if args.sample:     # start the way documents start in the corpus: the separator, a newline, the text
        prompt = torch.tensor([tok.encode("<|endoftext|>\n", allow_special=True) + tok.encode(args.sample)], device=device)
        out = model.generate(prompt, max_new_tokens=60, temperature=0.7, top_p=0.9,
                             repetition_penalty=1.1, eos_token_id=tok.eot_id)
        print("sample:", tok.decode(out[0].tolist()).replace("\n", " "))
    return result


if __name__ == "__main__":
    main()
