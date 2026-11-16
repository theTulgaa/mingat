"""
sweep.py — train several settings one after another, then compare them on the test books
=========================================================================================

    python sweep.py --data data/clean --test data/test --out runs --smoke   # first: a check of a few minutes
    python sweep.py --data data/clean --test data/test --out runs           # then: every run in PLAN

train_colab.ipynb runs this on a Colab GPU. Each run is one train.py command with its own
folder in --out. If the session ends (a Colab disconnect, a closed laptop), run the same
command again: a finished run (it has result.json) is skipped, and an interrupted one
continues from its last checkpoint. All runs share one tokenizer, --out/tokenizer.json.

The smoke check trains `small` for 300 steps on the real books, pausing halfway and resuming,
so it goes through everything the long runs use: the GPU, 16-bit training, the checkpoint,
the resume, the final scores and the sample. It also measures the speed, and from it the
time the PLAN will take.

At the end, a table: bits per byte on the validation text (5% of the training books) and on
the test books (books no run trained on), for every run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent

# The runs, in order: name, train.py flags, the question it answers. Every run also gets the
# flags in COMMON. The first two ask one thing each; the third builds on the answer.
PLAN = [
    ("micro_6ep", "--preset micro --dropout 0.1 --epochs 6",
     "the recipe of the 1.018 model on 2.3 times the books: what the new books alone give"),
    ("small_6ep", "--preset small --dropout 0.1 --epochs 6",
     "3 times the parameters, the same number of passes"),
    ("small_12ep", "--preset small --dropout 0.1 --epochs 12 --patience 8",
     "twice the passes; it stops early once the validation loss has not improved for 4,000 steps"),
]
COMMON = "--eval-interval 500 --ckpt-minutes 10"
EARLIER_BEST = ("micro, 6 epochs, 40 books (CPU, 2026-11-13)", 1.018)


def run_train(flags: List[str], log: Path) -> None:
    """train.py with these flags; its output goes to the screen and to the log as it comes."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, str(HERE / "train.py"), *flags]
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
        if proc.wait() != 0:
            sys.exit(f"train.py failed (exit code {proc.returncode}); the end of its output is above and in {log}")


def muon_speed() -> None:
    """Time one Muon step of `small` on the GPU, one matrix at a time and stacked by shape."""
    import torch

    import train
    from model import ModelConfig, MongolHistorySLM
    if not torch.cuda.is_available():
        print("no GPU: Muon is not timed (a CPU always takes the matrices one at a time)")
        return
    model = MongolHistorySLM(ModelConfig.preset("small", vocab_size=2048)).cuda()
    mats = [p for n, p in model.named_parameters() if p.dim() == 2 and n.startswith("layers.")]
    for p in mats:
        p.grad = torch.randn_like(p) * 1e-3
    for batched in (False, True):
        opt = train.Muon(mats, lr=1e-9, batched=batched)
        for _ in range(5):                      # warm-up: memory pools, kernel selection
            opt.step()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            opt.step()
        torch.cuda.synchronize()
        print(f"Muon step of `small` on the GPU, {'stacked by shape' if batched else 'one matrix at a time'}: "
              f"{(time.perf_counter() - start) / 30 * 1e3:.1f} ms")


def drive_check(folder: Path) -> None:
    """The runs keep their checkpoint in --out (on Google Drive in Colab), written to a
    temporary file and renamed over the previous one. Check that this works there, with
    a small file, before a long run depends on it."""
    tmp, target = folder / "write_check.tmp", folder / "write_check"
    for payload in (b"first" * 200_000, b"second" * 200_000):
        tmp.write_bytes(payload)
        os.replace(tmp, target)                                  # the second time over an existing file
        assert target.read_bytes() == payload, f"a file written to {folder} did not read back"
    target.unlink()
    print(f"checkpoints can be written to {folder}")


def epochs_of(flags: str) -> float:
    words = flags.split()
    return float(words[words.index("--epochs") + 1])


def smoke(args) -> None:
    passed = args.out / "smoke_passed"
    if passed.exists():
        print(f"== smoke check: passed in an earlier session, skipped (delete {passed} to run it again)")
        return
    out = Path(tempfile.mkdtemp(prefix="mingat_smoke_"))      # local disk: its checkpoints are not worth uploading
    flags = ["--data", str(args.data), "--test", str(args.test), "--out", str(out),
             "--tokenizer", str(args.out / "tokenizer.json"), "--preset", "small", "--dropout", "0.1",
             "--steps", "300", "--eval-interval", "100", "--warmup", "20"]
    print("== smoke check: `small`, 300 steps, paused at 150 and resumed")
    run_train(flags + ["--stop-after", "150", "--sample", ""], out / "train.log")
    run_train(flags + ["--resume"], out / "train.log")
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["steps"] == 300 and [h[0] for h in result["history"]] == [100, 200, 300], \
        "the resumed run did not continue where the paused one stopped"
    tokens_per_s = 150 * 16 * 128 / result["seconds"]         # the resumed half: 150 steps of 16 x 128 tokens
    hours = sum(epochs_of(flags_) for _, flags_, _ in PLAN) * result["train_tokens"] / tokens_per_s / 3600
    print(f"\n== smoke check passed: {tokens_per_s:,.0f} tokens/s for `small`, checks included. One pass over the "
          f"books is {result['train_tokens']:,} tokens, so the {len(PLAN)} runs take at most {hours:.1f} h "
          f"(less: `micro` is faster, and the last run may stop early).")
    muon_speed()
    drive_check(args.out)
    shutil.rmtree(out, ignore_errors=True)
    passed.write_text(f"{tokens_per_s:.0f} tokens/s for small\n", encoding="utf-8")


def summary(args) -> None:
    rows = []
    for name, flags, why in PLAN:
        path = args.out / name / "result.json"
        if path.exists():
            r = json.loads(path.read_text(encoding="utf-8"))
            rows.append((name, r))
    if not rows:
        return
    print(f"\n{'run':<12} {'params':>7} {'passes':>6} {'best step':>9} {'val bpb':>8} {'test bpb':>8} {'minutes':>7}")
    for name, r in rows:
        best_step = r["best_ema"][1] if r["kept"] == "ema" else r["best_raw"][1]
        print(f"{name:<12} {r['params'] / 1e6:6.1f}M {r['epochs']:6.1f} {best_step:9,} {r['val']['bits_per_byte']:8.3f} "
              f"{r['test']['bits_per_byte']:8.3f} {r['seconds'] / 60:7.0f}")
    print(f"earlier best: {EARLIER_BEST[0]}: test {EARLIER_BEST[1]:.3f}")
    by_val = min(rows, key=lambda nr: nr[1]["val"]["bits_per_byte"])[0]
    by_test = min(rows, key=lambda nr: nr[1]["test"]["bits_per_byte"])[0]
    print(f"lowest on validation: {by_val}; lowest on the test books: {by_test}")
    (args.out / "summary.json").write_text(json.dumps(
        {name: {k: r[k] for k in ("params", "epochs", "steps", "seconds", "kept", "val", "test")} for name, r in rows},
        indent=1), encoding="utf-8")
    print(f"minutes: the time of the last session of each run (a resumed run counts only its last part)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train the runs in PLAN one after another and compare them.")
    ap.add_argument("--data", type=Path, required=True, help="the training books (a folder of .txt)")
    ap.add_argument("--test", type=Path, required=True, help="the test books (a folder of .txt)")
    ap.add_argument("--out", type=Path, required=True, help="a folder for every run, e.g. on Google Drive")
    ap.add_argument("--smoke", action="store_true", help="only the smoke check")
    ap.add_argument("--only", help="only this run of PLAN")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    args.out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        smoke(args)
        return
    for name, flags, why in PLAN:
        if args.only and name != args.only:
            continue
        out = args.out / name
        if (out / "result.json").exists():
            print(f"== {name}: finished earlier, skipped")
            continue
        print(f"\n== {name}: {why}" + (" (continuing)" if (out / "ckpt.pt").exists() else ""))
        run_train(["--data", str(args.data), "--test", str(args.test), "--out", str(out),
                   "--tokenizer", str(args.out / "tokenizer.json"), *COMMON.split(), *flags.split(), "--resume"],
                  out / "train.log")
        (out / "ckpt.pt").unlink(missing_ok=True)       # finished: the checkpoint (~400 MB for `small`) is not needed
    summary(args)


if __name__ == "__main__":
    main()
