"""CPU diagnostic reproductions for CODE_REVIEW.md; no corpus or checkpoint needed.

Run from the repository root: python review/reproduce_findings.py
Requires recent PyTorch with CPU GradScaler (tested on 2.15.0).
Outputs demonstrate current behavior, not a passing regression-test suite.
"""

from __future__ import annotations

import math
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import persona
import train
from memory import Memory, mixture_loss, tune
from model import IGNORE_INDEX, ModelConfig, MongolHistorySLM
from tokenizer import _SAMPLE, BPETokenizer


def tiny_model(**overrides) -> MongolHistorySLM:
    cfg = dict(vocab_size=40, d_model=32, n_layers=4, n_unique_layers=2,
               n_heads=4, n_kv_heads=2, head_dim=8, ffn_hidden=48,
               max_seq_len=32, ce_chunk_size=3)
    return MongolHistorySLM(ModelConfig(**{**cfg, **overrides}))


def check_core_gradients() -> None:
    """Force actual loss chunks, shared blocks, and activation checkpointing."""
    a, b = tiny_model(), tiny_model()
    b.load_state_dict(a.state_dict())
    a.gradient_checkpointing = True
    x, y = torch.randint(0, 40, (2, 12)), torch.randint(0, 40, (2, 12))
    y[0, ::2] = IGNORE_INDEX
    chunked = a(x, y)[1]
    dense = F.cross_entropy(b(x)[0].reshape(-1, 40), y.reshape(-1))
    chunked.backward()
    dense.backward()
    gap = max((p.grad - q.grad).abs().max().item() for p, q in zip(a.parameters(), b.parameters()))
    print(f"Core check: chunked={chunked.item():.7f}, dense={dense.item():.7f}, max gradient error={gap:.2e}")
    assert gap < 1e-5, "core reference comparison failed"


def probe_amp_clipping() -> None:
    block = torch.nn.Parameter(torch.ones(2, 2))
    embedding = torch.nn.Parameter(torch.ones(2, 2))
    opts = [train.Muon([block]), torch.optim.AdamW([embedding], lr=0.01)]
    scaler = torch.amp.GradScaler("cpu", init_scale=2.0)
    scaler.scale(block.square().sum() + embedding.square().sum()).backward()
    block.grad[0, 0] = float("nan")
    for opt in opts:
        scaler.unscale_(opt)
    initially_finite = bool(embedding.grad.isfinite().all())
    torch.nn.utils.clip_grad_norm_([block, embedding], 1.0)
    for opt in opts:
        scaler.step(opt)
    scaler.update()
    print(f"1 AMP: clean group initially finite={initially_finite}; weights after step finite="
          f"{bool(embedding.isfinite().all())} (expected True)")


def probe_memory_normalization() -> None:
    mem = Memory(torch.zeros(16, 32), torch.ones(16, dtype=torch.long),
                 k=16, temperature=0.01, lam=0.2, recall_temperature=None)
    plain = torch.tensor([[0.5, 0.25, 0.25]])
    sims, values = torch.zeros(1, 16), torch.ones(1, 16, dtype=torch.long)
    got = mem.combine(plain.log(), sims, values).exp()
    votes = torch.zeros_like(plain).scatter_add_(1, values, (sims / mem.temperature).softmax(-1))
    expected = (1 - mem.lam) * plain + mem.lam * votes
    print(f"2 Voting: probabilities={got.tolist()}, sum={got.sum().item():.6f}; expected={expected.tolist()}")


def probe_preference_labels() -> None:
    tok = BPETokenizer.train([_SAMPLE, "system user assistant old history preferred rejected"], 300)
    history = [{"role": "user", "content": "old"},
               {"role": "assistant", "content": "history"},
               {"role": "user", "content": "preferred"}]
    chosen = persona.pad_batch([persona.encode_chat(tok, history + [{"role": "assistant", "content": "preferred"}])])
    rejected = persona.pad_batch([persona.encode_chat(tok, history + [{"role": "assistant", "content": "rejected"}])])
    learned = tok.decode(rejected[1][rejected[1] != IGNORE_INDEX].tolist(), skip_special=False)
    print(f"3 ORPO labels: {learned!r}; expected only 'rejected<|im_end|>' for preference training")

    prefix_len = len(persona.chat_tokens(tok, history, add_generation_prompt=True)[0])

    def final_reply_only(pair):
        x, y = pair
        y = y.clone()
        y[:, :prefix_len - 1] = IGNORE_INDEX
        return x, y

    model = tiny_model(vocab_size=tok.vocab_size, n_layers=2, max_seq_len=256).eval()
    with torch.no_grad():
        full = persona.orpo_loss(model, chosen, rejected).item()
        final = persona.orpo_loss(model, final_reply_only(chosen), final_reply_only(rejected)).item()
    print(f"  ORPO with existing labels={full:.7f}; final-reply-only labels={final:.7f}")


def probe_tuning_baseline() -> None:
    mem = Memory(torch.zeros(16, 32), torch.ones(16, dtype=torch.long), recall_temperature=None)
    part = (torch.tensor([math.log(0.9)]), torch.full((1, 16), 0.9),
            torch.ones(1, 16, dtype=torch.long), torch.zeros(1, 16, dtype=torch.bool))
    tune(mem, part)
    plain = -part[0].double().mean().item()
    chosen = mixture_loss(*part, k=mem.k, temperature=mem.temperature, lam=mem.lam)
    print(f"4 Tuning wrong neighbors: plain loss={plain:.7f}, selected loss={chosen:.7f}, "
          f"lambda={mem.lam}; expected lambda=0")


def probe_context_one() -> None:
    # Bound this known infinite Python loop to avoid unbounded allocations.
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        print("5 Context=1: skipped; POSIX timer unavailable")
        return
    model = tiny_model()

    def timeout(signum, frame):
        raise TimeoutError("window loop did not advance")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 0.05)
    try:
        Memory.build(model, torch.tensor([2, 3, 4]), context=1)
        print("5 Context=1: returned")
    except TimeoutError:
        print("5 Context=1: nonterminating loop interrupted; expected prompt return or ValueError")
    except ValueError as exc:
        print(f"5 Context=1: rejected promptly: {exc}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def probe_sampler() -> None:
    x, y = train.get_batch(torch.arange(5), 100, 3, torch.Generator().manual_seed(0))
    print(f"6 Sampler: starts={x[:, 0].unique().tolist()}, last targets={y[:, -1].unique().tolist()}; "
          "expected starts [0, 1] and last targets [3, 4]")


def probe_fp16_normalization() -> None:
    g = torch.randn(16, 16) * 1e5
    direction = train.newton_schulz(g, dtype=torch.float16)
    print(f"7 fp16 Muon on large finite input: output finite={bool(direction.isfinite().all())} (expected True)")


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}; CPU diagnostic probes")
    check_core_gradients()
    probe_amp_clipping()
    probe_memory_normalization()
    probe_preference_labels()
    probe_tuning_baseline()
    probe_context_one()
    probe_sampler()
    probe_fp16_normalization()


if __name__ == "__main__":
    main()
