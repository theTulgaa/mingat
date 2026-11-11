"""
memory.py — a memory of the training text, searched while the model writes
==========================================================================

A small model cannot store a book in its weights without also memorizing the noise, and a
well-regularized one deliberately does not. So the facts go somewhere else. After training,
the model reads its corpus once more and stores, for every position:

    key   = its hidden state there (what the text so far "means" to the model)
    value = the token that actually came next

To predict a next token, the model finds the stored keys closest to its current hidden state
and looks at what followed them. Nothing is trained, and new text can be added at any time.

Two ways of using the neighbours are combined. s_i is the cosine similarity of neighbour i
to the query (1 = identical context) and v_i is its stored next token.

1. Similar contexts vote (the kNN-LM of Khandelwal et al., 2020). It helps on new text.
       w_i       = exp((s_i - 1) / temperature)                    over the k nearest
       p_soft(y) = (1 - lam) * p_model(y) + lam * (sum of w_i with v_i = y) / (sum of w_i)

2. Recognized contexts are recalled. A much sharper kernel counts only near-identical
   contexts, and the count enters the way an n-gram model backs off (Witten & Bell, 1991):
       c_i  = exp((s_i - 1) / recall_temperature)                  ~1 if identical, ~0 otherwise
       N    = sum of c_i                 how many stored contexts match this one
       T    = number of different tokens that followed them (at least 1)
       p(y) = (T * p_soft(y) + sum of c_i with v_i = y) / (T + N)

   If nothing matches, N = 0 and p = p_soft. If exactly one stored context matches (N = 1,
   T = 1), its token gets at least half the probability, so greedy decoding recites it
   whatever the model itself prefers. If many match and agree, the memory is trusted almost
   fully. If they disagree (T is large), it is trusted less and the model decides.

Usage
    python memory.py --model out/model.pt --data data/books.txt     # build, tune, report, save
    python memory.py --model out/model.pt --memory out/memory.pt --ask "Чингис хаан"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model import IGNORE_INDEX, ModelConfig, MongolHistorySLM
from tokenizer import BPETokenizer

RECALL_K = 16      # the recall kernel looks at the 16 nearest neighbours


class Memory:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor, k: int = 64, temperature: float = 0.05,
                 lam: float = 0.2, recall_temperature: Optional[float] = 0.005):
        self.keys = keys            # (N, d) unit-length hidden states
        self.values = values        # (N,)  the token that followed each one
        self.k, self.temperature, self.lam = k, temperature, lam
        self.recall_temperature = recall_temperature        # None switches recall off

    def __len__(self) -> int:
        return len(self.values)

    # -- building --------------------------------------------------------------------------
    @classmethod
    @torch.no_grad()
    def build(cls, model: MongolHistorySLM, ids: torch.Tensor, context: int, eot_id: int = 1,
              batch: int = 32) -> "Memory":
        """Read `ids` (a 1-D token tensor) and store one (key, next token) pair per position.

        Each document (it starts at an <|endoftext|> token) is read from its own beginning in
        windows of `context` tokens that overlap by half. A key is taken from the window that
        gives its position the most left context, so every key past the first half-window
        has at least context/2 tokens behind it. Because documents are read from their
        start, a prompt that repeats the start of a stored document reproduces its keys."""
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        ids = ids.long()
        cuts = sorted({0, *(ids == eot_id).nonzero().flatten().tolist()})
        half, windows = context // 2, []                  # (start, length, first offset to keep)
        for start, end in zip(cuts, cuts[1:] + [len(ids)]):
            pos = start
            while True:
                length = min(context, end - pos)
                windows.append((pos, length, 0 if pos == start else half))
                if pos + length >= end:
                    break
                pos += half
        keys = torch.full((len(ids) - 1, model.config.d_model), float("nan"))
        for i in range(0, len(windows), batch):
            rows = windows[i:i + batch]
            x = torch.zeros(len(rows), context, dtype=torch.long)
            for r, (pos, length, _) in enumerate(rows):
                x[r, :length] = ids[pos:pos + length]
            h = F.normalize(model.forward_hidden(x.to(device)).float(), dim=-1).cpu()
            for r, (pos, length, keep) in enumerate(rows):
                stop = min(pos + length, len(ids) - 1)    # the very last token has no successor
                keys[pos + keep:stop] = h[r, keep:stop - pos]
        model.train(was_training)
        assert not keys.isnan().any(), "a position was left without a key"
        return cls(keys, ids[1:].clone())

    # -- retrieval -------------------------------------------------------------------------
    @torch.no_grad()
    def neighbours(self, h: torch.Tensor, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """h: (B, d) hidden states -> (similarities, values), each (B, k), nearest first."""
        k = min(k or max(self.k, RECALL_K), len(self))
        q = F.normalize(h.float(), dim=-1).to(self.keys.device)
        sims, idx = [], []
        step = max(1, 50_000_000 // len(self))            # bounds the (rows x N) similarity matrix
        for i in range(0, len(q), step):
            top = (q[i:i + step] @ self.keys.T).topk(k, dim=-1)
            sims.append(top.values)
            idx.append(top.indices)
        return torch.cat(sims), self.values[torch.cat(idx)]

    def combine(self, logp_model: torch.Tensor, sims: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Mix the model's log-probabilities (B, V) with the retrieved neighbours (B, k),
        nearest first. Returns log-probabilities (B, V). The formulas are in the module docstring."""
        p = logp_model.exp()
        sims, values = sims.to(p), values.to(p.device)
        if self.lam > 0:
            w = torch.exp((sims[:, :self.k] - 1.0) / self.temperature)
            votes = torch.zeros_like(p).scatter_add_(1, values[:, :self.k], w)
            p = (1 - self.lam) * p + self.lam * votes / votes.sum(-1, keepdim=True).clamp_min(1e-30)
        if self.recall_temperature:
            c = torch.exp((sims[:, :RECALL_K] - 1.0) / self.recall_temperature)
            counts = torch.zeros_like(p).scatter_add_(1, values[:, :RECALL_K], c)
            n = counts.sum(-1, keepdim=True)
            t = counts.clamp(max=1).sum(-1, keepdim=True).clamp(min=1)
            p = (t * p + counts) / (t + n)
        return p.clamp_min(1e-30).log()

    @torch.no_grad()
    def mix(self, h: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """The logit_hook for model.generate: (B, d), (B, V) -> log-probabilities (B, V)."""
        sims, values = self.neighbours(h)
        return self.combine(logits.float().log_softmax(-1), sims, values)

    # -- saving and loading ----------------------------------------------------------------
    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"keys": self.keys.half(), "values": self.values.int(), "k": self.k,
                    "temperature": self.temperature, "lam": self.lam,
                    "recall_temperature": self.recall_temperature}, path)

    @classmethod
    def load(cls, path) -> "Memory":
        d = torch.load(path, map_location="cpu", weights_only=True)
        keys = F.normalize(d["keys"].float(), dim=-1)      # undo the rounding of half precision
        return cls(keys, d["values"].long(), d["k"], d["temperature"], d["lam"], d["recall_temperature"])


# ----------------------------------------------------------------------------------------
# Scoring and tuning
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def collect(model: MongolHistorySLM, memory: Memory, ids: torch.Tensor, context: int, k: int = 64,
            batch: int = 32, max_positions: int = 20_000, offset: int = 0):
    """For positions of `ids`: the model's log-probability of the true next token, the
    similarities and values of its k nearest neighbours, and which of them stored that true
    token. That is everything the mixture needs, so tuning can try many settings without
    searching again. The text is read in back-to-back windows starting at `offset`; if it is
    longer than max_positions, evenly spaced windows are scored."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    ids = ids.long()
    n = len(ids) - 1
    starts = list(range(offset, n, context))
    starts = starts[::max(1, math.ceil(len(starts) * context / max_positions))]
    logp_true, sims, values, hits = [], [], [], []
    for i in range(0, len(starts), batch):
        rows = starts[i:i + batch]
        x = torch.zeros(len(rows), context, dtype=torch.long)
        y = torch.full((len(rows), context), IGNORE_INDEX, dtype=torch.long)
        for r, s in enumerate(rows):
            length = min(context, n - s)
            x[r, :length] = ids[s:s + length]
            y[r, :length] = ids[s + 1:s + 1 + length]
        h = model.forward_hidden(x.to(device)).float()
        real = (y != IGNORE_INDEX).to(device)
        h, target = h[real], y.to(device)[real]
        logp = model.lm_head(h).float().log_softmax(-1)
        s, v = memory.neighbours(h, k)
        logp_true.append(logp.gather(1, target[:, None]).squeeze(1).cpu())
        sims.append(s.cpu())
        values.append(v)
        hits.append(v == target.cpu()[:, None])
    model.train(was_training)
    return torch.cat(logp_true), torch.cat(sims), torch.cat(values), torch.cat(hits)


def mixture_loss(logp_true, sims, values, hits, k: int, temperature: float, lam: float,
                 recall_temperature: Optional[float] = None) -> float:
    """Mean loss (nats) of the mixture on collected positions. Same formulas as combine()."""
    p = logp_true.double().exp()
    if lam > 0:
        w = torch.exp((sims[:, :k].double() - 1.0) / temperature)
        p = (1 - lam) * p + lam * (w * hits[:, :k]).sum(-1) / w.sum(-1).clamp_min(1e-30)
    if recall_temperature:
        c = torch.exp((sims[:, :RECALL_K].double() - 1.0) / recall_temperature)
        v = values[:, :RECALL_K]
        of_token = ((v[:, :, None] == v[:, None, :]) * c[:, None, :]).sum(-1)   # count of each neighbour's token
        # T = sum over distinct tokens of min(1, count): each neighbour carries its share of that.
        t = (c * of_token.clamp(max=1) / of_token.clamp_min(1e-300)).sum(-1).clamp(min=1)
        p = (t * p + (c * hits[:, :RECALL_K]).sum(-1)) / (t + c.sum(-1))
    return -p.clamp_min(1e-30).log().mean().item()


def tune(memory: Memory, new_text, known_text=None, tolerance: float = 0.005, verbose: bool = False) -> None:
    """Choose the memory's settings from collected positions.

    k, temperature, lam: whatever gives the lowest loss on `new_text` (held-out text).
    recall_temperature: if `known_text` is given (the corpus itself, read with shifted
    windows), the setting that predicts it best while costing at most `tolerance` nats on
    the new text. In words: recall as much as possible without hurting new text."""
    best = (float("inf"), None)
    for k in (8, 16, 32, 64):
        if k > new_text[1].shape[1]:
            continue
        for temperature in (0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2):
            for lam in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
                loss = mixture_loss(*new_text, k=k, temperature=temperature, lam=lam)
                if loss < best[0]:
                    best = (loss, (k, temperature, lam))
    memory.k, memory.temperature, memory.lam = best[1]
    if known_text is None:
        return
    soft = dict(k=memory.k, temperature=memory.temperature, lam=memory.lam)
    rows = [(None, best[0], mixture_loss(*known_text, **soft))]
    for rt in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        rows.append((rt, mixture_loss(*new_text, **soft, recall_temperature=rt),
                     mixture_loss(*known_text, **soft, recall_temperature=rt)))
    allowed = [r for r in rows if r[1] <= best[0] + tolerance]
    memory.recall_temperature = min(allowed, key=lambda r: r[2])[0]
    if verbose:
        print("  recall temperature | loss on new text | loss on its own corpus (shifted windows)")
        for rt, new, known in rows:
            mark = "  <- chosen" if rt == memory.recall_temperature else ""
            print(f"  {'off' if rt is None else rt:>18} | {new:16.3f} | {known:8.3f}{mark}")


def prompt_starts(ids: torch.Tensor, eot_id: int, length: int, offset: int = 0, limit: int = 60) -> list:
    """Where to cut test prompts of `length` tokens out of the corpus. With offset = 0: the
    start of every document. With offset > 0: every `length` tokens inside each document,
    from `offset` tokens in, so a single long document (a book) still gives many prompts.
    At most `limit` positions are returned, evenly spread."""
    cuts = sorted({0, *(ids == eot_id).nonzero().flatten().tolist()}) + [len(ids)]
    starts = []
    for s, e in zip(cuts, cuts[1:]):
        last = e - length
        starts += [s] if offset == 0 and s <= last else list(range(s + offset, last + 1, length)) if offset else []
    return starts[::max(1, math.ceil(len(starts) / limit))]


@torch.no_grad()
def recitation(model: MongolHistorySLM, memory: Optional[Memory], ids: torch.Tensor, eot_id: int,
               prompt_len: int = 16, target_len: int = 32, offset: int = 0) -> Tuple[float, float, int]:
    """Does the model know its own corpus? Give it `prompt_len` tokens of the corpus (see
    prompt_starts for where they come from) and decode greedily. Returns (share of the next
    `target_len` tokens reproduced before the first mistake, share of prompts continued
    without any mistake, number of prompts)."""
    ids = ids.long()
    starts = prompt_starts(ids, eot_id, prompt_len + target_len, offset)
    device = next(model.parameters()).device
    matched, full = 0.0, 0
    for s in starts:
        prompt = ids[s:s + prompt_len][None].to(device)
        target = ids[s + prompt_len:s + prompt_len + target_len]
        out = model.generate(prompt, max_new_tokens=target_len, temperature=0.0,
                             logit_hook=memory.mix if memory else None)[0, prompt_len:].cpu()
        wrong = (out != target).nonzero()
        first = int(wrong[0]) if len(wrong) else target_len
        matched += first / target_len
        full += first == target_len
    return matched / max(len(starts), 1), full / max(len(starts), 1), len(starts)


def load_model(path) -> Tuple[MongolHistorySLM, BPETokenizer]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = MongolHistorySLM(ModelConfig(**ck["config"])).eval()
    model.load_state_dict(ck["model"])
    return model, BPETokenizer.from_dict(ck["tokenizer"])


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Build, tune and use the memory of a trained model.")
    ap.add_argument("--model", type=Path, default=Path("out/model.pt"))
    ap.add_argument("--data", type=Path, help="the training text: build the memory from it")
    ap.add_argument("--val", type=Path, help="held-out text for tuning (default: the same 5%% train.py held out)")
    ap.add_argument("--test", type=Path, help="extra held-out text, scored with the tuned settings")
    ap.add_argument("--memory", type=Path, default=Path("out/memory.pt"))
    ap.add_argument("--context", type=int, default=128, help="window length; use the training --context")
    ap.add_argument("--ask", action="append", default=[], help="a prompt to continue (repeatable)")
    ap.add_argument("--positions", type=int, default=20_000, help="positions scored per text when tuning and reporting")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok = load_model(args.model)
    encode = lambda text: torch.tensor(tok.encode(text, allow_special=True), dtype=torch.long)

    if args.data:
        from train import hold_out, read_text
        train_text = read_text(args.data)
        if args.val:
            val_text = read_text(args.val)
        else:
            train_text, val_text = hold_out(train_text)
        train_ids, val_ids = encode(train_text), encode(val_text)
        memory = Memory.build(model, train_ids, args.context, tok.eot_id)
        print(f"memory: {len(memory):,} entries of {memory.keys.shape[1]} numbers "
              f"({memory.keys.numel() * 2 / 2**20:.0f} MB on disk)")

        # Tune on the first half of the held-out text; report on the second half.
        mid = len(val_ids) // 2
        score = lambda ids, **kw: collect(model, memory, ids, args.context, max_positions=args.positions, **kw)
        tune_part, report_part = score(val_ids[:mid]), score(val_ids[mid:])
        known = score(train_ids, offset=args.context // 4)
        tune(memory, tune_part, known, verbose=True)
        settings = dict(k=memory.k, temperature=memory.temperature, lam=memory.lam)

        def report(name: str, part) -> None:
            plain = -part[0].double().mean().item()
            soft = mixture_loss(*part, **settings)
            both = mixture_loss(*part, **settings, recall_temperature=memory.recall_temperature)
            print(f"{name}: model alone {plain:.3f} (perplexity {math.exp(plain):.1f}) -> with voting {soft:.3f} "
                  f"-> with voting and recall {both:.3f} (perplexity {math.exp(both):.1f})")

        print(f"settings: k {memory.k}, temperature {memory.temperature}, lam {memory.lam}, "
              f"recall temperature {memory.recall_temperature}")
        report("new text (held out)", report_part)
        if args.test:
            report("new text (test file)", score(encode(read_text(args.test))))
        report("its own corpus, shifted windows", known)
        print("reciting the corpus (greedy decoding, share of the next 32 tokens right before the first mistake):")
        for label, offset, prompt_len in (("the first 16 tokens of a document", 0, 16),
                                          ("64 tokens from the middle of a document", 200, 64),
                                          ("16 tokens from the middle of a document", 200, 16)):
            a = recitation(model, None, train_ids, tok.eot_id, prompt_len=prompt_len, offset=offset)
            b = recitation(model, memory, train_ids, tok.eot_id, prompt_len=prompt_len, offset=offset)
            if a[2] >= 5:       # with a handful of documents the first line would mean nothing
                print(f"  given {label}: {a[0]:.0%} without memory, {b[0]:.0%} with it "
                      f"(all 32 exact: {a[1]:.0%} -> {b[1]:.0%}; {a[2]} prompts)")
        if not args.val:
            # The held-out blocks were needed only to tune and to report honestly. The saved
            # memory should know the whole corpus, so build it again over all of the text.
            tuned = (memory.k, memory.temperature, memory.lam, memory.recall_temperature)
            memory = Memory.build(model, encode(read_text(args.data)), args.context, tok.eot_id)
            memory.k, memory.temperature, memory.lam, memory.recall_temperature = tuned
            print(f"rebuilt over the whole corpus: {len(memory):,} entries")
        memory.save(args.memory)
        print("saved", args.memory)
    else:
        memory = Memory.load(args.memory)

    for prompt in args.ask:
        # Start the way documents start in the corpus: the separator, a newline, then the text.
        x = torch.tensor([tok.encode("<|endoftext|>\n", allow_special=True) + tok.encode(prompt)])
        for name, hook in (("model only ", None), ("with memory", memory.mix)):
            out = model.generate(x, max_new_tokens=60, temperature=0.0, eos_token_id=tok.eot_id, logit_hook=hook)
            print(f"{name}: {tok.decode(out[0].tolist())!r}")


if __name__ == "__main__":
    main()
