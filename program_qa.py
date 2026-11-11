"""
program_qa.py: weight-space programming, tried on MongolHistorySLM
====================================================================

Nothing in this file is trained: there is no loss, no gradient and no optimizer. The
script writes every weight of a MongolHistorySLM (the class in model.py, with a one-layer
config) so that the model answers the questions in your Q&A file word for word.

Where the idea comes from
    "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang and Khattab, MIT CSAIL,
    2 Nov 2026). A frontier LLM reads a network's raw weights and writes new values
    directly, with no gradient descent. Here Claude played the weight programmer: the
    program below was designed by hand and then compiled into tensors.

The program, in one transformer layer
    embedding   Each token's vector is split into named blocks:
                  const      a large constant. It makes RMSNorm a fixed rescale, and it
                             gives the bias-free layers a bias input.
                  tok        a random 128-d code that answers "which token is this?"
                  qw         question-word code: the word's stem code times its IDF
                  is_sep, is_answer_id, answer_id   flags
    attention   Each head copies one block into another block:
                  BAG        q = 0, so every logit is equal and attention averages qw over
                             the prefix. At <sep> that average is a TF-IDF fingerprint of
                             the question.
                  PREV k     constant q and k, placed so that RoPE peaks exactly k tokens
                             back. It copies that token's code (n-gram context).
                  ANSWER     the key reads the is_answer_id flag, so the head copies which
                             answer is being recited.
    lm_head     At <sep>: one row per stored question holds that question's fingerprint.
                The logits are similarities, so softmax picks the closest question. The
                model "says" an answer-id token such as <q17>, or <q?> for "I don't know".
    FFN         One SwiGLU neuron per answer token. Each neuron is a rule:
                  IF answer = 17 AND the context ends with " хаан", " 1162"
                  THEN write the code of " онд"
                The gate holds the IF (a thresholded match score); down_proj holds the THEN.
    lm_head     After <sep>: turns that code back into the next token.

Usage
    python program_qa.py                       # data/qa.* if it exists, else the sample
    python program_qa.py --data data/qa.csv    # .csv .tsv .xlsx .json .jsonl
    python program_qa.py --ask "Чингис хаан хэдэн онд төрсөн бэ?"
    python program_qa.py --chat
    python program_qa.py --explain 3           # read answer #3's rules back from the weights
    python program_qa.py --save out/programmed_qa.pt   # an ordinary model.py checkpoint
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from model import ModelConfig, MongolHistorySLM

DONT_KNOW = "Уучлаарай, би энэ асуултын хариултыг мэдэхгүй байна."
SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
STEM_LEN = 4
SAMPLE = Path(__file__).resolve().parent / "examples" / "sample_qa.csv"


# ----------------------------------------------------------------------------------------
# 1. Loading the Q&A file
# ----------------------------------------------------------------------------------------
Q_KEYS = ("question", "q", "асуулт", "prompt", "input", "instruction", "query")
A_KEYS = ("answer", "a", "хариулт", "response", "output", "completion", "reply")


def _find_col(names: Sequence, keys: Sequence[str]) -> Optional[int]:
    names = [str(n or "").strip().lower() for n in names]
    return next((names.index(k) for k in keys if k in names), None)


def _rows_to_pairs(rows: List[list]) -> List[Tuple[str, str]]:
    """Table rows -> (question, answer). A header row is used if it names the columns
    (question/answer, асуулт/хариулт, ...); otherwise the first two columns are used."""
    rows = [["" if c is None else str(c) for c in r] for r in rows]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    qi, ai = _find_col(rows[0], Q_KEYS), _find_col(rows[0], A_KEYS)
    if qi is not None and ai is not None:
        rows = rows[1:]
    else:
        qi, ai = 0, 1
    return [(r[qi], r[ai]) for r in rows if len(r) > max(qi, ai)]


def _item_to_pair(item) -> Optional[Tuple[str, str]]:
    if isinstance(item, dict) and isinstance(item.get("messages"), list):   # chat format
        msgs = item["messages"]
        q = next((m.get("content") for m in msgs if m.get("role") == "user"), None)
        a = next((m.get("content") for m in msgs if m.get("role") == "assistant"), None)
        return (q, a) if q and a else None
    if isinstance(item, dict):
        keys = list(item)
        qi, ai = _find_col(keys, Q_KEYS), _find_col(keys, A_KEYS)
        return None if qi is None or ai is None else (str(item[keys[qi]]), str(item[keys[ai]]))
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    return None


def load_qa(path: Path) -> List[Tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        import openpyxl  # pip install openpyxl
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
        raw = _rows_to_pairs([list(r) for r in sheet.iter_rows(values_only=True)])
    elif suffix in (".json", ".jsonl"):
        text = path.read_text(encoding="utf-8-sig")
        if suffix == ".jsonl":
            items = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            items = json.loads(text)
            if isinstance(items, dict):                     # {"data": [...]} and similar
                items = next((v for v in items.values() if isinstance(v, list)), [])
        raw = [p for p in map(_item_to_pair, items) if p]
    else:                                                   # .csv .tsv .txt
        text = path.read_text(encoding="utf-8-sig")
        best = None
        for delim in (["\t"] if suffix == ".tsv" else [",", ";", "\t", "|"]):
            rows = list(csv.reader(io.StringIO(text), delimiter=delim))
            good = sum(1 for r in rows if sum(1 for c in r if c.strip()) >= 2)
            if best is None or good > best[0]:
                best = (good, rows)
        raw = _rows_to_pairs(best[1])
    pairs, seen = [], set()
    for q, a in raw:
        q, a = q.replace("\r\n", "\n").strip(), a.replace("\r\n", "\n").strip()
        if q and a and (q, a) not in seen:                  # drop empty rows and exact repeats
            seen.add((q, a))
            pairs.append((q, a))
    return pairs


# ----------------------------------------------------------------------------------------
# 2. Tokenizer: word-level, lossless for the stored texts
# ----------------------------------------------------------------------------------------
# GPT-2-style pieces: a word or number with its leading space, punctuation, whitespace.
PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\w\s]+|\s+(?!\S)|\s+|.", re.DOTALL)


def stem(piece: str) -> Optional[str]:
    """The retrieval key of a piece: lowercase, first 4 letters. This is a crude Mongolian
    stemmer: Чингис, Чингисийн and Чингисээс all become "чинг". Numbers stay whole, and
    whitespace and punctuation have no key."""
    w = piece.strip().lower()
    if not any(ch.isalnum() for ch in w):
        return None
    return w if w.isdigit() else w[:STEM_LEN]


class QATokenizer:
    """Ids: the 5 specials, then one answer-id token per answer (<q0>, <q1>, ..., and <q?>
    for "I don't know"), then every piece seen in the data, most frequent first."""

    def __init__(self, itos: List[str], n_answers: int):
        self.itos = itos
        self.stoi = {p: i for i, p in enumerate(itos)}
        self.n_answers = n_answers                  # stored answers + 1 ("I don't know")
        self.pad, self.bos, self.sep, self.eos, self.unk = range(len(SPECIALS))
        self.first_answer_id = len(SPECIALS)
        self.first_piece = self.first_answer_id + n_answers
        # Fallbacks for questions a person types (stored texts never need them).
        self.by_norm: Dict[str, int] = {}           # "чингис" -> " Чингис"
        self.by_stem: Dict[str, int] = {}           # "чинг"   -> " Чингис"
        for i in range(self.first_piece, len(itos)):
            if itos[i].strip():
                self.by_norm.setdefault(itos[i].strip().lower(), i)
            if stem(itos[i]):
                self.by_stem.setdefault(stem(itos[i]), i)

    @classmethod
    def build(cls, texts: List[str], n_answers: int) -> "QATokenizer":
        counts = Counter(p for t in texts for p in PRETOKEN.findall(t))
        pieces = sorted(counts, key=lambda p: (-counts[p], p))
        ids = [f"<q{j}>" for j in range(n_answers - 1)] + ["<q?>"]
        return cls(SPECIALS + ids + pieces, n_answers)

    def answer_token(self, j: int) -> int:
        return self.first_answer_id + j

    def encode(self, text: str) -> List[int]:
        """Exact encoding. Every piece must be known, which holds for the stored Q&A."""
        return [self.stoi[p] for p in PRETOKEN.findall(text)]

    def encode_question(self, text: str) -> List[int]:
        """Lossy encoding for a typed question. An unknown word becomes the same word in
        another case, else a known word with the same stem, else <unk>. Unknown
        punctuation is dropped."""
        ids = []
        for p in PRETOKEN.findall(text):
            if p in self.stoi:
                ids.append(self.stoi[p])
            elif p.strip().lower() in self.by_norm:
                ids.append(self.by_norm[p.strip().lower()])
            elif stem(p) is not None:
                ids.append(self.by_stem.get(stem(p), self.unk))
        return ids

    def decode(self, ids) -> str:
        return "".join(self.itos[i] for i in ids if i >= self.first_piece)


# ----------------------------------------------------------------------------------------
# 3. The compiler: Q&A -> weights
# ----------------------------------------------------------------------------------------
@dataclass
class Program:
    """Knobs of the weight program. The defaults work; compile_qa() and check_recall()
    verify them on your data."""
    tok_dim: int = 128           # random code per token (tok, prev-k and out blocks)
    qw_dim: int = 256            # code per word stem (retrieval)
    head_dim: int = 128
    const: float = 1000.0        # the constant channel: dominates rms(x), so RMSNorm is ~fixed
    answer_weight: float = 2.0   # in a rule's key, the answer counts 2 and each context token 1
    margin: float = 0.5          # a rule fires when its match score is within 0.5 of perfect
    rule_gain: float = 40.0      # an exact match gives silu(40 * 0.5) = 20, a miss ~0
    out_gain: float = 2.0        # lm_head gain on the out block: an exact rule gives a ~40 logit
    retrieval_gain: float = 100.0
    never: float = 1e4           # logit penalty for tokens that must not win
    head_gap: float = 20.0       # PREV heads: logit gap between the target and any other position
    answer_words: float = 0.3    # words of the answer also count for retrieval, at 0.3 weight
    unknown_weight: float = 1.0  # an unknown word in a typed question weighs this much times the
                                 # rarest known word, and points at "I don't know". Lower values
                                 # answer more often and guess more often.
    max_context: int = 32        # cap on the context a rule may use; the program uses only what
                                 # the data needs (PREV heads = longest context used - 1)
    seed: int = 0


@dataclass
class Compiled:
    model: MongolHistorySLM
    tok: QATokenizer
    pairs: List[Tuple[str, str]]
    blocks: Dict[str, slice]
    heads: List[Tuple[object, slice, slice]]
    rules: List[Tuple[int, Tuple[int, ...], int]]   # (answer, context tokens, next token)
    ambiguous: List[Tuple[int, int]]                # (answer, position) that max_context can't settle
    n_prev: int
    max_answer: int
    seconds: float


def rope_freqs(head_dim: int, theta: float) -> torch.Tensor:
    """The angular speeds model.RotaryEmbedding uses. Pair j = dims (j, j + head_dim/2)."""
    return 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)


def positional_profile(head_dim: int, theta: float, max_len: int) -> Tuple[float, torch.Tensor]:
    """How to spread a PREV head's q/k over the RoPE pairs.

    With energy p_j in pair j, the logit for "d positions away from the target" is
    proportional to sum_j p_j cos(d * w_j): the peak (1) is at d = 0. We try "uniform over
    the n fastest pairs" for several n and keep the profile whose worst drop below the peak
    (over every d up to max_len) is largest."""
    w = rope_freqs(head_dim, theta)
    d = torch.arange(1, max_len + 1, dtype=torch.float64)[:, None]
    best = None
    for n in (4, 6, 8, 12, 16, 24, 32, 48, 64):
        if n > head_dim // 2:
            continue
        p = torch.zeros(head_dim // 2, dtype=torch.float64)
        p[:n] = 1.0 / n
        drop = ((1 - torch.cos(d * w)) * p).sum(1).min().item()
        if best is None or drop > best[0]:
            best = (drop, p)
    return best


def rule_table(answers: List[List[int]], answer_token, eos: int, max_context: int):
    """One rule per answer token: (answer j, the shortest context that pins down the next
    token inside answer j) -> next token. Position 0 of every answer is its id token
    <qj>, which occurs only there, so a context that reaches back to it is always unique."""
    rules: Dict[Tuple[int, Tuple[int, ...]], int] = {}
    ambiguous = []
    for j, ans in enumerate(answers):
        seq = [answer_token(j)] + ans + [eos]
        nexts = [defaultdict(set) for _ in range(max_context + 1)]
        for k in range(len(seq) - 1):
            for m in range(1, min(k + 1, max_context) + 1):
                nexts[m][tuple(seq[k - m + 1:k + 1])].add(seq[k + 1])
        for k in range(len(seq) - 1):
            for m in range(1, min(k + 1, max_context) + 1):
                ctx = tuple(seq[k - m + 1:k + 1])
                if len(nexts[m][ctx]) == 1:
                    break
            else:
                ambiguous.append((j, k))
            rules.setdefault((j, ctx), seq[k + 1])
    return [(j, ctx, nxt) for (j, ctx), nxt in rules.items()], ambiguous


def compile_qa(pairs: List[Tuple[str, str]], prog: Program = Program()) -> Compiled:
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(prog.seed)
    N = len(pairs)                                  # answer index N = "I don't know"
    tok = QATokenizer.build([t for qa in pairs for t in qa] + [DONT_KNOW], N + 1)
    V, hd = len(tok.itos), prog.head_dim

    def unit(n: int, d: int) -> torch.Tensor:       # n random unit vectors
        x = torch.randn(n, d, generator=gen)
        return x / x.norm(dim=1, keepdim=True)

    # ---- codes --------------------------------------------------------------------------
    tok_code = unit(V, prog.tok_dim)                                    # "which token?"
    stems = sorted({s for p in tok.itos[tok.first_piece:] if (s := stem(p))})
    stem_code = dict(zip(stems, unit(len(stems), prog.qw_dim)))
    unknown_code = unit(1, prog.qw_dim)[0]                              # "I don't know"
    questions = [tok.encode(q) for q, _ in pairs]
    answers = [tok.encode(a) for _, a in pairs] + [tok.encode(DONT_KNOW)]
    df = Counter(s for ids in questions for s in {stem(tok.itos[i]) for i in ids} if s)
    qw = torch.zeros(V, prog.qw_dim)
    for i in range(tok.first_piece, V):
        if (s := stem(tok.itos[i])):
            qw[i] = (math.log((N + 1) / (df[s] + 1)) + 1.0) * stem_code[s]     # IDF x stem code
    # An unknown word weighs as much as the rarest known word, and points at "I don't know".
    qw[tok.unk] = prog.unknown_weight * (math.log(N + 1) + 1.0) * unknown_code
    # A stored question's fingerprint: its own words (unit length), plus its answer's distinct
    # words at 0.3 weight (also unit length), so a long answer can't drown out its question.
    unit_sum = lambda ids: (lambda v: v / v.norm().clamp_min(1e-9))(qw[ids].sum(0))
    fingerprint = torch.zeros(N + 1, prog.qw_dim)
    for j in range(N):
        v = unit_sum(questions[j]) + prog.answer_words * unit_sum(sorted(set(answers[j])))
        fingerprint[j] = v / v.norm().clamp_min(1e-9)
    fingerprint[N] = unknown_code

    rules, ambiguous = rule_table(answers, tok.answer_token, tok.eos, prog.max_context)
    n_prev = max(len(ctx) for _, ctx, _ in rules) - 1                  # PREV heads needed

    # ---- residual stream layout ---------------------------------------------------------
    blocks: Dict[str, slice] = {}
    for name, size in [("const", 1), ("is_sep", 1), ("is_answer_id", 1),
                       ("tok", prog.tok_dim), ("qw", prog.qw_dim), ("answer_id", N + 1),
                       ("bag", prog.qw_dim)] \
            + [(f"prev{k}", prog.tok_dim) for k in range(1, n_prev + 1)] \
            + [("answer", N + 1), ("out", prog.tok_dim)]:
        start = max((s.stop for s in blocks.values()), default=0)
        blocks[name] = slice(start, start + size)
    D = blocks["out"].stop
    c0 = blocks["const"].start

    heads: List[Tuple[object, slice, slice]] = []   # (pattern, source block, destination block)

    def copy_heads(pattern, src: slice, dst: slice) -> None:
        for off in range(0, src.stop - src.start, hd):  # a block wider than a head -> several heads
            w = min(hd, src.stop - src.start - off)
            heads.append((pattern, slice(src.start + off, src.start + off + w),
                          slice(dst.start + off, dst.start + off + w)))

    copy_heads("bag", blocks["qw"], blocks["bag"])
    for k in range(1, n_prev + 1):
        copy_heads(k, blocks["tok"], blocks[f"prev{k}"])
    copy_heads("answer", blocks["answer_id"], blocks["answer"])

    max_answer = max(map(len, answers))
    max_len = max(1024, 2 * (max(map(len, questions)) + max_answer + 4))
    cfg = ModelConfig(vocab_size=V, d_model=D, n_layers=1, n_heads=len(heads),
                      n_kv_heads=len(heads), head_dim=hd, ffn_hidden=len(rules),
                      max_seq_len=max_len, tie_embeddings=False, dropout=0.0)
    model = MongolHistorySLM(cfg).eval()
    blk = model.layers[0]

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # (1) Embedding: each token's fingerprint.
        E = model.embed_tokens.weight
        E[:, c0] = prog.const
        E[:, blocks["tok"]] = tok_code
        E[:, blocks["qw"]] = qw
        E[tok.sep, blocks["is_sep"].start] = 1.0
        for j in range(N + 1):
            E[tok.answer_token(j), blocks["is_answer_id"].start] = 1.0
            E[tok.answer_token(j), blocks["answer_id"].start + j] = 1.0

        # (2) RMSNorm gains. The constant dominates, so rms(x) ~ const / sqrt(D) at every
        #     position. A gain of const / sqrt(D) hands each layer the raw features, and the
        #     constant channel arrives as ~1.0: the "bias" input.
        gain = torch.full((D,), prog.const / math.sqrt(D))
        gain[c0] = 1.0 / math.sqrt(D)
        for norm in (blk.attn_norm, blk.ffn_norm, model.norm):
            norm.weight.copy_(gain)

        # (3) Attention. q/k/v rows of head h are [h*hd, (h+1)*hd) inside each third of qkv_proj.
        drop, profile = positional_profile(hd, cfg.rope_theta, max_len)
        g = math.sqrt(prog.head_gap / (math.sqrt(hd) * drop))  # QK-Norm gain, shared by all heads
        blk.attn.q_norm.weight.fill_(g)
        blk.attn.k_norm.weight.fill_(g)
        W, Wo = blk.attn.qkv_proj.weight, blk.attn.o_proj.weight
        Hq = len(heads) * hd
        amp = (profile * hd).sqrt()                      # RMS 1, so QK-Norm leaves it unchanged
        w_rope = rope_freqs(hd, cfg.rope_theta)
        for h, (pattern, src, dst) in enumerate(heads):
            q0, k0, v0 = h * hd, Hq + h * hd, 2 * Hq + h * hd
            idx = torch.arange(src.stop - src.start)
            W[v0 + idx, src.start + idx] = 1.0           # value = the source block
            Wo[dst.start + idx, h * hd + idx] = 1.0      # head output -> destination block
            if pattern == "bag":
                pass                                     # q = 0: equal logits, uniform average
            elif pattern == "answer":
                slow = hd // 2 - 1                       # slowest RoPE pair: ~no rotation
                W[q0 + slow, c0] = 1.0                   # query: always the same
                W[k0 + slow, blocks["is_answer_id"].start] = 1.0   # key: only <qj> tokens
            else:                                        # PREV k: q = (a_j, 0), k = q turned by k*w_j
                ang = pattern * w_rope
                W[q0:q0 + hd // 2, c0] = amp.float()
                W[k0:k0 + hd // 2, c0] = (amp * torch.cos(ang)).float()
                W[k0 + hd // 2:k0 + hd, c0] = (amp * torch.sin(ang)).float()

        # (4) FFN: one neuron per rule. gate = rule_gain * (match score - threshold), up = 1.
        Wg, Wd = blk.ffn.gate_up_proj.weight, blk.ffn.down_proj.weight
        H, beta = len(rules), prog.rule_gain
        for n, (j, ctx, nxt) in enumerate(rules):
            perfect = prog.answer_weight + len(ctx)
            Wg[n, blocks["answer"].start + j] = beta * prog.answer_weight
            Wg[n, blocks["tok"]] = beta * tok_code[ctx[-1]]
            for back in range(1, len(ctx)):
                Wg[n, blocks[f"prev{back}"]] = beta * tok_code[ctx[-1 - back]]
            Wg[n, c0] = -beta * (perfect - prog.margin)
            Wg[H + n, c0] = 1.0
            Wd[blocks["out"], n] = tok_code[nxt]

        # (5) lm_head. Text tokens read the out block; answer-id tokens read the bag block
        #     and can only win at <sep>; specials never win.
        L, sep_col = model.lm_head.weight, blocks["is_sep"].start
        for rows in (slice(tok.first_piece, V), slice(tok.eos, tok.eos + 1)):
            L[rows, blocks["out"]] = prog.out_gain * tok_code[rows]
            L[rows, sep_col] = -prog.never
        for j in range(N + 1):
            r = tok.answer_token(j)
            L[r, blocks["bag"]] = prog.retrieval_gain * fingerprint[j]
            L[r, sep_col] = prog.never
            L[r, c0] = -prog.never
        L[tok.answer_token(N), c0] += 1.0        # no evidence at all -> "I don't know"
        for s in (tok.pad, tok.bos, tok.sep, tok.unk):
            L[s, c0] = -prog.never

    return Compiled(model, tok, pairs, blocks, heads, rules, ambiguous, n_prev, max_answer,
                    time.perf_counter() - t0)


# ----------------------------------------------------------------------------------------
# 4. Checks, retrieval and generation
# ----------------------------------------------------------------------------------------
@torch.inference_mode()
def check_recall(c: Compiled):
    """Teacher-forced exactness test. Feed  <bos> question <sep> <qj> answer <eos>  and check
    the argmax at every position from <sep> on. If every one is right, greedy decoding
    reproduces the stored answer exactly. Returns (failures, worst logit margin)."""
    tok = c.tok
    seqs = []
    for j, (q, a) in enumerate(c.pairs):
        qi = tok.encode(q)
        seqs.append((j, [tok.bos] + qi + [tok.sep, tok.answer_token(j)] + tok.encode(a) + [tok.eos],
                     len(qi) + 1))
    seqs.sort(key=lambda s: len(s[1]))
    hidden = c.model.config.ffn_hidden
    failures, worst, i = [], float("inf"), 0
    while i < len(seqs):
        T = len(seqs[min(i + 63, len(seqs) - 1)][1])
        B = max(1, min(64, int(3e7 // (T * 2 * hidden))))   # bound the FFN activation memory
        chunk = seqs[i:i + B]
        i += B
        T = max(len(x) for _, x, _ in chunk)
        inp = torch.full((len(chunk), T), tok.pad)
        for b, (_, x, _) in enumerate(chunk):
            inp[b, :len(x)] = torch.tensor(x)
        logits, _ = c.model(inp)
        for b, (j, x, s) in enumerate(chunk):
            lg = logits[b, s:len(x) - 1]
            tgt = torch.tensor(x[s + 1:])
            right = lg.gather(1, tgt[:, None]).squeeze(1)
            margins = right - lg.scatter(1, tgt[:, None], float("-inf")).max(1).values
            worst = min(worst, margins.min().item())
            bad = (margins <= 0).nonzero().flatten()
            if len(bad):
                p = int(bad[0])
                failures.append((j, p, tok.itos[x[s + 1 + p]], tok.itos[int(lg[p].argmax())]))
    return sorted(failures), worst


@torch.inference_mode()
def pick_answers(c: Compiled, questions: List[str]) -> List[torch.Tensor]:
    """Retrieval only: the softmax over answer-id tokens at <sep>, for each question."""
    tok, N = c.tok, len(c.pairs)
    prompts = [[tok.bos] + tok.encode_question(q) + [tok.sep] for q in questions]
    out = []
    for i in range(0, len(prompts), 32):
        chunk = prompts[i:i + 32]
        x = torch.full((len(chunk), max(map(len, chunk))), tok.pad)
        for b, p in enumerate(chunk):
            x[b, :len(p)] = torch.tensor(p)
        h = c.model.forward_hidden(x)[torch.arange(len(chunk)), [len(p) - 1 for p in chunk]]
        lg = c.model.lm_head(h)[:, tok.first_answer_id:tok.first_answer_id + N + 1]
        out += list(lg.softmax(-1))
    return out


@torch.inference_mode()
def ask(c: Compiled, question: str) -> Tuple[int, str, List[Tuple[int, float]]]:
    """Plain greedy generation with model.generate(). The first token it emits is the
    answer id it chose; the rest is the answer. Returns (answer index, text, top-3 ids)."""
    tok = c.tok
    prompt = torch.tensor([[tok.bos] + tok.encode_question(question) + [tok.sep]])
    out = c.model.generate(prompt, max_new_tokens=c.max_answer + 2, temperature=0.0,
                           eos_token_id=tok.eos)[0, prompt.shape[1]:].tolist()
    probs = pick_answers(c, [question])[0]
    top = probs.topk(min(3, len(probs)))
    return out[0] - tok.first_answer_id, tok.decode(out[1:]), \
        [(int(i), float(p)) for p, i in zip(top.values, top.indices)]


def perturbations(q: str, rng: random.Random) -> Dict[str, str]:
    words = q.split()
    out = {"exact question": q,
           "lowercase, no punctuation": re.sub(r"[^\w\s]", "", q).lower()}
    shuffled = words[:]
    rng.shuffle(shuffled)
    out["words shuffled"] = " ".join(shuffled)
    long = [i for i, w in enumerate(words) if sum(ch.isalpha() for ch in w) >= 4]
    if long:
        w = words[:]
        i = rng.choice(long)
        w[i] = re.sub(r"[^\w]+$", "", w[i]) + "ийн"
        out["suffix added to a word"] = " ".join(w)
    if len(words) >= 3:
        w = words[:]
        del w[rng.randrange(len(w))]
        out["one word dropped"] = " ".join(w)
    return out


OUT_OF_SCOPE = [
    "Өнөөдөр цаг агаар ямар байна?",
    "Хоёр дээр хоёрыг нэмбэл хэд болох вэ?",
    "Бууз яаж хийдэг вэ?",
    "Футболын дэлхийн аварга аль улс вэ?",
    "Наполеон хэдэн онд төрсөн бэ?",
    "What is the capital of France?",
]


def run_probes(c: Compiled, seed: int = 0) -> List[str]:
    rng, N = random.Random(seed), len(c.pairs)
    tests: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for j, (q, _) in enumerate(c.pairs):
        for kind, text in perturbations(q, rng).items():
            tests[kind].append((text, j))
    tests['out of scope -> "I don\'t know"'] = [(q, N) for q in OUT_OF_SCOPE]
    lines = []
    for kind, items in tests.items():
        picks = [int(p.argmax()) for p in pick_answers(c, [t for t, _ in items])]
        ok = sum(p == j for p, (_, j) in zip(picks, items))
        lines.append(f"  {kind:<34} {ok:>4}/{len(items):<4} ({ok / len(items):.0%})")
    return lines


# ----------------------------------------------------------------------------------------
# 5. Reading the weights back (the "read" half of weight-space programming)
# ----------------------------------------------------------------------------------------
def _block_of(blocks: Dict[str, slice], index: int) -> str:
    return next((n for n, s in blocks.items() if s.start <= index < s.stop), "?")


@torch.inference_mode()
def describe_heads(c: Compiled) -> List[str]:
    """What each attention head does, worked out from its weights alone."""
    attn, cfg = c.model.layers[0].attn, c.model.config
    hd, Hq, c0 = cfg.head_dim, cfg.n_heads * cfg.head_dim, c.blocks["const"].start
    W, w = attn.qkv_proj.weight, rope_freqs(cfg.head_dim, cfg.rope_theta)
    lines = []
    for h in range(cfg.n_heads):
        q, k = W[h * hd:(h + 1) * hd], W[Hq + h * hd:Hq + (h + 1) * hd]
        v, o = W[2 * Hq + h * hd:2 * Hq + (h + 1) * hd], attn.o_proj.weight[:, h * hd:(h + 1) * hd]
        src = _block_of(c.blocks, int(v.abs().sum(0).argmax()))
        dst = _block_of(c.blocks, int(o.abs().sum(1).argmax()))
        if q.abs().sum() == 0:
            what = "q = 0: all logits equal, so it averages the whole prefix"
        elif k[:, c0].abs().sum() == 0:
            key = _block_of(c.blocks, int(k.abs().sum(0).argmax()))
            what = f"key reads '{key}': it attends to the token that has that flag"
        else:   # constant q and k: the logit depends only on distance, through RoPE
            qv, kv = q[:, c0].double(), k[:, c0].double()
            q1, q2, k1, k2 = qv[:hd // 2], qv[hd // 2:], kv[:hd // 2], kv[hd // 2:]
            d = torch.arange(0, 33, dtype=torch.float64)[:, None]
            logit = ((q1 * k1 + q2 * k2) * torch.cos(d * w) + (q1 * k2 - q2 * k1) * torch.sin(d * w)).sum(1)
            what = f"constant q and k; RoPE puts the peak {int(logit.argmax())} token(s) back"
        lines.append(f"  head {h:>2}: {src:>9} -> {dst:<7} {what}")
    return lines


@torch.inference_mode()
def describe_rules(c: Compiled, j: int, limit: int = 20) -> List[str]:
    """Decompile the FFN neurons that belong to answer j. Token codes are read back from
    the embedding, the rule's answer from its gate weights, the output from down_proj."""
    b, tok = c.blocks, c.tok
    codes = c.model.embed_tokens.weight[:, b["tok"]]
    near = lambda v: None if v.norm() < 1e-6 else tok.itos[int((codes @ v).argmax())]
    ffn = c.model.layers[0].ffn
    H = ffn.down_proj.weight.shape[1]
    gates = ffn.gate_up_proj.weight[:H]
    mine = (gates[:, b["answer"]].argmax(1) == j).nonzero().flatten().tolist()
    lines = []
    for n in mine[:limit]:
        row = gates[n]
        ctx = [near(row[b[f"prev{k}"]]) for k in range(c.n_prev, 0, -1)] + [near(row[b["tok"]])]
        nxt = near(ffn.down_proj.weight[b["out"], n])
        lines.append(f"  neuron {n:>6}: IF answer = {j} AND context ends with "
                     f"{[t for t in ctx if t is not None]} THEN next = {nxt!r}")
    if len(mine) > limit:
        lines.append(f"  ... and {len(mine) - limit} more neurons for this answer")
    return lines


# ----------------------------------------------------------------------------------------
# 6. Command line
# ----------------------------------------------------------------------------------------
def default_data() -> Path:
    data = Path(__file__).resolve().parent / "data"
    for name in ("qa.csv", "qa.tsv", "qa.xlsx", "qa.json", "qa.jsonl"):
        if (data / name).exists():
            return data / name
    found = sorted(p for p in data.glob("*") if p.suffix.lower() in (".csv", ".tsv", ".xlsx", ".json", ".jsonl"))
    return found[0] if len(found) == 1 else SAMPLE


def show(c: Compiled, question: str) -> None:
    j, text, top = ask(c, question)
    N = len(c.pairs)
    label = lambda i: '"I don\'t know"' if i == N else f"#{i} {c.pairs[i][0]!r}"
    print(f"\nQ: {question}")
    print("   closest stored questions: " + "; ".join(f"{label(i)} p={p:.2f}" for i, p in top))
    print(f"A: {text}")


def main() -> None:
    for stream in (sys.stdout, sys.stdin):          # Mongolian text on a Windows console
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Write MongolHistorySLM weights that answer a Q&A file. No training.")
    ap.add_argument("--data", type=Path, help="Q&A file (.csv .tsv .xlsx .json .jsonl); default data/qa.* or the sample")
    ap.add_argument("--ask", action="append", default=[], help="a question to answer (repeatable)")
    ap.add_argument("--chat", action="store_true", help="type questions interactively")
    ap.add_argument("--explain", type=int, metavar="J", help="decompile answer J's rules from the weights")
    ap.add_argument("--save", type=Path, help="save the programmed checkpoint, e.g. out/programmed_qa.pt")
    ap.add_argument("--max-context", type=int, default=32, help="cap on the context (tokens) a rule may use")
    ap.add_argument("--unknown-weight", type=float, default=1.0,
                    help="how strongly unknown words push toward \"I don't know\" (lower = guesses more)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    path = args.data or default_data()
    pairs = load_qa(path)
    if not pairs:
        sys.exit(f"no question/answer pairs found in {path}")
    print(f"{len(pairs)} Q&A pairs from {path}" + ("  (the built-in sample: put your file in data/)" if path == SAMPLE else ""))
    groups = defaultdict(set)
    for q, a in pairs:
        groups[" ".join(re.findall(r"\w+", q.lower()))].add(a)
    clashes = [q for q, ans in groups.items() if len(ans) > 1]
    if clashes:
        print(f"  note: {len(clashes)} question(s) appear with different answers; only one answer can win each")

    c = compile_qa(pairs, Program(seed=args.seed, max_context=args.max_context,
                                  unknown_weight=args.unknown_weight))
    cfg, n_params = c.model.config, c.model.num_parameters()
    print(f"\nProgrammed a MongolHistorySLM in {c.seconds:.2f} s with 0 gradient steps:")
    print(f"  vocab {cfg.vocab_size:,} | d_model {cfg.d_model:,} | {cfg.n_layers} layer | "
          f"{cfg.n_heads} heads x {cfg.head_dim} | FFN {cfg.ffn_hidden:,} neurons, one rule each")
    print(f"  {n_params / 1e6:.1f}M parameters ({n_params * 4 / 2**20:.0f} MB in fp32); "
          f"longest rule context: {c.n_prev + 1} tokens")
    if c.ambiguous:
        print(f"  warning: {len(c.ambiguous)} answer positions repeat more than {args.max_context} "
              f"tokens of context with different continuations; try --max-context {2 * args.max_context}")

    t = time.perf_counter()
    failures, worst = check_recall(c)
    print(f"\nRecall check (every answer token, teacher-forced): {len(pairs) - len({f[0] for f in failures})}"
          f"/{len(pairs)} answers exact, worst logit margin {worst:.1f}  [{time.perf_counter() - t:.1f} s]")
    for j, p, want, got in failures[:10]:
        if p == 0:
            k = c.tok.stoi[got] - c.tok.first_answer_id
            other = '"I don\'t know"' if k == len(pairs) else f"#{k} {pairs[k][0]!r}"
            print(f"  question #{j} {pairs[j][0]!r} retrieves {other} instead")
        else:
            print(f"  answer #{j}: recitation goes wrong at token {p}: expected {want!r}, model says {got!r}")

    print("\nRetrieval probes (does it pick the right stored question?):")
    print("\n".join(run_probes(c, args.seed)))

    print("\nWhat the attention heads do, read back from the weights:")
    print("\n".join(describe_heads(c)))

    if args.explain is not None:
        j = args.explain
        q, a = c.pairs[j] if j < len(pairs) else ("(anything it doesn't know)", DONT_KNOW)
        print(f"\nAnswer #{j}: Q: {q!r}\n           A: {a!r}")
        print("\n".join(describe_rules(c, j)))

    questions = args.ask or ([] if args.chat else [q for q, _ in random.Random(args.seed).sample(pairs, min(3, len(pairs)))]
                             + ["Наполеон хэдэн онд төрсөн бэ?"])
    for q in questions:
        show(c, q)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "model": c.model.state_dict(),
                    "tokenizer": {"itos": c.tok.itos, "n_answers": c.tok.n_answers}}, args.save)
        print(f"\nsaved {args.save} (load it with MongolHistorySLM(ModelConfig(**ckpt['config'])))")

    if args.chat:
        print('\nAsk anything (empty line to quit). It can only answer what it was programmed with.')
        while True:
            try:
                q = input("\nАсуулт> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            show(c, q)


if __name__ == "__main__":
    main()
