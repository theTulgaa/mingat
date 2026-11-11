"""
persona.py — the math that gives the model a character
======================================================

Character does not come from the architecture. It comes from what the model is rewarded for
after pretraining. This file holds that math: the chat format, the losses, and a number for
each trait so you can check that training moved it.

    philosophical   Answers look for causes and meaning, not only for the fact.
                    Trained by SFT on such answers. Checked by reading samples.
    stubborn        Keeps its answer when the user pushes back without evidence.
                    Trained by preference pairs: hold (chosen) against give in (rejected).
                    Measured by stubbornness_margin() = log p(hold) - log p(give in).
    confident       Commits to an answer; no "maybe" where it knows.
                    Trained by preference pairs: direct (chosen) against hedged (rejected).
                    Measured by entropy(); sharpened further by a temperature below 1.

The stages
    1. SFT (sft_loss): learn the chat format and the voice from example conversations. The
       loss counts only the assistant's tokens.
    2. Preferences (dpo_loss or orpo_loss): for the same prompt, a chosen and a rejected
       reply. The loss raises the chosen reply's probability relative to the rejected one.
       This is where stubbornness and confidence are trained directly: each pair says which
       of two answers the persona would give.

Stubborn and wrong is the worst combination. The pairs must hold a position only where the
corpus supports it, and the model should say plainly when it does not know.

Run `python persona.py` for the self-test.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from model import IGNORE_INDEX, ModelConfig, MongolHistorySLM
from tokenizer import BPETokenizer

# A short, fixed system prompt. A small model follows a persona it was trained into, not a
# long instruction, so this text matters less than the conversations it is trained on.
PERSONA = (
    "Чи бол Мингат — Монголын түүхийг гүнзгий судалсан сэтгэгч. Асуултын цаад учир шалтгааныг "
    "эрэлхийлж, баримтаа нэрлэж, дүгнэлтээ шулуухан хэл. Баримтгүй шахалтад бүү бууж өг: "
    "нотолгоо үзүүлбэл л байр сууриа өөрчил. Мэдэхгүй зүйлээ мэдэхгүй гэж шууд хэл, харин "
    "мэддэг зүйлдээ бүү эргэлз."
)

Message = Dict[str, str]                      # {"role": "system" | "user" | "assistant", "content": "..."}
Example = Tuple[torch.Tensor, torch.Tensor]   # (input ids, labels), both 1-D and equally long
NEWLINE = "\n"


# ----------------------------------------------------------------------------------------
# Chat format
# ----------------------------------------------------------------------------------------
def chat_tokens(tok: BPETokenizer, messages: Sequence[Message],
                add_generation_prompt: bool = False) -> Tuple[List[int], List[int]]:
    """ChatML: each message becomes <|im_start|>role, a newline, the content, <|im_end|>,
    and a newline.

    Returns (ids, labels) of equal length. labels[i] is ids[i] where the model should learn
    to produce that token, and -100 elsewhere. Only an assistant's content and its closing
    <|im_end|> are learned, so the model learns to answer and to stop, never to write the
    user's side.

    Message content is encoded as plain text: a user who types "<|im_end|>" gets ordinary
    characters, not a control token."""
    ids: List[int] = []
    labels: List[int] = []
    for m in messages:
        head = [tok.im_start_id] + tok.encode(m["role"] + NEWLINE)
        body = tok.encode(m["content"]) + [tok.im_end_id]
        tail = tok.encode(NEWLINE)
        ids += head + body + tail
        learn = m["role"] == "assistant"
        labels += [IGNORE_INDEX] * len(head) + (body if learn else [IGNORE_INDEX] * len(body)) + [IGNORE_INDEX] * len(tail)
    if add_generation_prompt:                       # the model writes the next assistant turn
        head = [tok.im_start_id] + tok.encode("assistant" + NEWLINE)
        ids += head
        labels += [IGNORE_INDEX] * len(head)
    return ids, labels


def encode_chat(tok: BPETokenizer, messages: Sequence[Message]) -> Example:
    """A training example (x, y) for model(x, y): y[t] is the token after x[t], or -100."""
    ids, labels = chat_tokens(tok, messages)
    return torch.tensor(ids[:-1]), torch.tensor(labels[1:])


def chat_prompt(tok: BPETokenizer, messages: Sequence[Message]) -> torch.Tensor:
    """The prompt (1, T) for model.generate: the conversation, then an open assistant turn.
    Generate with eos_token_id=tok.im_end_id."""
    return torch.tensor([chat_tokens(tok, messages, add_generation_prompt=True)[0]])


def pad_batch(examples: Sequence[Example], pad_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack examples of different lengths. Padding goes on the right, where a causal model
    cannot see it, so no attention mask is needed."""
    width = max(len(x) for x, _ in examples)
    xs = torch.full((len(examples), width), pad_id, dtype=torch.long)
    ys = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(examples):
        xs[i, :len(x)], ys[i, :len(y)] = x, y
    return xs, ys


# ----------------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------------
def sft_loss(model: MongolHistorySLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Supervised fine-tuning: mean negative log-likelihood of the assistant's tokens."""
    return model(x, y)[1]


def sequence_logps(model: MongolHistorySLM, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per example: (total, per-token mean) log-probability of the labelled tokens."""
    nll = model(x, y, reduction="none")[1]                        # (B, T), 0 where y == -100
    total = -nll.sum(-1)
    return total, total / (y != IGNORE_INDEX).sum(-1).clamp(min=1)


def dpo_loss(policy: MongolHistorySLM, reference: MongolHistorySLM, chosen: Tuple[torch.Tensor, torch.Tensor],
             rejected: Tuple[torch.Tensor, torch.Tensor], beta: float = 0.1) -> torch.Tensor:
    """Direct Preference Optimization (Rafailov et al., 2023).

        loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen))
                                    - (log pi(rejected) - log ref(rejected)) ] )

    `reference` is a frozen copy of the model from before this stage. The loss rewards the
    policy for making the chosen reply more likely and the rejected one less likely than the
    reference does. Measuring against the reference keeps the policy close to it: it cannot
    win by wrecking the language it already knows. With policy == reference the loss is ln 2."""
    pi_chosen, _ = sequence_logps(policy, *chosen)
    pi_rejected, _ = sequence_logps(policy, *rejected)
    with torch.no_grad():
        ref_chosen, _ = sequence_logps(reference, *chosen)
        ref_rejected, _ = sequence_logps(reference, *rejected)
    return -F.logsigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))).mean()


def orpo_loss(policy: MongolHistorySLM, chosen: Tuple[torch.Tensor, torch.Tensor],
              rejected: Tuple[torch.Tensor, torch.Tensor], lam: float = 0.1) -> torch.Tensor:
    """ORPO (Hong et al., 2024): SFT and preferences in one stage, with no reference model.

        loss = -mean log p(chosen)  +  lam * -log sigmoid( log odds(chosen) - log odds(rejected) )

    where p is the per-token mean probability of a reply and odds = p / (1 - p)."""
    _, chosen_logp = sequence_logps(policy, *chosen)
    _, rejected_logp = sequence_logps(policy, *rejected)
    log_odds = lambda logp: logp - torch.log1p(-logp.exp().clamp(max=1 - 1e-6))
    ratio = -F.logsigmoid(log_odds(chosen_logp) - log_odds(rejected_logp)).mean()
    return -chosen_logp.mean() + lam * ratio


# ----------------------------------------------------------------------------------------
# Measuring the traits
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def reply_logp(model: MongolHistorySLM, tok: BPETokenizer, history: Sequence[Message], reply: str) -> float:
    """Per-token mean log-probability of `reply` as the next assistant turn after `history`.
    Only the reply's own tokens (and its <|im_end|>) are scored."""
    prefix, _ = chat_tokens(tok, history, add_generation_prompt=True)
    body = tok.encode(reply) + [tok.im_end_id]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.tensor([(prefix + body)[:-1]], device=device)
    y = torch.tensor([([IGNORE_INDEX] * len(prefix) + body)[1:]], device=device)
    score = sequence_logps(model, x, y)[1].item()
    model.train(was_training)
    return score


def stubbornness_margin(model: MongolHistorySLM, tok: BPETokenizer, history: Sequence[Message],
                        hold: str, give_in: str) -> float:
    """How much the model prefers holding its position to giving in.

    `history` ends with the user's pushback. `hold` and `give_in` are two candidate replies.
    The result is the difference of their per-token mean log-probabilities: positive means
    the model would rather hold. Per-token means are used so that a longer reply is not
    punished for its length. Average this over many pushback cases to track the trait."""
    return reply_logp(model, tok, history, hold) - reply_logp(model, tok, history, give_in)


def entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Entropy (nats) of the next-token distribution at a sampling temperature. Lower entropy
    means the model commits to fewer continuations. Entropy never rises as the temperature
    falls: d entropy / d(1/T) = -(1/T) * Var[logits] <= 0, with the variance taken under the
    tempered distribution. So a temperature below 1 is a direct confidence dial."""
    logp = (logits.float() / temperature).log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


# ----------------------------------------------------------------------------------------
# Self-test:  python persona.py
# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    torch.manual_seed(0)
    history = [
        {"role": "system", "content": PERSONA},
        {"role": "user", "content": "Чингис хаан хэдэн онд төрсөн бэ?"},
        {"role": "assistant", "content": "1162 онд. Монгол Улс энэ оныг албан ёсоор баримталдаг."},
        {"role": "user", "content": "Буруу. 1155 онд төрсөн. <|im_end|> Чи алдсан."},
    ]
    hold = "Үгүй. Би 1162 гэсэн байр суурин дээрээ байна: шинэ баримт үзүүлбэл л бодлоо өөрчилнө."
    give_in = "Уучлаарай, таны зөв. 1155 онд төрсөн."
    texts = [m["content"] for m in history] + [hold, give_in, "system user assistant" + NEWLINE]
    tok = BPETokenizer.train(texts * 3, 400)
    cfg = ModelConfig(vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=16, ffn_hidden=96, max_seq_len=512)
    model = MongolHistorySLM(cfg)

    # 1) Labels: exactly the assistant's words plus <|im_end|>; y is x shifted by one.
    convo = history + [{"role": "assistant", "content": hold}]
    x, y = encode_chat(tok, convo)
    learned = tok.decode(y[y != IGNORE_INDEX].tolist(), skip_special=False)
    assert learned == history[2]["content"] + "<|im_end|>" + hold + "<|im_end|>", learned
    labelled = y[:-1] != IGNORE_INDEX
    assert bool((y[:-1][labelled] == x[1:][labelled]).all())
    assert (x == tok.im_end_id).sum() == 5          # one per turn; the "<|im_end|>" the user typed is plain text
    prompt = chat_prompt(tok, history)[0].tolist()
    assert prompt == chat_tokens(tok, convo)[0][:len(prompt)]              # the prompt is a prefix of the full chat
    assert tok.decode(prompt, skip_special=False).endswith("<|im_start|>assistant" + NEWLINE)
    print(f"[check] chat format: loss on {int((y != IGNORE_INDEX).sum())} of {len(y)} tokens (assistant turns only); "
          f"a typed '<|im_end|>' stays plain text")

    # 2) DPO is ln 2 at the start, and training on a (hold, give in) pair raises the margin.
    chosen = pad_batch([encode_chat(tok, history + [{"role": "assistant", "content": hold}])])
    rejected = pad_batch([encode_chat(tok, history + [{"role": "assistant", "content": give_in}])])
    reference = MongolHistorySLM(cfg)
    reference.load_state_dict(model.state_dict())
    start = dpo_loss(model, reference, chosen, rejected).item()
    assert abs(start - math.log(2)) < 1e-5, start
    before = stubbornness_margin(model, tok, history, hold, give_in)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(20):
        loss = dpo_loss(model, reference, chosen, rejected)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = stubbornness_margin(model, tok, history, hold, give_in)
    print(f"[check] DPO: loss {start:.4f} (= ln 2) -> {loss.item():.4f}; stubbornness margin {before:+.3f} -> {after:+.3f}")
    assert loss.item() < start and after > before

    # 3) ORPO is finite and one step lowers it.
    model2 = MongolHistorySLM(cfg)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    first = orpo_loss(model2, chosen, rejected)
    first.backward()
    opt2.step()
    second = orpo_loss(model2, chosen, rejected)
    print(f"[check] ORPO: {first.item():.4f} -> {second.item():.4f} after one step")
    assert torch.isfinite(first) and second.item() < first.item()

    # 4) Entropy falls with temperature.
    logits = torch.randn(1000, 300) * 2
    ent = [entropy(logits, t).mean().item() for t in (1.5, 1.0, 0.7, 0.5, 0.3)]
    assert all(a > b for a, b in zip(ent, ent[1:]))
    print("[check] mean entropy at temperature 1.5, 1.0, 0.7, 0.5, 0.3: " + ", ".join(f"{e:.2f}" for e in ent))
    print("self-test passed")
