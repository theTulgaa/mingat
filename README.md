# Mingat: MongolHistorySLM

A ~104M-parameter language model, trained from scratch, that specializes in the history of
Mongolia. This repo currently contains the architecture (`model.py`) and a weight-space
programming experiment (`program_qa.py`). The tokenizer, data pipeline and training scripts
come next.

```bash
python model.py   # prints the parameter budget and runs a self-test (CPU is fine)
```

## Architecture

The model follows the Llama 3 / Qwen 3 recipe in the deep-and-thin shape that works best
for small models.

| Part | Choice | Why |
|---|---|---|
| Layout | Decoder-only, pre-norm | Stable at depth |
| Normalization | RMSNorm computed in fp32, no bias anywhere | Cheaper than LayerNorm, same quality |
| Positions | RoPE, θ = 100,000 | Relative positions, with room to extend the context later |
| Attention | GQA (9 query heads, 3 KV heads) + QK-Norm | 3× smaller KV cache; QK-Norm bounds attention logits (stable at high LR and in fp16) |
| FFN | SwiGLU, 1536 hidden | Beats GELU at equal parameters |
| Output head | Tied to the input embedding | Saves 18.9M params (18% of the model) |
| Shape | 24 layers × 576 wide | Below 1B params, depth beats width (MobileLLM; SmolLM2 uses the same block) |

The model is mathematically identical to Hugging Face `Qwen3ForCausalLM`. I verified this:
logits match to 1e-6 once the fused `qkv_proj` / `gate_up_proj` weights are split. So a
trained checkpoint can later be exported to TRL, vLLM or llama.cpp (GGUF).

### Hyperparameters

| | Value |
|---|---|
| `vocab_size` | 32,768 |
| `d_model` | 576 |
| `n_layers` | 24 |
| `n_heads` / `n_kv_heads` / `head_dim` | 9 / 3 / 64 |
| `ffn_hidden` | 1,536 |
| `max_seq_len` | 2,048 |
| `rope_theta` | 100,000 |
| `norm_eps` | 1e-5 |
| Dropout | 0.0 for pretraining |
| Init | N(0, 0.02); `o_proj` and `down_proj` use N(0, 0.02/√48) |

### Parameter budget (exact)

| Component | Shape | Params |
|---|---|---|
| Token embedding = LM head (tied) | 32,768 × 576 | 18,874,368 |
| Block: `qkv_proj` | 576 → 960 (q 576, k 192, v 192) | 552,960 |
| Block: `o_proj` | 576 → 576 | 331,776 |
| Block: `gate_up_proj` | 576 → 3,072 | 1,769,472 |
| Block: `down_proj` | 1,536 → 576 | 884,736 |
| Block: RMSNorm gains | 2 × 576, plus QK-Norm 2 × 64 | 1,280 |
| One block | | 3,540,224 |
| 24 blocks | | 84,965,376 |
| Final RMSNorm | 576 | 576 |
| **Total** | | **103,840,320** |

FFN is 61.3% of the parameters, attention 20.4% and the embedding 18.2%. Non-embedding
parameters: 84,965,952. Training cost: 0.96 GFLOP per token at T = 2048.

### Tensor dimensions (B = batch, T ≤ 2048)

| Tensor | Shape | dtype under autocast |
|---|---|---|
| `input_ids`, `labels` | (B, T) | int64 |
| Residual stream | (B, T, 576) | fp32 |
| q | (B, 9, T, 64) | fp16 / bf16 |
| k, v | (B, 3, T, 64), repeated to 9 heads only inside the attention call | fp16 / bf16 |
| SwiGLU gate, up | (B, T, 1536) each | fp16 / bf16 |
| Logits | (B, T, 32768) at inference; in training, 2048-token chunks only | fp32 for the loss |
| KV cache (per layer) | 2 × (B, 3, T_max, 64) | fp16 / bf16 |
| RoPE tables | cos, sin: (2048, 32) | fp32 |

## Memory on Colab / Kaggle

The fixed cost is about 3 GiB:

| Item | Size |
|---|---|
| fp32 weights, gradients and AdamW states (16 bytes per param) | 1.55 GiB |
| 16-bit weight copies made by autocast | 0.2 GiB |
| One loss chunk | < 1 GiB |
| CUDA context | ~0.5 GiB |

Activations add **819 KiB per token**. That figure was measured with autograd
saved-tensor hooks under 16-bit autocast, and it grows linearly with T because SDPA never
stores the T×T attention matrix. With `model.gradient_checkpointing = True` it drops to
about 93 KiB per token.

| GPU | Precision | Micro-batch (seqs × 2048) | Activations | Est. peak | Rough throughput |
|---|---|---|---|---|---|
| T4 16 GB (Colab free, Kaggle) | fp16 + GradScaler | 4 | 6.4 GiB | ~9.5 GiB | 10–20k tok/s |
| T4 with gradient checkpointing | fp16 + GradScaler | 16 | 2.9 GiB | ~6 GiB | ~30% slower per token |
| P100 16 GB (Kaggle) | fp16 + GradScaler | 4 | 6.4 GiB | ~9.5 GiB | 4–8k tok/s |
| A100 40 GB (Colab Pro) | bf16 | 16 | 25.6 GiB | ~29 GiB | 80–150k tok/s |

The throughput figures are estimates from FLOPs and typical GPU utilization. Measure the
real numbers with `torch.cuda.max_memory_allocated()` and a timer. At these rates,
1B tokens takes about 14–28 h on a T4 and 2–3.5 h on an A100.

Training tips:

- **Precision.** T4 and P100 have no native bf16, so `autocast_dtype()` returns fp16 and you
  need `torch.amp.GradScaler`. A100 and L4 use bf16 and need no scaler. The training-step
  snippet is in the docstring at the top of `model.py`.
- **Global batch.** Use gradient accumulation to reach 0.25–0.5M tokens per optimizer
  step. On a T4: 4 × 2048 × 32 accumulation steps = 262k tokens.
- **Optimizer.** `model.configure_optimizer()` sets up AdamW with β = (0.9, 0.95), weight decay 0.1
  (matrices only) and peak LR 2e-3, with 1–2k warmup steps and gradient clipping at 1.0. Use a
  warmup-stable-decay schedule: it lets you resume across Colab sessions and run the LR
  decay only at the very end.
- **`torch.compile(model)`.** Works on T4 and A100 and usually reduces memory further. It does
  not work on P100 (Triton needs compute capability ≥ 7.0).
- **Context length.** Pretraining most steps at T = 1024 and switching to 2048 for the last
  10–20% is about 18% cheaper per token.
- **KV cache.** It is 18 KiB per token in fp16, so 36 MiB for a full 2048-token sequence.
  Full multi-head attention would need 108 MiB.

## Post-training alignment: the historian persona

The pipeline is: **pretrain → SFT → DPO (or ORPO in place of both SFT and DPO)**. SFT teaches
the format and the voice. The preference stage teaches judgment: being confident where the
evidence is strong, and open about it where it is not.

### 0. Reserve chat tokens in the tokenizer now

When you train the tokenizer (SentencePiece or BPE, 32,768 tokens, Mongolian-heavy, with
byte fallback), reserve these special tokens so their embeddings exist during pretraining:

```
<|endoftext|>   document separator in pretraining
<|im_start|>    start of a chat turn (ChatML, as in Qwen)
<|im_end|>      end of a chat turn; also the stop token for generation
<|pad|>         padding
```

### 1. Define the persona as a rubric

Write the rubric down once. Both the SFT answers and the preference judgments follow it.

1. **Answer first, plainly.** No throat-clearing.
2. **Name the evidence.** Cite primary sources (*Монголын нууц товчоо*, Rashid al-Din's
   *Jami' al-tawarikh*, Juvaini, the *Yuan Shi*, *Altan Tobchi*, Carpini, Rubruck) and
   the main modern historians.
3. **Calibrate.** Separate consensus from contested and from unknown. Example: the dating of
   the Secret History (1228 vs. 1240).
4. **Criticize the sources.** Say whose perspective a source carries: Ilkhanate court
   historians, the Ming compilers of the *Yuan Shi*, Soviet-era historiography, modern
   national narratives.
5. **Never invent** dates, citations or quotations. If something is unknown, say what
   evidence would settle it.
6. **Reply in the user's language.** Mongolian Cyrillic by default.

A short, fixed system prompt works best. Small models follow a persona they were trained
into, not a long instruction.

```
Та бол Монголын түүхийн мэргэжлийн судлаач. Асуултад шууд, тодорхой хариулж, гол эх
сурвалжаа нэрлэ. Маргаантай асуудлыг тэмдэглэж, эх сурвалжийн найдвартай байдлыг үнэл.
Баримтгүй он, нэр, эшлэл хэзээ ч бүү зохио.
```

Here is what the rubric looks like in practice:

> **Q:** When was Chinggis Khaan born?
>
> **Chosen:** Most likely in 1162. That is the year Mongolia officially uses, and it
> follows from the *Yuan Shi*, which says he died in 1227 aged 66 by Chinese reckoning.
> The sources do disagree: Rashid al-Din's figures imply 1155, and Paul Pelliot argued for
> 1167. The Secret History gives no year at all. So 1162 is the consensus working date,
> not a certainty.
>
> **Rejected (fabrication):** He was born on 31 May 1162 at Delüün Boldog, as recorded in
> the Secret History.
>
> **Rejected (evasion):** Historians have many opinions, so it is impossible to say.

### 2. SFT (supervised fine-tuning)

- **Data.** Collect 10k–50k conversations. The best source is your own corpus: give a
  stronger LLM a passage and have it write question → historian-style answer pairs, then
  have a historian spot-check them. Check the terms of service of the model you use for
  this. Cover factual Q&A, source criticism ("How reliable is Juvaini's casualty figure
  for Nishapur?"), historiography ("How did Soviet-era and post-1990 scholarship treat
  Choibalsan?"), chronology and causation, contested and unknown questions, plus about 5%
  small talk.
- **Grounded examples.** Make about 30% of the examples carry a source passage in the
  prompt, with an answer that cites it. A 100M model cannot memorize every date. Training
  it to reason over supplied sources is how historians work, and it makes retrieval (RAG)
  work later. The 2048-token context fits 3–5 passages.
- **Loss only on assistant tokens.** Labels are `-100` everywhere else. Include
  `<|im_end|>` in the loss so the model learns to stop.

  ```python
  from model import IGNORE_INDEX

  ids, lab = [], []
  for turn in conversation:                       # [{"role": ..., "content": ...}, ...]
      head = tok.encode(f"<|im_start|>{turn['role']}\n")
      body = tok.encode(f"{turn['content']}<|im_end|>\n")
      ids += head + body
      lab += [IGNORE_INDEX] * len(head) + (body if turn["role"] == "assistant" else [IGNORE_INDEX] * len(body))
  x, y = ids[:-1], lab[1:]                        # model(x, y): y[t] is the token after x[t]
  ```

  Pad `x` on the right with `<|pad|>` and `y` with `-100`. No attention mask is needed
  with right padding.
- **Hyperparameters.** LR 1e-4 to 3e-4 with cosine decay and 3% warmup, 2–3 epochs, about 64
  sequences per step. Use dropout 0.1 if you have fewer than 20k examples. Mix in about 10%
  pretraining text so the model does not forget facts.

### 3. Preference stage: DPO or ORPO

- **Pairs.** Collect 5k–20k. For each prompt, sample 4 answers from the SFT model at
  temperature 0.8. A judge picks the best and worst against the rubric; the judge can be a
  historian, or a strong LLM given the reference passage. Add targeted rejected answers
  of these kinds:
  - invented dates or citations
  - hedging with no answer
  - myth stated as fact
  - anachronism
  - a single national or ideological lens presented as neutral
- **DPO** uses the frozen SFT model as the reference. Start with β = 0.1 and LR 2e-6
  (range 1e-6 to 5e-6), for 1–2 epochs. Precompute the reference log-probs once, so only
  the policy needs to be on the GPU. Watch two numbers: the reward margin should rise, and
  the chosen log-prob should not collapse. If it does collapse, lower the LR or raise β.
- **ORPO** does SFT and preference learning in one stage, with no reference model. Use
  λ = 0.1. Use LR ~1e-4 when starting from the base model, or ~1e-5 when starting from the
  SFT model.

Both losses work directly on this model through `reduction="none"`. These functions were
tested against `model.py`.

```python
import torch, torch.nn.functional as F
from model import IGNORE_INDEX

def sequence_logps(model, input_ids, labels):
    """Sum and per-token-mean log-prob of the response tokens (labels != -100)."""
    _, nll = model(input_ids, labels, reduction="none")              # (B, T)
    logp = -nll.sum(-1)
    return logp, logp / (labels != IGNORE_INDEX).sum(-1).clamp(min=1)

def dpo_loss(policy, ref, chosen, rejected, beta=0.1):
    """chosen / rejected = (input_ids, labels), prompt tokens masked with -100."""
    pi_w, _ = sequence_logps(policy, *chosen)
    pi_l, _ = sequence_logps(policy, *rejected)
    with torch.no_grad():                                            # frozen SFT model
        ref_w, _ = sequence_logps(ref, *chosen)
        ref_l, _ = sequence_logps(ref, *rejected)
    return -F.logsigmoid(beta * ((pi_w - ref_w) - (pi_l - ref_l))).mean()

def orpo_loss(policy, chosen, rejected, lam=0.1):
    """Single-stage SFT + odds-ratio preference loss; no reference model needed."""
    _, avg_w = sequence_logps(policy, *chosen)
    _, avg_l = sequence_logps(policy, *rejected)
    log_odds = lambda a: a - torch.log1p(-a.exp().clamp(max=1 - 1e-6))
    l_or = -F.logsigmoid(log_odds(avg_w) - log_odds(avg_l)).mean()
    return -avg_w.mean() + lam * l_or
```

Running 2–3 rounds of DPO (sample → judge → train → repeat) works better than one large
round.

### 4. Evaluate the persona, not just the loss

Have a historian write 300–500 held-out questions; never generate them. Score:

- accuracy of dates and names
- whether cited sources are real and attributed correctly
- calibration: does the model flag contested questions?
- fabrication rate
- judge win-rate against the SFT model

Track perplexity on held-out historical texts throughout.

## Weight-space programming: answers with no training

`program_qa.py` writes every weight of a one-layer `MongolHistorySLM` by hand, so that the
model answers the questions in a Q&A file. There is no loss, no gradient and no optimizer,
and it runs on a CPU in seconds.

```bash
python -m venv .venv && .venv\Scripts\activate              # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy openpyxl
python program_qa.py --data data/qa.csv                # .csv .tsv .xlsx .json .jsonl
python program_qa.py --data data/qa.csv --chat         # ask your own questions
python program_qa.py --data data/qa.csv --explain 3    # read answer #3's rules back from the weights
python program_qa.py --data data/qa.csv --save out/programmed_qa.pt   # an ordinary model.py checkpoint
```

Without `--data`, the script uses `data/qa.*`, or `examples/sample_qa.csv` if that is missing.
`data/` is gitignored. The header can be `question,answer` or `асуулт,хариулт`; with no header,
the first two columns are used.

### Where the idea comes from

In [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang and Khattab, MIT CSAIL, 2 November 2026), a frontier LLM (GPT-6 Astra)
reads a network's raw weights and writes new values directly, without gradient descent:

- It told what small networks were trained to do from their weights alone.
- It wrote game-playing networks (600 → 32 → 32 → 6) as text in one reply, starting from all-zero weights.
- It traced Qwen3.5-0.8B's looping to "one direction written by the MLPs of layers 15 and 19" and
  subtracted it from 64 output columns in each. That is 131,072 weights (0.016%), and with 50
  forward passes MATH-500 rose from 7.9% to 48.1%.
- It could not improve Gemma 3 270M at math (6.8%, at best 7.7%). The authors' explanation:
  editing weights "could remove an obstacle that hid an ability, but it could not add an ability
  the model lacked."

That last result is why weight-space programming cannot replace training here. The weights of
an untrained model are random, so there is nothing to read and no ability to unlock. What it can
do is write a network for a task you can state as rules, and "recite these answers" is one.

### The program

| Part | What the hand-written weights do |
|---|---|
| Embedding | Splits each token's vector into named blocks: a large constant (it makes RMSNorm a fixed rescale and serves as a bias input), a random 128-d token code, a stem code × IDF, and flags |
| BAG heads | q = 0, so attention is uniform. At `<sep>` the output is the average stem code of the question: a TF-IDF fingerprint |
| PREV k heads | Constant q and k, placed so that RoPE peaks exactly k tokens back. Each copies that token's code |
| ANSWER heads | The key reads the answer-id flag, so the head copies which answer is being recited |
| `lm_head` at `<sep>` | One row per stored question holds its fingerprint. Softmax picks the closest question, and the model emits `<q17>`, or `<q?>` for "I don't know" |
| SwiGLU FFN | One neuron per answer token, read as a rule: IF answer = 17 AND the context ends with " хаан", " 1162" THEN write " онд" |

### What to expect

- Every stored answer comes back word for word. The script checks every answer token.
- Questions are matched by their words. Case, word order, Mongolian suffixes (the first 4
  letters are compared) and one missing word are tolerated. A word the program never saw pushes
  toward "I don't know", so off-topic questions and unknown names get no made-up answer.
- There is no understanding. To the program, аав and эцэг are unrelated words. It cannot answer
  anything that isn't stored, or combine two stored facts (birth year + death year → age).
  `--unknown-weight 0.5` makes it answer more often, and it then gives Napoleon Chinggis Khaan's
  birth year.
- It is very wasteful: about one neuron per answer token. The 12-row sample takes 10 MB; a
  207-row stress set took 90M parameters. Trained networks store facts far more densely, in
  superposition.

On a synthetic 207-row stress set (near-duplicate questions, long answers that repeat phrases),
all 207 answers came back exact. 100% of exact, lowercased, shuffled and suffixed questions were
matched, 83% with one word dropped, and all 6 off-topic questions got "I don't know". The whole
run took about 20 s on a CPU.

### Where real weight-space programming fits

Once `MongolHistorySLM` is pretrained, the MIT version becomes possible. An LLM agent with a
Python sandbox and a small forward-pass budget reads the trained weights and patches a localized
fault, such as the repetition loops small models fall into. It will not add knowledge the model
never learned; that still comes from data.

## Status and next steps

- [x] `model.py`: architecture, chunked loss, KV-cache generation, self-test
- [x] `program_qa.py`: weight-space programming experiment (hand-written weights answer a Q&A file)
- [ ] Tokenizer (32,768 tokens, Mongolian-heavy, with the special tokens above)
- [ ] Pretraining data pipeline and `train.py` (AMP, gradient accumulation, WSD schedule, checkpoint resume)
- [ ] SFT / DPO scripts and a historian eval set
