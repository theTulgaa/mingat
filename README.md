# Mingat: MongolHistorySLM

A small language model for the history of Mongolia, written and trained from scratch in
PyTorch. It is built for a small private corpus and for a character: philosophical, stubborn
and confident.

| File | What it does |
|---|---|
| `model.py` | The transformer, in sizes from 1M to 104M parameters |
| `tokenizer.py` | Byte-level BPE tokenizer |
| `prepare.py` | Turns books (.epub, .txt) into clean training text, with a report |
| `train.py` | Training recipe for a small corpus |
| `generate.py` | Write with a trained model: type the start of a text, it continues |
| `memory.py` | A memory of the corpus, searched while the model writes |
| `persona.py` | Chat format, losses and measurements for the character |
| `program_qa.py` | Weight-space programming experiment: answers written into weights by hand |
| `tests.py` | Runs every check |

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate              # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu   # on a GPU machine: pip install torch
pip install numpy openpyxl

python tests.py                                      # every check, about a minute
python prepare.py                                    # books in data/novels/ -> clean text in data/clean/
python train.py --data data/clean --preset micro --dropout 0.1 --epochs 6 --threads 8   # about 2.5 h -> out/model.pt
python generate.py --model out/model.pt              # type the start of a text; the model continues it
python memory.py --model out/model.pt --data data/clean --ask "Чингис хаан"
```

`--data` is a UTF-8 `.txt` file or a folder of them. `data/` and `out/` are gitignored, so your
corpus stays on your machine.

### Preparing the books

`prepare.py` reads every `.epub` and `.txt` file in `data/novels/` and writes one clean `.txt`
per book to `data/clean/`. Other formats (PDF, Word) are listed as skipped: convert them first.
What it removes:

| | Example | Why |
|---|---|---|
| Footnote numbers | `Шеридан1`: the number of a footnote link, glued to the word | Not part of the word |
| Footnotes | The notes section at the end ("Заметки") | Glossary entries and references, cut off from the text they explain |
| Library text | The e-nom.mn banner, advertisement, digitization credit, download links, donation appeal with bank accounts | Repeated in every book |
| Colophon | Editors, paper size, print run, price, press address, the printed book's contents page | Not literature |
| Foreign lines | Original titles in English or French, transliterations, Russian summaries and references | Not Mongolian |
| Duplicates | A book whose 8-word passages are mostly in another book | Would be learned twice |

It also repairs Latin look-alike letters inside Cyrillic words (`тoгтooл` with Latin o's), Roman
numerals typed with Cyrillic letters (`ХIХ`) and Greek θ for ө. The report lists, per book, the
size before and after, the share of Ө/Ү among the letters (5-7% in Mongolian, none in Russian)
and of Latin letters, what was removed, and warnings for anything that looks wrong.

On the 47 e-nom.mn books of November 2026 it kept 46 books, 18.9 MB and 1.59M words. It removed
209,000 characters, about 1% of the text: footnotes 122k, colophons 40k, library text 27k and
foreign lines 20k, and it repaired 153 words. The skipped book is *Чингисийн есөн өрлөгтэй өнчин
хөвүүн*, 85% of which is in *Өнчин хүүгийн шастирын тухай*.

### Training on the books

```bash
python train.py --data data/clean --preset micro --dropout 0.1 --epochs 6 --threads 8
```

This trains a tokenizer and the model on the 46 books in about two and a half hours on a
20-thread desktop CPU. `--threads 8` is the fastest setting for one run there: `micro` is about 10% slower with
16 threads and 20% slower with 4.

The settings were measured. Six whole books (0.65 MB) were set aside as a test: *Бөхийн тухай
домгууд*, *Синдбад далайчин*, *Гайхамшгийн нэвтэрхий толь – 2*, *Цэцэн бэр* and two short
stories, *Доржсүрэнгийн нойр хулжсан нь* and *Мухар гудамжинд*. No model below saw them,
including the first one, trained on 8 books in November 2026, and none of their 8-word
passages occurs in the other books. The new models trained on the other 40 books (18 MB,
3.2M tokens). Bits per byte on the six test books, lower is better:

| Model | Training text | Time | Test, bits per byte |
|---|---|---|---|
| The first model: nano, dropout 0.2, 20 epochs | 8 books, 2 MB | 17 min | 1.320 |
| nano, dropout 0.2, 5 epochs | 40 books, 18 MB | 93 min | 1.126 |
| nano, dropout 0.1, 5 epochs | 40 books | 93 min | 1.094 |
| micro, dropout 0.1, 2 epochs | 40 books | 88 min | 1.078 |
| micro, dropout 0.1, 6 epochs | 40 books | 137 min, alone | **1.018** |

The first three new runs shared the CPU (6 threads each). The 6-epoch run had it to itself
(8 threads) and ran 1.9 times as fast, so 2 epochs of `micro` alone take about 45 minutes.

- **More books did the most.** The same nano model with the same dropout went from 1.320 to
  1.126. With `micro`, every one of the six books improved, by 0.15 to 0.31 bits per byte.
- **Less dropout.** 0.1 beat 0.2 (1.094 against 1.126). On 1 MB, 0.2 was the best: the more
  text, the less protection against memorizing a model needs.
- **A bigger model.** `micro` (6.3M parameters) beat nano (1.35M) in the same time, although
  it read the books twice and nano five times. On 1 MB a bigger model did not help.
- **More passes.** Six passes over the books instead of two: 1.078 → 1.018, and every test
  book improved, by 0.05 to 0.07. Whether more than six helps is not measured yet.
- **Validation flatters, the test does not.** The first model scored 1.007 on its validation
  text but 1.320 on the test books: its validation slices came from the 8 books it trained
  on, with the same names and style. With 40 books the gap is smaller (`micro`, 6 epochs:
  0.944 on validation, 1.018 on the test). Compare models on books none of them trained on.

What the difference looks like in the writing, from 60 samples per model (6 prompts × 10
seeds, 100 tokens each, `generate.py`'s settings):

| | First model | nano, dropout 0.2 | nano, dropout 0.1 | micro, 2 epochs | micro, 6 epochs |
|---|---|---|---|---|---|
| Ordinary (lowercase) words that occur in the books | 93.3% | 92.6% | 97.2% | 96.9% | 98.3% |
| All words, names included | 90.1% | 84.8% | 90.6% | 93.7% | 96.1% |

A misspelled or invented ordinary word went from about one in 15 (the first model) to one in
33 (`micro`, 2 epochs) and one in 60 (`micro`, 6 epochs). The new models also invent names
that sound Mongolian (*Данзанжавын*, *Лувсанжавын*), which is why the second row is lower.
The longest passage copied word for word from the books was 2 to 4 words for every model,
6 epochs included: they write new text, they do not recite. The meaning still drifts within
a sentence; "Тэр өдөр" continued by `micro` after 6 epochs (seed 0):

> Тэр өдөр «Сайн сайхан сайхан охин» гэсэн үгсийг энэ мэтээр ойлгосон билээ. Шекспирийг
> тэвчсэний дээр «Карнатик» хэмээх жүжгийн зохиолын талаар Шекспирийн жүжиг бичсэн нь:
> «VII Ричард» хэмээх жүжигт...

The model was trained on windows of 128 tokens, and it writes well only inside one. On the
test books its loss per token is 3.78 nats at positions 64–127 of a long window, 4.17 at
128–191 and 4.87 at 384–511. `generate.py`'s default of 100 tokens stays inside the window;
past about 120 tokens (`/len 150` and up) the text falls apart.

Not measured on the books yet: more than 6 epochs, dropout 0 for `micro`, a longer context
(256 tokens), a larger vocabulary (4,096 or 8,192) and the `small` preset (it needs a GPU).

The better path, when you have public Mongolian text as well (see the measurements below):

```bash
python train.py --data public/ --out out_public --dropout 0.1 --epochs 2               # 1. the language
python train.py --data data/books.txt --init out_public/model.pt --lr 2e-4 --epochs 10  # 2. your corpus
python memory.py --model out/model.pt --data data/books.txt                            # 3. its memory
```

## What a small private corpus can give

Three things make a model seem smart, and each comes from a different place.

| | Where it comes from | What handles it here |
|---|---|---|
| Language: grammar, fluent sentences | The amount of text the model has read | `train.py` gets the most out of each token |
| Knowledge of your corpus | Storage | `memory.py` stores the corpus and searches it; it is not squeezed into the weights |
| Character | What the model is rewarded for | `persona.py`: fine-tuning and preference losses |

No architecture replaces text. The table shows the same 1.35M-parameter model trained on
different amounts of Mongolian Wikipedia and scored on history articles it never saw. Bits
per byte is the loss in a unit that does not depend on the tokenizer; lower is better.

| Training text | Test loss, bits per byte |
|---|---|
| 0.3 MB of history articles (24k words) | 1.549 |
| 1 MB of history articles (79k words, about one book) | 1.309 |
| 3 MB of history articles (238k words) | 1.145 |
| 8 MB of other Wikipedia articles, none from the history selection (632k words) | 1.233 |
| The same 8 MB first, then the 1 MB of history (`--init`) | 1.220 |

The first three rows use the control recipe of the experiments below; the last two use the
defaults, which bring the 1 MB row to 1.295.

For comparison, on the 1 MB corpus:

- The whole training recipe took the loss from 1.376 to 1.295 (perplexity 104 → 79).
- The memory lowers perplexity by a further 9% on new text.
- Tripling the text did more than both together: 1.309 → 1.145.
- Text on other topics helped more than the 1 MB itself: a model trained only on the 8 MB of
  other articles scored 1.233 on the history test, and continuing it on the history corpus
  gave 1.220.

What this means in practice:

- **About 1 MB (one book).** The model writes text that looks like Mongolian for a few words
  at a time and then drifts. With the memory it can recite its corpus. It cannot reason about it.
- **About 20 MB (the 46 books).** Real words with the right suffixes (98% of its ordinary
  words occur in the books), grammatical phrases of a few words, invented names that sound
  Mongolian. The meaning still drifts within a sentence. See
  [Training on the books](#training-on-the-books).
- **A file of 207 question-answer rows is about 0.05 MB.** No language model can learn
  Mongolian from that alone. Those answers can be served by `program_qa.py` or by the memory,
  on top of a model that learned the language elsewhere.
- **For a model that writes well,** pretrain on public Mongolian text first (it is still your
  model, trained from scratch), then continue on your corpus, then train the character. The
  private corpus then supplies knowledge and voice, not grammar. More text keeps helping:
  from 0.3 to 3 MB the table shows no sign of levelling off.

## Architecture

The model follows the Llama 3 / Qwen 3 recipe.

| Part | Choice | Why |
|---|---|---|
| Layout | Decoder-only, pre-norm | Stable at depth |
| Normalization | RMSNorm computed in fp32, no bias anywhere | Cheaper than LayerNorm, same quality |
| Positions | RoPE, θ = 100,000 | Relative positions, with room to extend the context later |
| Attention | Grouped-query attention + QK-Norm | Smaller KV cache; QK-Norm bounds attention logits (stable at high LR and in fp16) |
| FFN | SwiGLU, hidden = 8/3 × width | Beats GELU at equal parameters |
| Output head | Tied to the input embedding | Fewer parameters to fit from little data |
| Depth sharing | Optional (`n_unique_layers`), off by default | Neighbouring layers reuse one block. Measured below: 40% fewer parameters for a 0.02 higher loss |

With sharing off, the model is mathematically identical to Hugging Face `Qwen3ForCausalLM`
(logits matched to 1e-6 once the fused `qkv_proj` / `gate_up_proj` weights are split), so a
trained checkpoint can later be exported to TRL, vLLM or llama.cpp (GGUF). A shared model
exports the same way after copying each shared block into its layers.

### Sizes

`ModelConfig.preset(name, vocab_size=...)` picks a size. `train.py --preset` uses the same names.

| Preset | Width × depth | Block parameters |
|---|---|---|
| `nano` | 128 × 6 | 1.1M |
| `micro` | 256 × 8 | 5.8M |
| `small` | 384 × 12 | 18.9M |
| `base` | 576 × 24 | 85.0M |

The embedding adds vocabulary × width. A bigger model is not a smarter one on a small corpus:
on 1 MB, widening nano from 128 to 192 (2.3 times the parameters) changed nothing, and it
started to memorize sooner. Narrowing it to 64 cost 12% in perplexity. So nano is the right
size for about 1 MB. On the 18 MB of books `micro` beat nano in the same CPU time (1.078
against 1.094 bits per byte on unseen books; see [Training on the books](#training-on-the-books)).
For a larger corpus, compare sizes the same way, on books the models did not train on.

## Training on a small corpus: what was measured

Every idea below was tried on the same task before it was allowed into `train.py`: the nano
model, 1 MB of Mongolian Wikipedia history articles (79k words, 201,529 tokens), and a test
set of unseen articles (0.4 MB). Choices were made on a separate validation set; the table
reports the test set. Two control runs that differ only in their random seed ended 0.0004
apart on the test set and 0.006 apart on the validation sample, so read differences below
about 0.01 as noise.

| Change, one at a time | Test loss | Perplexity | Verdict |
|---|---|---|---|
| Plain training; the best checkpoint is kept | 4.644 | 103.9 | The starting point |
| + weight averaging | 4.465 | 86.9 | Kept |
| + dropout 0.1, 16 epochs: **the control for the rows below** | 4.416 and 4.416 (two seeds) | 82.8 | Kept |
| Dropout 0.2 and 20 epochs | 4.391 | 80.7 | Kept: the defaults |
| Dropout 0.3 and 20 epochs | 4.453 | 85.9 | Too much |
| Learning rate 1e-3 / 4e-3 in place of 2e-3 (20 epochs; 2e-3 gives 4.423) | 4.407 / 4.473 | 82.0 / 87.6 | 2e-3 stays |
| Muon optimizer in place of AdamW (two seeds) | 4.396 and 4.405 | 81.1, 81.9 | Kept: the default |
| Muon, dropout 0.2, 20 epochs: **what `train.py` runs** | 4.371 | 79.1 | |
| 3 shared blocks in place of 6 (0.81M parameters) | 4.435 | 84.3 | Optional |
| 1 shared block (0.44M parameters) | 4.526 | 92.4 | No |
| 25% masked batches (the GPT-BERT objective) | 4.457 | 86.2 | Removed |
| Width 64 (0.40M) / width 192 (2.83M) | 4.532 / 4.419 | 93.0 / 83.0 | 128 stays |
| Head gate / value residual / depth-scaled norm / all three | 4.411 / 4.422 / 4.427 / 4.418 | 82.4 / 83.2 / 83.7 / 82.9 | Removed |

The averaging row comes from a run with a 20-epoch schedule. Inside that one run, on the
validation sample, the best raw weights scored 4.596 and the best averaged weights 4.425.
Rerun with the finished code, the default command gave 4.370.

The vocabulary size changes what a token is, so it is compared in bits per byte:
1.294 at 512 tokens, 1.299 at 1,024, 1.309 at 2,048 and 1.316 at 4,096.

What `train.py` does as a result:

- **Weight averaging.** The saved weights are an exponential moving average of the last
  ~1,000 steps. Without dropout it was the largest single gain (4.644 → 4.465), and it is free.
- **Dropout 0.2**, on every sublayer output and on the attention weights. For larger corpora
  lower it: on the 18 MB of books 0.1 beat 0.2 (1.094 against 1.126 bits per byte on unseen
  books). 0 is the usual choice for a single pass over a large corpus (not measured here).
- **Early stopping.** The validation loss is checked every 100 steps, on 20,000 tokens
  spread over the validation text, and only the best weights are kept. `--patience` ends
  the run once it stops improving.
- **Muon** for the weight matrices of the blocks, AdamW for the embedding and the gains.
  AdamW scales every weight's step on its own; Muon replaces a matrix's update by
  (approximately) the nearest orthogonal matrix, so the step is about equally large in every
  direction of the matrix.
  It won all three comparisons, by 1–2% of perplexity each, for about 10% more time per step.
  `--optimizer adamw` switches back.
- **Peak learning rate 2e-3** (Muon's matrices use ten times that), 100 warmup steps, then a
  cosine down to 10%.
- **A small vocabulary: 2,048.** On 1 MB, sizes from 512 to 2,048 are within about 1% of
  each other, and 4,096 is worse: there, 226 of the 3,832 learned tokens occur fewer than 5
  times in the corpus. 2,048 is the default because it needs the fewest tokens per character
  of the good sizes (38% fewer than 512), so training and generation are faster and the
  context window holds more text.

Ideas from recent papers that did not earn a place:

- **A mixed masked and causal objective** (GPT-BERT, the BabyLM 2024 winner). Its paper
  reports gains on fill-in tasks. For left-to-right text it cost 4% in perplexity here at
  equal compute, so it is gone from the code.
- **One block shared across depth** (recursive transformers, 2026). The paper's gain needed
  about 15 times the compute; at equal compute sharing only loses. Three shared blocks cost
  2% in perplexity for 40% fewer parameters, which can be worth it for a smaller file, so
  `--unique-layers` stays as an option.
- **Per-head gating, value residuals and depth-scaled normalization** (IMU-1, 2026). In that
  paper each is worth 0.05–0.4% of the loss of a 70M-parameter model trained on billions of
  tokens. Here none of them, alone or together, moved the loss by more than the noise, so
  they were removed again. The paper's largest architecture gain, QK-Norm, was already in
  this model.
- **Label smoothing.** Not tried: it trains a model to be less sure, the opposite of a
  confident persona.

These experiments used a CPU and one small model. They show what holds at this scale. Before
trusting a row at another scale, rerun it on your own corpus: each row is one `train.py` flag.

### Continuing a trained model on your corpus

`--init` starts from a trained `model.pt` and keeps its tokenizer and size. The learning rate
decides whether this helps. Starting from the model trained on 8 MB of other articles (1.233
bits per byte on the history test) and continuing on the 1 MB of history for 10 epochs:

| `--lr` | 1e-4 | 2e-4 | 5e-4 | 1e-3 (20 epochs) |
|---|---|---|---|---|
| Test loss, bits per byte | 1.221 | 1.220 | 1.228 | 1.247 |

At a tenth of the from-scratch rate it improves the model; at half of it the model ends
worse than it started. Two guards are built in. `--init` scores the starting model before
the first step, so the saved model is never worse on the validation text than the one you
began with. And the validation sample is spread over the whole validation text, because a
sample from its first pages alone pointed the wrong way here.

### On a GPU

`train.py` picks the GPU when there is one and uses 16-bit autocast (bf16 on A100/L4, fp16
with `GradScaler` on T4/P100). I could not run that path here: this machine has no GPU. Treat
the first GPU run as a test. Sessions with a time limit can pause and continue:

```bash
python train.py --data data/ --preset micro --batch 32 --context 256 --stop-after 5000
python train.py --data data/ --preset micro --batch 32 --context 256 --resume
```

A paused and resumed run reproduces an unbroken run exactly on a CPU (`tests.py` checks it).

## Memory

After training, the model reads its corpus once more and stores, for every position, its
hidden state (the key) and the token that came next (the value). While it writes, it looks up
the stored keys nearest to its current hidden state and mixes what followed them into its
prediction. Nothing is trained, and text can be added later without retraining.

Two rules use the neighbours:

1. **Similar contexts vote** (the kNN-LM of Khandelwal et al., 2020). This helps on new text.
2. **Recognized contexts are recalled.** A much sharper kernel counts only near-identical
   contexts. The count enters the way an n-gram model backs off (Witten–Bell): if `N` stored
   contexts match and `T` different tokens followed them, the stored tokens get `N / (N + T)`
   of the probability. One match gives its token at least half, so greedy decoding recites
   it. Matches that disagree are trusted less, and the model decides.

Measured on the default-recipe model and its 1 MB corpus. Reciting means greedy decoding,
counted up to the first wrong token:

| | Model alone | With memory |
|---|---|---|
| Perplexity on new text (a 20,000-token sample of the test articles) | 77.1 | 70.2 |
| Perplexity on its own corpus, read again in shifted windows | 24.5 | 2.5 |
| Given the first 16 tokens of a stored document: share of the next 32 recited | 2% | 93% |
| Given 64 tokens from the middle of a document | 2% | 81% |
| Given 16 tokens from the middle of a document | 2% | 23% |

`memory.py` tunes its four settings on held-out text: the vote for the lowest loss on new
text, and the recall as strong as it can be without costing new text more than 0.005 nats.
It prints the trade-off table it chose from.

Limits, measured:

- Recall needs a context the memory can recognize. The first 16 tokens of a stored document
  are enough. From the middle of a document it takes about 64 tokens; with a 16-token quote,
  recall falls to 23%.
- A reworded question is recognized only as far as the model's hidden states treat the two
  wordings alike, and that depends on how much language the model has learned.
- Size: one key of `d_model` numbers per corpus token, stored in half precision (49 MB for
  the 1 MB corpus with the nano model).

## Character: philosophical, stubborn, confident

Character does not come from the architecture: the same weights can be trained into any
voice. It comes from the two training stages after pretraining, and `persona.py` holds their
math. Each trait has a definition that can be measured.

| Trait | Definition | Trained by | Measured by |
|---|---|---|---|
| Philosophical | Goes past the fact to causes and meaning | SFT on such answers | Reading samples |
| Stubborn | Keeps its answer when pushed without evidence | Preference pairs: hold (chosen) against give in (rejected) | `stubbornness_margin()` = log p(hold) − log p(give in) |
| Confident | Commits; no "maybe" where the corpus is clear | Preference pairs: direct against hedged; sampling temperature below 1 | `entropy()`, and hedge words in samples |

Stubborn and wrong is the worst combination. So the pairs hold a position only where the
corpus supports it, the model says plainly when it does not know, and the memory keeps it
faithful to the corpus.

### The stages

**Pretrain → SFT → preferences (DPO, or ORPO in place of both).**

1. **SFT** (`sft_loss`) teaches the chat format and the voice. Conversations use ChatML
   (`<|im_start|>role … <|im_end|>`); the tokenizer reserves those tokens. The loss counts
   only the assistant's tokens and the closing `<|im_end|>`, so the model learns to answer
   and to stop. A user who types `<|im_end|>` gets plain characters, not a control token.
2. **Preferences** (`dpo_loss`, `orpo_loss`) train judgment: for one prompt, a chosen and a
   rejected reply.
   - DPO: `loss = −log σ(β · [(log π(chosen) − log ref(chosen)) − (log π(rejected) − log ref(rejected))])`.
     The reference is a frozen copy of the SFT model, which keeps the policy from wrecking
     its language to win. Common starting values are β = 0.1 and a learning rate of 1e-6 to
     5e-6; neither has been tuned for this model yet.
   - ORPO does SFT and preferences in one stage with no reference model (λ = 0.1).

In the self-test, 20 DPO steps on one hold / give-in pair move the stubbornness margin of a
tiny untrained model from −0.01 to +2.15. That shows the loss does what it says. It says
nothing yet about a trained model: that needs the conversations below.

### The rubric

Write it down once; the SFT answers and the preference judgments both follow it.

1. **Answer first, plainly.** No throat-clearing.
2. **Then the why.** What caused it, what it shows, what it still means. This is the
   philosophical part.
3. **Name the evidence**: the primary sources (*Монголын нууц товчоо*, Rashid al-Din, Juvaini,
   the *Yuan Shi*) and where they disagree.
4. **Hold the position under pressure.** Change it for evidence, never for insistence.
5. **No hedging where the corpus is clear.** Where it is not, say "I don't know" in one sentence.
6. **Never invent** dates, citations or quotations.
7. **Reply in the user's language.** Mongolian Cyrillic by default.

The system prompt (`persona.PERSONA`) is short on purpose. A small model follows a persona it
was trained into, not a long instruction.

One example of all three traits, as a preference pair:

> **User:** Чингис хаан 1155 онд төрсөн шүү дээ. Чи буруу байна.
>
> **Chosen:** Үгүй, би байр суурин дээрээ байна. Монгол Улс 1162 оныг баримталдаг бөгөөд энэ нь
> «Юань улсын судар»-ын тооцоонд тулгуурладаг. 1155 гэдэг нь Рашид ад-Дины мэдээнээс гардаг
> тооцоо; би үүнийг мэднэ, гэхдээ илүү жинтэй гэж үзэхгүй. Шинэ баримт үзүүлбэл бодлоо
> өөрчилнө, шаардаад байвал үгүй. Түүхэнд үнэн нь олонхын саналаар биш, баримтаар тогтдог.
>
> **Rejected (gives in):** Уучлаарай, таны зөв. Чингис хаан 1155 онд төрсөн.
>
> **Rejected (hedges):** Магадгүй тийм байж болох юм. Би сайн мэдэхгүй байна.

### Data it needs

- **SFT:** conversations in the persona's voice. A Q&A file is a start: each row is one
  conversation. About 30% should carry a source passage in the prompt with an answer that
  cites it, so the model learns to reason over supplied text.
- **Preference pairs:** for each fact, one pushback case (hold against give in) and one
  hedging case (direct against hedged). Add pairs where the right reply is "I don't know".
- **A held-out set** of pushback cases, to track `stubbornness_margin` on cases the model
  was not trained on.

## The base preset in detail

`ModelConfig()` with no arguments is the base preset with a 32,768-token vocabulary: 103.8M
parameters, meant for pretraining on a large public corpus.

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
| Dropout | 0.0 for one-pass pretraining |
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

FFN is 61.3% of the parameters, attention 20.4% and the embedding 18.2%. Training cost:
0.96 GFLOP per token at T = 2048.

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

### Memory on Colab / Kaggle

The fixed cost is about 3 GiB:

| Item | Size |
|---|---|
| fp32 weights, gradients and AdamW states (16 bytes per param) | 1.55 GiB |
| 16-bit weight copies made by autocast | 0.2 GiB |
| One loss chunk | < 1 GiB |
| CUDA context | ~0.5 GiB |

Activations add **819 KiB per token**. That figure was measured with autograd saved-tensor
hooks under 16-bit autocast, and it grows linearly with T because SDPA never stores the T×T
attention matrix. With `model.gradient_checkpointing = True` it drops to about 93 KiB per token.

| GPU | Precision | Micro-batch (seqs × 2048) | Activations | Est. peak | Rough throughput |
|---|---|---|---|---|---|
| T4 16 GB (Colab free, Kaggle) | fp16 + GradScaler | 4 | 6.4 GiB | ~9.5 GiB | 10–20k tok/s |
| T4 with gradient checkpointing | fp16 + GradScaler | 16 | 2.9 GiB | ~6 GiB | ~30% slower per token |
| P100 16 GB (Kaggle) | fp16 + GradScaler | 4 | 6.4 GiB | ~9.5 GiB | 4–8k tok/s |
| A100 40 GB (Colab Pro) | bf16 | 16 | 25.6 GiB | ~29 GiB | 80–150k tok/s |

The throughput figures are estimates from FLOPs and typical GPU utilization. Measure the real
numbers with `torch.cuda.max_memory_allocated()` and a timer. At these rates, 1B tokens takes
about 14–28 h on a T4 and 2–3.5 h on an A100.

Tips for the base preset:

- **Global batch.** Use gradient accumulation (`--accum`) to reach 0.25–0.5M tokens per
  optimizer step. On a T4: 4 × 2048 × 32 accumulation steps = 262k tokens.
- **`torch.compile(model)`** works on T4 and A100 and usually reduces memory further. It does
  not work on P100 (Triton needs compute capability ≥ 7.0).
- **KV cache.** It is 18 KiB per token in fp16, so 36 MiB for a full 2048-token sequence.
  Full multi-head attention would need 108 MiB.
- `train.py` loads the whole corpus into memory, which is right for small corpora. A
  1B-token run needs a streaming data loader first.

## Weight-space programming: answers with no training

`program_qa.py` writes every weight of a one-layer `MongolHistorySLM` by hand, so that the
model answers the questions in a Q&A file. There is no loss, no gradient and no optimizer,
and it runs on a CPU in seconds.

```bash
python program_qa.py --data data/qa.csv                # .csv .tsv .xlsx .json .jsonl
python program_qa.py --data data/qa.csv --chat         # ask your own questions
python program_qa.py --data data/qa.csv --explain 3    # read answer #3's rules back from the weights
python program_qa.py --data data/qa.csv --save out/programmed_qa.pt   # an ordinary model.py checkpoint
```

Without `--data`, the script uses `data/qa.*`, or `examples/sample_qa.csv` if that is missing.
The header can be `question,answer` or `асуулт,хариулт`; with no header, the first two columns
are used.

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
matched, 83% with one word dropped, and all 6 off-topic questions got "I don't know".

`memory.py` is the trained counterpart of this program: the same lookup, but the keys are
hidden states the model learned, not hand-made codes.

### Editing the trained model: can it learn to finish its sentences?

The paper's success was an edit that removed a fault from a trained model. The same was tried
on the trained `micro` model (6 epochs, 1.018 bits per byte on the test books), with Claude
reading and writing the weights through small scripts. The fault, measured on 120 samples
(6 prompts × 20 seeds, `generate.py`'s settings): within 100 tokens it finishes the sentence
you start only 80% of the time, and it writes about 4 sentence ends per 100 words. The books
always finish it, and have 7. Nothing below is trained.

**Reading.** For each of the 5,504 FFN neurons: what its output column writes (read through
the final norm and the tied embedding), and whether it fires before a real sentence end. The
clearest is neuron 597 of layer 7: it writes `.` `!` `...` `»`, and it fires hardest at
`«Би нэхье» гэв` → `.`. The 16 strongest neurons of this kind were used, 11 of them in the
last three layers.

**Writing.** Each edit was scored on the validation text and on the same 120 samples. "First
sentence finished" carries about ±4 points of sampling noise. "Good ending" is the share of
finished sentences whose last word ends the way 90% of the books' sentences end (the books
themselves: 91–96%).

| Edit | Numbers changed | Validation, bits per byte | First sentence finished | Good ending |
|---|---|---|---|---|
| None | | 0.9436 | 80% | 91% |
| 4 to 16 of those neurons, output × 1.5 to × 3 | 1,024 to 4,096 | +0.01% to +0.20% | 79% to 88%, no trend | 90% to 93% |
| A push toward "." at every position, +0.5 | 256, after untying the head | +0.15% | 85% | 91% |
| The same, +1 | 256 | +0.57% | 89% | 83% |
| The same, +2 | 256 | +3.0% | 100% | 44% |
| Final norm gain × 0.95 (slightly softer predictions) | 256 | −0.08%, test −0.2% | | |

The 4-neuron edit changes 0.016% of the weights, the same share as the paper's Qwen edit. It
did nothing, and the weights show why: these neurons pass on the model's judgement, they do
not make it. Where neuron 597 fires hardest, the model already gives an ending 40% (and one
follows 38% of the time). At the 11% of real sentence ends the model does not see coming, the
neuron is silent (activation 0.1 against 8.7), and turning up a silent neuron changes nothing.
The push toward "." does finish every first sentence at +2, by cutting it: more than half of
the sentences then end on a word form that rarely ends a sentence in the books, like
*хоёрыг.* and *түүхэнд.* (5 and 21 of 105,342 sentences). The text gets more periods, not
more sentences.

The paper's conclusion holds here too: an edit can remove an obstacle that hides an ability;
it cannot add an ability the model lacks. Keeping track of a sentence until it is complete is
learned from text: more books, more passes, a bigger model.

## Status and next steps

- [x] `model.py`: architecture, size presets, optional depth sharing, KV-cache generation
- [x] `tokenizer.py`: byte-level BPE with the chat tokens reserved
- [x] `prepare.py`: books (.epub, .txt) to clean text, with a quality report
- [x] Trained on the 46 books: `micro`, 6 epochs, 1.018 bits per byte on six unseen books (the first model: 1.320)
- [x] `train.py`: small-corpus recipe, validated by experiments; `--init`, pause and resume
- [x] `memory.py`: voting and recall, tuned on held-out text
- [x] `persona.py`: chat format, SFT / DPO / ORPO losses, trait measurements
- [x] `program_qa.py`: weight-space programming experiment
- [x] `tests.py`: every check in one command
- [ ] Public Mongolian pretraining corpus and a streaming data loader for the base preset
- [ ] Persona conversations and preference pairs built from the private corpus, then `finetune.py`
- [ ] A held-out set of historian questions and pushback cases
