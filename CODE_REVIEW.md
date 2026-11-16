# Architecture and Numerical Code Review

Reviewed on 2026-11-17 against commit `fd5dc777b03ff3f23aaccb7883fdcd9bfea42950`.
This is a handoff for the next contributor or coding agent. **The findings below
are open; this review does not implement fixes.**

## Scope and Evidence

Reviewed the transformer, tokenizer, training/resume loop, generation, retrieval
memory, preference losses, data preparation, Q&A compiler, and experiment scripts.
The existing `tests.py` suite passed on CPU with Python 3.13.1 and PyTorch 2.15.0.

An additional comparison forced actual LM-head chunks (`ce_chunk_size=3`),
activation checkpointing, and shared layers. Its maximum gradient difference from
a dense cross-entropy reference was approximately `2.98e-8`. No core transformer
formula error was identified in these checks. CUDA execution, private corpora,
trained checkpoints, and README benchmark scores were not independently verified.

Run the additional probes from the repository root:

```bash
python review/reproduce_findings.py
```

The probes need a recent PyTorch with CPU `torch.amp.GradScaler` support (tested on
2.15.0). They print observed behavior and expected invariants; they are diagnostic
reproductions, not additions to the passing test suite. After fixing a finding,
its observed output should change. The context-length hang probe uses a bounded
POSIX timer and is skipped on platforms without that facility.

## 1. High: Shared Gradient Clipping Can Bypass AMP Protection

**Location:** `train.py:485-494`, optimizer unscaling, clipping, stepping, and EMA.

`scaler.unscale_(opt)` records nonfinite gradients separately for each optimizer.
The subsequent global `clip_grad_norm_` couples both groups. If one group contains
a NaN, clipping introduces NaNs into the previously finite group. Its scaler
check has already passed, so that optimizer still steps and corrupts its weights.

**Reproduction:** With a Muon matrix and a separate AdamW matrix, inject a NaN
only into the Muon gradient, then follow the trainer's unscale/clip/step sequence.
The AdamW gradient is initially finite, but its weights become nonfinite. This
was reproduced with enabled CPU GradScaler to exercise the same scaler ordering;
occurrence in an actual CUDA training run was not measured.

**Suggested fix:** Check the global norm after unscaling and before clipping or
stepping. On a nonfinite norm, skip both parameter updates and the EMA update,
clear gradients, and let the scaler update its scale. Preserve normal behavior
when scaling is disabled as well.

**Regression check:** Inject a NaN into one optimizer group and verify that all
weights and optimizer state remain finite and neither optimizer advances.

Reference: [PyTorch AMP documentation on independent optimizer checks](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers).

## 2. Medium: Retrieval Voting Can Lose Probability Mass

**Location:** `memory.py:127-129` and the corresponding `mixture_loss` computation
at `memory.py:203-204`.

Voting uses `exp((similarity - 1) / temperature)`, then clamps the denominator to
`1e-30`. At low similarities and a supported temperature of `0.01`, the real
denominator can be much smaller than the clamp. The resulting votes no longer
sum to one. Float32 underflow can make the problem worse.

**Reproduction:** For model probabilities `[0.5, 0.25, 0.25]`, 16 neighbors with
similarity zero, all voting for token 1, `lam=0.2`, and recall disabled, the
implementation returns approximately `[0.4, 0.2, 0.2]` (sum `0.8`). The specified
mixture should be `[0.4, 0.4, 0.2]` (sum `1.0`). This affects generation and the
interpretation of reported mixture losses.

**Suggested fix:** Normalize the voting kernel with a stable softmax over
`similarity / temperature`, or subtract the row maximum before exponentiating.
Apply equivalent math in tuning. Keep the separate recall kernel's absolute
similarity scale: its counts intentionally measure whether a context matches.

**Regression check:** Check normalization and direct/tuning loss agreement for
low and negative similarities, sharp temperatures, repeated token IDs, and recall
both enabled and disabled.

## 3. Medium: Multi-turn ORPO Includes Historical Assistant Replies

**Location:** `persona.py:80-92`, `orpo_loss` at `persona.py:153-154`, and the
multi-turn examples in that module's self-test.

`encode_chat` correctly labels every assistant turn for SFT. Reusing these labels
for ORPO causes its per-token average to include assistant messages in the
conversation prefix, as well as the chosen or rejected completion. Prefix terms
do not cancel out of differently length-normalized replies. The ORPO formula
itself accepts correctly masked inputs; the problem is the supplied preparation
path for multi-turn preference examples.

**Reproduction:** The labeled text includes
`history<|im_end|>rejected<|im_end|>` instead of only the final reply. In the probe,
masking historical turns changes ORPO loss from about `5.7629` to `5.7529` without
changing any input tokens or model weights.

**Suggested fix:** Add preference-specific encoding that masks the complete
prefix and labels only the candidate completion and its closing token. Retain
all-assistant-turn labeling for SFT. Use the preference encoder in examples.

**Regression check:** Verify that historical assistant tokens contribute neither
to the preference likelihood nor its length denominator. Keep a separate SFT
check demonstrating that earlier assistant turns remain labeled there.

Reference: [ORPO, section 4.1, completion log-likelihood](https://arxiv.org/html/2403.07691v2#S4.SS1).

## 4. Medium: Memory Tuning Cannot Disable Harmful Voting

**Location:** `memory.py:222-231`, mixing-weight search.

The search only tries `lam` values from `0.05` to `0.5`. It must select voting even
when every candidate is worse than the plain model.

**Reproduction:** With true-token model probability `0.9` and all retrieved
neighbors voting for the wrong token, tuning selects `lam=0.05`. Loss increases
from `0.1053605` to `0.1566538` nats.

**Suggested fix:** Include `lam=0` and initialize the search with the plain-model
baseline. When comparing against the plain model, also disable recall. Preserve
the distinction between choosing voting and subsequently tuning recall within
its documented tolerance.

**Regression check:** An all-wrong memory should select no voting and match the
plain loss; helpful neighbors should still allow a positive mixing weight.

## 5. Medium: Context Length One Makes Memory Construction Hang

**Location:** `memory.py:84-92`, overlapping-window construction.

With `context=1`, `half=context//2` is zero. On a document longer than one token,
the loop repeatedly appends windows without advancing `pos`, growing memory
without bound. There is no argument validation preventing this input.

**Suggested fix:** Reject context lengths below two with a clear `ValueError`,
or implement a dedicated length-one path. Merely changing the stride also
requires checking which positions each window retains.

**Regression check:** `Memory.build(model, tensor([2, 3, 4]), context=1)` must
return valid results or fail promptly, without allocating an unbounded list.

## 6. Low: Training Omits the Final Valid Window

**Location:** `train.py:102`, `get_batch`.

`torch.randint` has an exclusive upper bound. A data sequence of length `N` with
context `C` permits starts `0` through `N-C-1`, inclusive. The current upper bound
`N-C-1` excludes the last valid start, so the corpus's last token is never a target.

**Reproduction:** For `arange(5)` and context three, starts should include zero
and one. The current implementation samples only zero, and targets never include
token four.

**Suggested fix:** Use exclusive upper bound `len(data)-context`; revisit the
minimum-corpus-length guard, since `context+1` tokens are sufficient for one
window. Preserve deterministic data-generator behavior for resume checks.

**Regression check:** Exercise a sequence with two valid starts and a sequence
with exactly one valid start.

## 7. Low: fp16 Muon Conversion Happens Before Safe Scaling

**Location:** `train.py:192-196`, `newton_schulz`.

Large finite float32 gradients can overflow during `g.to(torch.float16)` before
normalization gets a chance to bound them. For `randn(16,16)*1e5`, the fp16 path
produces nonfinite output. The current trainer clips gradients first, limiting
exposure; this primarily affects direct optimizer/helper use.

**Suggested fix:** Normalize in float32 before converting the bounded matrix to
the requested product dtype. Preserve per-matrix normalization for stacks and
the intended treatment of zero and very small gradients.

**Regression check:** Check finite directions across input scales, zero inputs,
and both orientations; retain the existing stacked/single-matrix comparisons.

## 8. Compatibility: The Training Example Does Not Support PyTorch 2.1

**Location:** `model.py:7`, its minimal training example at `model.py:38`, and
`train.py:394`.

The model advertises PyTorch `>=2.1`, but the example and trainer call
`torch.amp.GradScaler`, which PyTorch 2.1 does not expose. The trainer evaluates
that attribute even when scaling would be disabled on CPU.

**Suggested fix:** State separate, accurate version requirements for the model
and trainer, or use a compatible scaler fallback. PyTorch 2.3 exposes the newer
namespace; document and test whichever minimum is chosen.

This compatibility finding was verified against published source rather than
by installing an old PyTorch runtime:
[PyTorch 2.1 AMP exports](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/amp/__init__.py),
[PyTorch 2.3 AMP exports](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/amp/__init__.py).

## Implementation Handoff

Address the nonfinite-update guard and voting normalization first. Add regression
checks for each fix in `tests.py` and invoke them in its main block. Re-run the
existing suite and the diagnostic probes. Validate the AMP guard on a CUDA device
before claiming GPU coverage. Keep private books, weights, and downloaded test
environments outside commits. Update this document's finding status as fixes land.
