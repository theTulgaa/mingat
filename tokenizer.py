"""
tokenizer.py — byte-level BPE tokenizer, written from scratch
=============================================================

Text -> token ids -> text, with no dependencies. The round trip is lossless for any string.

How it works
    1. The text is cut into pieces (PRETOKEN): a word with its leading space, a group of at
       most 2 digits, a run of punctuation, or whitespace. Merges never cross a piece.
    2. Every piece starts as its UTF-8 bytes. A Cyrillic letter is 2 bytes, so the first
       ~70 merges simply rebuild the alphabet.
    3. Training repeats one step: find the most frequent adjacent pair of tokens and replace
       it with a new token. Each merge adds one token to the vocabulary.
    4. Encoding replays the merges on new text in the order they were learned.

Why these choices suit a small dataset
    * Small vocabulary. A token that occurs 3 times in the corpus cannot be learned. With
      1 MB of text, 2,048 tokens is about right; 32,768 would leave most tokens almost unseen.
    * Digits in groups of 2. A year is always two tokens, century then year: 1206 -> "12" "06".
      Rare years then share their pieces with common ones instead of being split at random.
    * Bytes underneath. A word or symbol that never appeared in training still encodes.

Token ids
    0..7      special tokens (SPECIAL_TOKENS): fixed ids, whatever the vocabulary size
    8..263    the 256 byte values
    264..     learned merges, in the order they were learned

Usage
    tok = BPETokenizer.train([text], vocab_size=2048)
    ids = tok.encode("Чингис хаан 1162 онд төрсөн.")
    tok.decode(ids)
    tok.save("out/tokenizer.json");  tok = BPETokenizer.load("out/tokenizer.json")

    python tokenizer.py                       # self-test
    python tokenizer.py corpus.txt 2048       # train on a file and print statistics
"""

from __future__ import annotations

import heapq
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SPECIAL_TOKENS = [
    "<|pad|>",          # 0  padding
    "<|endoftext|>",    # 1  document separator in pretraining
    "<|im_start|>",     # 2  start of a chat turn (ChatML)
    "<|im_end|>",       # 3  end of a chat turn; the stop token for generation
    "<|mask|>",         # 4  reserved for a masked objective (train.py does not use it)
    "<|reserved_0|>", "<|reserved_1|>", "<|reserved_2|>",   # 5..7 free for later use
]
SPECIAL_ID = {s: i for i, s in enumerate(SPECIAL_TOKENS)}
N_SPECIAL = len(SPECIAL_TOKENS)
FIRST_MERGE = N_SPECIAL + 256

# The pieces, tried in this order: letters, 1-2 digits, punctuation (each with an optional
# leading space), newlines, other whitespace, and finally any single character.
PRETOKEN = re.compile(r" ?[^\W\d_]+| ?\d{1,2}| ?[^\s\w]+|\n+|\s+(?!\S)|\s+|.", re.DOTALL)
SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")")

Pair = Tuple[int, int]


def _merge_pair(symbols: List[int], pair: Pair, new_id: int) -> List[int]:
    """Replace every non-overlapping occurrence of `pair`, scanning left to right."""
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def _count_pieces(texts: Iterable[str]) -> Counter:
    counts: Counter = Counter()
    for text in texts:
        for part in SPECIAL_RE.split(text):       # special tokens are never split or merged
            if part and part not in SPECIAL_ID:
                counts.update(PRETOKEN.findall(part))
    return counts


class BPETokenizer:
    def __init__(self, merges: List[Pair]):
        self.merges = [tuple(m) for m in merges]
        self.rank: Dict[Pair, int] = {m: i for i, m in enumerate(self.merges)}
        self.vocab: List[bytes] = [s.encode("utf-8") for s in SPECIAL_TOKENS]
        self.vocab += [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self._cache: Dict[str, List[int]] = {}
        self.pad_id, self.eot_id, self.im_start_id, self.im_end_id, self.mask_id = range(5)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # -- training --------------------------------------------------------------------------
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, min_count: int = 2) -> "BPETokenizer":
        """Learn up to `vocab_size - 264` merges. Training stops early when no pair occurs
        `min_count` times, so check `tok.vocab_size` afterwards.

        Counts are kept per unique piece, and after a merge only the pieces that contained
        the pair are recounted. A heap gives the most frequent pair; entries whose count has
        changed since they were pushed are skipped. Ties go to the pair with the smaller ids,
        so training is deterministic."""
        if vocab_size < FIRST_MERGE:
            raise ValueError(f"vocab_size must be at least {FIRST_MERGE}")
        counts = _count_pieces(texts)
        words = [[b + N_SPECIAL for b in w.encode("utf-8")] for w in counts]
        freqs = list(counts.values())
        pair_count: Dict[Pair, int] = defaultdict(int)
        pair_words: Dict[Pair, set] = defaultdict(set)
        for wi, symbols in enumerate(words):
            for pair in zip(symbols, symbols[1:]):
                pair_count[pair] += freqs[wi]
                pair_words[pair].add(wi)
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        merges: List[Pair] = []
        while len(merges) < vocab_size - FIRST_MERGE and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg_count:
                continue                                   # stale entry
            if -neg_count < min_count:
                break
            new_id = FIRST_MERGE + len(merges)
            merges.append(pair)
            touched = set()
            for wi in list(pair_words[pair]):
                symbols, f = words[wi], freqs[wi]
                for p in zip(symbols, symbols[1:]):        # take this piece's pairs out
                    pair_count[p] -= f
                    pair_words[p].discard(wi)
                    touched.add(p)
                symbols = words[wi] = _merge_pair(symbols, pair, new_id)
                for p in zip(symbols, symbols[1:]):        # and put the new ones in
                    pair_count[p] += f
                    pair_words[p].add(wi)
                    touched.add(p)
            for p in touched:
                if pair_count[p] > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
                else:
                    del pair_count[p]
                    pair_words.pop(p, None)
        tok = cls(merges)
        tok._trained_pieces = dict(zip(counts, words))     # kept for the self-test only
        return tok

    # -- encoding and decoding -------------------------------------------------------------
    def _encode_piece(self, piece: str) -> List[int]:
        ids = self._cache.get(piece)
        if ids is None:
            ids = [b + N_SPECIAL for b in piece.encode("utf-8")]
            while len(ids) >= 2:
                # The earliest-learned merge that applies goes first, as in training.
                pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, len(self.rank)))
                if pair not in self.rank:
                    break
                ids = _merge_pair(ids, pair, FIRST_MERGE + self.rank[pair])
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, allow_special: bool = False) -> List[int]:
        """Text -> ids. With allow_special=False (the default) a string such as "<|im_end|>"
        is encoded as ordinary characters. Use that for anything a user typed, so nobody
        can inject a control token. Use allow_special=True for text you formatted yourself."""
        ids: List[int] = []
        for part in (SPECIAL_RE.split(text) if allow_special else [text]):
            if allow_special and part in SPECIAL_ID:
                ids.append(SPECIAL_ID[part])
            elif part:
                for piece in PRETOKEN.findall(part):
                    ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Ids -> text. Bytes that do not form valid UTF-8 (a model can emit half a letter)
        become the replacement character."""
        data = b"".join(self.vocab[i] for i in ids if not (skip_special and i < N_SPECIAL))
        return data.decode("utf-8", errors="replace")

    # -- saving and loading ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"type": "byte-bpe", "version": 1, "special_tokens": SPECIAL_TOKENS,
                "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, d: dict) -> "BPETokenizer":
        if d.get("special_tokens") != SPECIAL_TOKENS:
            raise ValueError("this tokenizer file was made with different special tokens")
        return cls([tuple(m) for m in d["merges"]])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------------------------
# Self-test:  python tokenizer.py
# ----------------------------------------------------------------------------------------
def _train_reference(texts: List[str], vocab_size: int, min_count: int = 2) -> List[Pair]:
    """The textbook algorithm: recount every pair after every merge. Slow, but obviously
    right. The fast trainer must produce exactly the same merges."""
    counts = _count_pieces(texts)
    words = [[b + N_SPECIAL for b in w.encode("utf-8")] for w in counts]
    freqs = list(counts.values())
    merges: List[Pair] = []
    while len(merges) < vocab_size - FIRST_MERGE:
        pair_count: Counter = Counter()
        for symbols, f in zip(words, freqs):
            for p in zip(symbols, symbols[1:]):
                pair_count[p] += f
        if not pair_count:
            break
        pair = min(pair_count, key=lambda p: (-pair_count[p], p))
        if pair_count[pair] < min_count:
            break
        new_id = FIRST_MERGE + len(merges)
        merges.append(pair)
        words = [_merge_pair(symbols, pair, new_id) for symbols in words]
    return merges


_SAMPLE = (
    "Чингис хаан 1162 онд Онон мөрний Дэлүүн болдогт төрсөн гэж үздэг. Тэмүжин 1206 онд "
    "Их Монгол Улсыг байгуулж, Чингис хаан хэмээх цол авчээ.\n\nӨгэдэй хаан 1229 онд хаан "
    "ширээнд сууж, 1235 онд Хархорум хотыг байгуулав. Хубилай хаан 1271 онд Юань улсыг "
    "тунхаглав. Монголын нууц товчоо бол XIII зууны түүхэн сурвалж юм.\n"
    "Түүхийг мэдэхгүй хүн ойд төөрсөн сармагчинтай адил гэдэг: энэ үг үнэн үү? 100%, 3.14, a_b\n"
) * 3


def _self_test() -> None:
    text = _SAMPLE
    # 1) The fast trainer matches the textbook algorithm, merge for merge.
    for size in (FIRST_MERGE, 300, 400, 5000):
        fast = BPETokenizer.train([text], size)
        assert fast.merges == _train_reference([text], size), f"merges differ at vocab {size}"
    tok = BPETokenizer.train([text], 400)
    print(f"[check] fast trainer == reference trainer ({len(tok.merges)} merges, vocab {tok.vocab_size})")

    # 2) Encoding a training piece gives the segmentation the trainer ended with.
    for piece, symbols in tok._trained_pieces.items():
        assert tok._encode_piece(piece) == symbols, piece
    print(f"[check] encode replays training exactly on all {len(tok._trained_pieces)} training pieces")

    # 3) Lossless round trip, including text the tokenizer never saw.
    tricky = [text, "", " ", "  хоёр  зай,\nшинэ мөр\t таб  ", "ҮҮЛ өөр Ёс ё Ъ", "1206-1227 он, 13-р зуун",
              "English, 中文, emoji 🙂, a_b_c __init__", "\n\n\n", "<|im_end|> typed by a user", "12345678901"]
    for s in tricky:
        assert tok.decode(tok.encode(s)) == s, repr(s)
        assert "".join(PRETOKEN.findall(s)) == s
    print(f"[check] decode(encode(s)) == s on {len(tricky)} tricky strings")

    # 4) Digits come in groups of at most 2, and merges never cross a group.
    year = [tok.decode([i]) for i in tok.encode(" 1206 онд")]
    assert year[:2] == [" 12", "06"], year
    assert all(len(tok.decode([i]).strip()) <= 2 for i in tok.encode("12345678901"))
    print(f"[check] ' 1206 онд' -> {year}")

    # 5) Special tokens: only when allowed, and they survive a save/load.
    chat = "<|im_start|>user\nСайн уу?<|im_end|>\n"
    with_special, plain = tok.encode(chat, allow_special=True), tok.encode(chat)
    assert with_special[0] == tok.im_start_id and tok.im_end_id in with_special
    assert all(i >= N_SPECIAL for i in plain) and tok.decode(plain) == chat
    assert tok.decode(with_special) == "user\nСайн уу?\n"
    assert tok.decode(with_special, skip_special=False) == chat
    again = BPETokenizer.from_dict(json.loads(json.dumps(tok.to_dict())))
    assert again.merges == tok.merges and again.encode(text) == tok.encode(text)
    print("[check] special tokens are opt-in; save/load round trip is identical")
    print("self-test passed")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) >= 3:
        corpus = Path(sys.argv[1]).read_text(encoding="utf-8")
        tok = BPETokenizer.train([corpus], int(sys.argv[2]))
        ids = tok.encode(corpus, allow_special=True)
        print(f"vocab {tok.vocab_size:,} | {len(ids):,} tokens | {len(corpus) / len(ids):.2f} chars/token | "
              f"{len(corpus.encode('utf-8')) / len(ids):.2f} bytes/token")
        seen = Counter(ids)
        rare = sum(1 for i in range(FIRST_MERGE, tok.vocab_size) if seen[i] < 5)
        print(f"merged tokens seen fewer than 5 times: {rare:,} of {len(tok.merges):,}")
        print("last merges:", [tok.decode([FIRST_MERGE + i]) for i in range(max(0, len(tok.merges) - 12), len(tok.merges))])
        if len(sys.argv) >= 4:
            tok.save(sys.argv[3])
            print("saved", sys.argv[3])
    else:
        _self_test()
