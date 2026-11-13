"""
tests.py — every check in one place
===================================

    python tests.py        # about a minute on a CPU

It runs the self-tests of tokenizer.py, model.py and persona.py, checks that program_qa.py
still reproduces its answers, and then tests the training, generation and memory math directly.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as F

import generate
import prepare
import train
from memory import RECALL_K, Memory, collect, mixture_loss, prompt_starts, recitation
from model import ModelConfig, MongolHistorySLM
from tokenizer import _SAMPLE, BPETokenizer

HERE = Path(__file__).resolve().parent
TINY = dict(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, ffn_hidden=96, max_seq_len=128)


def run_script(name: str, expect: str, *args: str) -> None:
    done = subprocess.run([sys.executable, "-W", "ignore", str(HERE / name), *args], capture_output=True,
                          text=True, encoding="utf-8", cwd=HERE)
    assert done.returncode == 0 and expect in done.stdout, f"{name} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    print(f"[ok] {name}: {expect}")


def test_muon() -> None:
    torch.manual_seed(0)
    low, high, cosine = 9.0, 0.0, 1.0
    for shape in ((96, 32), (32, 96), (688, 128), (128, 256)):      # the shapes of real weight matrices
        g = torch.randn(shape)
        o = train.newton_schulz(g)
        s = torch.linalg.svdvals(o)
        u, _, vh = torch.linalg.svd(g, full_matrices=False)
        exact = u @ vh                                              # the true nearest orthogonal matrix
        low, high = min(low, s.min().item()), max(high, s.max().item())
        cosine = min(cosine, ((o * exact).sum() / (o.norm() * exact.norm())).item())
    assert 0.6 < low and high < 1.25 and cosine > 0.95, (low, high, cosine)

    # One Muon step moves a square matrix by about lr in every direction, whatever the gradient's scale.
    for grad_scale in (1e-3, 1e3):
        w = torch.nn.Parameter(torch.zeros(48, 48))
        opt = train.Muon([w], lr=0.02, weight_decay=0.0)
        w.grad = torch.randn(48, 48) * grad_scale
        opt.step()
        size = torch.linalg.matrix_norm(w.detach(), ord=2).item()
        assert 0.6 * 0.02 < size < 1.25 * 0.02, size
    print(f"[ok] Muon: Newton-Schulz output has singular values in [{low:.2f}, {high:.2f}] and cosine "
          f"{cosine:.3f} to the exact orthogonal factor; the step size ignores the gradient's scale")


def test_schedule_and_ema() -> None:
    scales = [train.lr_scale(s, 1000, 100) for s in range(1000)]
    assert abs(scales[0] - 0.01) < 1e-9 and scales[99] == 1.0 and abs(scales[-1] - 0.1) < 1e-3
    assert all(a >= b - 1e-12 for a, b in zip(scales[99:], scales[100:]))      # never rises after warmup
    model = torch.nn.Linear(4, 4)
    ema = train.EMA(model, decay=0.9)
    start = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    decay = min(0.9, 2 / 11)                                        # first update: (1 + 1) / (10 + 1)
    for s, a, p in zip(ema.shadow, start, model.parameters()):
        assert torch.allclose(s, decay * a + (1 - decay) * p)
    live = [p.detach().clone() for p in model.parameters()]
    with ema.applied(model):
        assert all(torch.equal(p, s) for p, s in zip(model.parameters(), ema.shadow))
    assert all(torch.equal(p, w) for p, w in zip(model.parameters(), live))    # restored exactly
    print("[ok] learning-rate schedule and weight averaging")


def test_hold_out_and_evaluate() -> None:
    text = "".join(f"мөр {i}: Чингис хаан {1162 + i} он.\n" for i in range(2000))
    tr, va = train.hold_out(text)
    lines = lambda t: [l for l in t.replace("<|endoftext|>", "").split("\n") if l]
    assert sorted(lines(tr) + lines(va)) == sorted(lines(text))     # every line lands in exactly one part
    assert 0.03 < len(va) / len(text) < 0.07
    for odd in ("үг " * 5000, "ү" * 5000, "нэг мөр\n" + "үг " * 3000):         # no line ends to cut at
        tr2, va2 = train.hold_out(odd)
        kept = va2.replace("\n<|endoftext|>\n", "")
        assert len(tr2) + len(kept) == len(odd) and 0.03 < len(kept) / len(odd) < 0.07, (len(tr2), len(kept))
    torch.manual_seed(0)
    model = MongolHistorySLM(ModelConfig(vocab_size=300, **TINY)).eval()
    ids = torch.randint(0, 300, (1000,), dtype=torch.int32)
    total = 0.0
    with torch.no_grad():
        for s in range(0, 999, 64):
            x, y = ids[s:min(s + 64, 999)].long()[None], ids[s + 1:min(s + 65, 1000)].long()[None]
            total += F.cross_entropy(model(x)[0][0], y[0], reduction="sum").item()
    assert abs(train.evaluate(model, ids, 64, 5, "cpu") - total / 999) < 1e-4
    sampled = 0.0                                                   # with a token budget: every 4th window, spread out
    with torch.no_grad():
        for s in (0, 256, 512, 768):
            sampled += F.cross_entropy(model(ids[s:s + 64].long()[None])[0][0], ids[s + 1:s + 65].long(), reduction="sum").item()
    assert abs(train.evaluate(model, ids, 64, 5, "cpu", max_tokens=256) - sampled / 256) < 1e-4
    print(f"[ok] hold-out split ({len(va) / len(text):.1%} held out, nothing lost) and evaluation loss")


def _write_epub(path: Path, pages: dict) -> None:
    """A minimal EPUB: container.xml -> content.opf -> the pages, in the given order."""
    items = "".join(f'<item id="p{i}" href="{name}" media-type="application/xhtml+xml"/>' for i, name in enumerate(pages))
    spine = "".join(f'<itemref idref="p{i}"/>' for i in range(len(pages)))
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
        z.writestr("OEBPS/content.opf", f'<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf">'
                   f'<manifest>{items}</manifest><spine>{spine}</spine></package>')
        for name, body in pages.items():
            z.writestr(f"OEBPS/{name}", f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title>'
                       f'<style>p {{}}</style></head><body>{body}</body></html>')


def test_prepare() -> None:
    """prepare.py keeps the book's text and removes the e-library's additions, footnotes,
    the colophon and foreign lines; it repairs look-alike letters and skips a duplicate."""
    story = "".join(f"<p>Тэр өдөр {i} дахь удаагаа тал хээрээр давхиж, нар жаргах үед гэртээ буцаж ирэв.</p>" for i in range(60))
    pages = {
        "front.xhtml": "<p>Ц. ЗОХИОЛЧ</p><h1>ТАЛЫН ҮЛГЭР</h1>"
                       "<p>“Улсын хэвлэлийн газар” Улаанбаатар хотноо 1990 онд эрхлэн хэвлүүлсэн номыг</p>"
                       "<p>Цогт охин тэнгэр сангийн захиалгаар Цагаан бамбарууш хэвлэлийн газарт 2021 онд цахим хэлбэрт хөрвүүлэв.</p>"
                       "<p>ЦАХИМ НОМЫН САН: www.e-nom.mn</p><p>Уг номыг “Болор дуран” шалгуураар шүүсэн болно.</p>",
        "text.xhtml": '<p>Чингис хаан<sup><a href="notes.xhtml#note_1">1</a></sup> <a href="#x">Хэрлэн</a> мөрөнд буув.</p>'
                      "<p>Тэр тoгтooл гаргаж, ХIХ зуунд гap бариад явав.</p><p>Le Comte de Monte-Cristo est un roman</p>"
                      + story +
                      "<p>Уг номыг “Болор дуран” шалгуураар шүүсэн болно.</p><p>ӨГЛӨГИЙН ЭЗНИЙ СОНОРТ</p>"
                      "<p>Манай номын санд байгаа бүх номыг та уншиж болно.</p><p>5027 9685 00 (доллар) Tsogt okhin Tenger san</p>"
                      "<p>Qpay-гээр бол (линк)</p><p>1990 оны хэвлэлийн Хэвлэн нийтлэлийн мэдээлэл</p>"
                      "<p>Цаасны хэмжээ 84x108/32</p><p>Үнэ 4 тө 70 мө</p>",
        "notes.xhtml": '<p>Заметки</p><p><a href="text.xhtml#back_note_1">[←1]</a></p><p>Хаан — монгол улсын тэргүүн</p>',
    }
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        src, out = Path(tmp) / "src", Path(tmp) / "out"
        src.mkdir()
        _write_epub(src / "book.epub", pages)
        copy = prepare.clean_book(prepare.read_epub(src / "book.epub"))[0]
        (src / "copy.txt").write_text(copy[:len(copy) * 9 // 10], encoding="utf-8")   # 90% of the same book
        (src / "notes.pdf").write_bytes(b"%PDF-1.4")
        books = prepare.main(["--src", str(src), "--out", str(out)])
        text = (out / "book.txt").read_text(encoding="utf-8")
    lines = text.split("\n")
    assert lines[0] == "Ц. ЗОХИОЛЧ" and lines[2] == "ТАЛЫН ҮЛГЭР", lines[:4]
    assert "Чингис хаан Хэрлэн мөрөнд буув." in text                         # footnote number gone, plain link kept
    assert "Тэр тогтоол гаргаж, XIX зуунд гар бариад явав." in text           # look-alikes repaired
    assert text.rstrip().endswith("гэртээ буцаж ирэв.") and "Тэр өдөр 59 дахь" in text   # the story is all there
    for junk in ("e-nom", "Болор", "хөрвүүлэв", "эрхлэн", "СОНОРТ", "Tsogt", "Qpay", "Хэвлэн", "Цаасны", "Үнэ",
                 "Заметки", "монгол улсын тэргүүн", "Le Comte"):
        assert junk not in text, junk
    assert books["copy"]["duplicate_of"] == "book" and not (out / "copy.txt").exists()
    assert "notes" not in books
    assert prepare.fix_word("Тomus") == "Tomus" and prepare.fix_word("mанхимын") == "танхимын"
    assert prepare.fix_word("Шpu") == "Шpu"                                   # undecidable: left as it is
    print("[ok] prepare: book text kept, library text / footnotes / colophon / foreign lines removed, "
          "look-alike letters repaired, a 90% copy skipped")


def test_training_and_resume() -> None:
    """A short real run must learn, and a run paused and resumed must match an unbroken run."""
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        tmp, report = Path(tmp), []
        (tmp / "corpus.txt").write_text(_SAMPLE * 12, encoding="utf-8")
        for optimizer in ("adamw", "muon"):
            common = ["--data", str(tmp / "corpus.txt"), "--vocab-size", "400", "--context", "32", "--batch", "8",
                      "--steps", "60", "--eval-interval", "20", "--warmup", "10", "--dropout", "0.1",
                      "--optimizer", optimizer, "--threads", "2", "--sample", ""]
            straight = train.main(common + ["--out", str(tmp / optimizer / "a")])
            first, last = straight["history"][0], straight["history"][-1]
            assert last[1] < first[1] - 0.5 and last[2] < first[2] - 0.5, (first, last)     # train and val loss fell
            paused = train.main(common + ["--out", str(tmp / optimizer / "b"), "--stop-after", "33"])
            assert paused == {"paused_at": 33}
            resumed = train.main(common + ["--out", str(tmp / optimizer / "b"), "--resume"])
            assert resumed["history"] == straight["history"], "resume changed the run"
            a, b = (torch.load(tmp / optimizer / d / "model.pt", weights_only=False)["model"] for d in ("a", "b"))
            assert all(torch.equal(a[k], b[k]) for k in a)
            report.append(f"{optimizer} {first[1]:.2f} -> {last[1]:.2f}")

        # generate.py continues a prompt with a trained model. The same seed gives the same text,
        # and a prompt longer than the context window keeps its end instead of failing.
        trained = str(tmp / "adamw" / "a" / "model.pt")
        once = generate.main(["--model", trained, "--prompt", "Чингис хаан", "--tokens", "8", "--seed", "0"])
        again = generate.main(["--model", trained, "--prompt", "Чингис хаан", "--tokens", "8", "--seed", "0"])
        assert once == again and once[0].startswith("Чингис хаан"), (once, again)
        long = generate.main(["--model", trained, "--prompt", "Чингис хаан 1162 онд төрсөн. " * 200, "--tokens", "8"])
        assert len(long) == 1 and "төрсөн" in long[0]

        # --init continues a trained model on new text: same tokenizer and size, and it starts
        # from what the model already knows, not from scratch.
        (tmp / "more.txt").write_text(_SAMPLE * 6 + "Мөнх хаан 1251 онд хаан ширээнд суув.\n" * 40, encoding="utf-8")
        more = train.main(["--data", str(tmp / "more.txt"), "--init", str(tmp / "adamw" / "a" / "model.pt"),
                           "--out", str(tmp / "c"), "--context", "32", "--batch", "8", "--steps", "20",
                           "--eval-interval", "10", "--warmup", "5", "--lr", "5e-4", "--threads", "2", "--sample", ""])
        start = torch.load(tmp / "adamw" / "a" / "model.pt", weights_only=False)
        after = torch.load(tmp / "c" / "model.pt", weights_only=False)
        assert after["tokenizer"] == start["tokenizer"] and more["vocab"] == straight["vocab"]
        assert {**after["config"], "dropout": 0} == {**start["config"], "dropout": 0}
        assert more["history"][0][0] == 0                       # the starting weights are scored before any step
        assert more["history"][0][2] < first[2] - 1.0, (more["history"][0], first)
        assert more["val"]["loss"] <= more["history"][0][2] + 1e-6   # so the result is never worse than the start

        # A continuation that can only do harm (a huge learning rate) hands back the starting weights.
        harm = train.main(["--data", str(tmp / "more.txt"), "--init", str(tmp / "adamw" / "a" / "model.pt"),
                           "--out", str(tmp / "d"), "--context", "32", "--batch", "8", "--steps", "20",
                           "--eval-interval", "10", "--warmup", "1", "--lr", "0.5", "--threads", "2", "--sample", ""])
        kept = torch.load(tmp / "d" / "model.pt", weights_only=False)
        assert harm["best_raw"][1] == 0 and all(torch.equal(kept["model"][k], start["model"][k]) for k in start["model"])
    print(f"[ok] training: loss in 60 steps, {' and '.join(report)}; pause + resume reproduces the unbroken run exactly; "
          f"generate.py continues a prompt, the same seed gives the same text")


def test_memory() -> None:
    torch.manual_seed(0)
    docs = [_SAMPLE.split("\n\n")[0], "Өгэдэй хаан 1229 онд хаан ширээнд суув. Тэрээр Хархорум хотыг байгуулав.",
            "Хубилай хаан 1271 онд Юань улсыг тунхаглав. Нийслэлээ Дайду хотод байгуулав.",
            "Монголын нууц товчоо бол XIII зууны түүхэн сурвалж юм. Зохиогч нь тодорхойгүй."]
    tok = BPETokenizer.train(docs, 330)
    ids = torch.tensor(tok.encode("".join("<|endoftext|>" + d * 2 for d in docs), allow_special=True))
    model = MongolHistorySLM(ModelConfig(vocab_size=tok.vocab_size, **TINY)).eval()   # untrained on purpose
    mem = Memory.build(model, ids, context=32, eot_id=tok.eot_id)
    assert len(mem) == len(ids) - 1 and torch.equal(mem.values, ids[1:])
    assert torch.allclose(mem.keys.norm(dim=-1), torch.ones(len(mem)), atol=1e-5)
    again = Memory.build(model, ids, context=32, eot_id=tok.eot_id, batch=1)
    assert torch.allclose(mem.keys, again.keys, atol=1e-5)            # batching does not change the keys

    # A stored context retrieves itself, and with it the token that followed. The first two
    # positions are left out: every document starts with <|endoftext|>, and three of these
    # start with the same byte after it, so there the memory rightly holds several
    # continuations for one context.
    with torch.no_grad():
        h = model.forward_hidden(ids[None, :24])[0]
        logits = model.lm_head(h)
    sims, values = mem.neighbours(h, k=4)
    assert (sims[:, 0] > 0.9999).all() and torch.equal(values[2:, 0], ids[3:25])
    assert (sims[2:, 1] < 0.999).all()                                # and no other context ties with it

    # The mixture is a probability distribution with voting, with recall, with both, and
    # with neither (then it is the model's own distribution).
    logp_model = logits.log_softmax(-1)
    for lam, recall in ((0.2, None), (0.0, 0.005), (0.2, 0.005), (0.0, None)):
        mem.lam, mem.recall_temperature = lam, recall
        mixed = mem.mix(h, logits)
        assert torch.allclose(mixed.exp().sum(-1), torch.ones(len(h)), atol=1e-5)
    assert torch.allclose(mixed, logp_model, atol=1e-5)

    # Recall on the real keys: at a stored context the memory's top choice is the stored token.
    mem.lam, mem.recall_temperature = 0.2, 0.005
    assert torch.equal(mem.mix(h, logits).argmax(-1)[2:], ids[3:25])

    # Recall arithmetic on hand-made neighbours, against a model that is 99.9% sure of token 7.
    sure = torch.full((1, 50), -12.0)
    sure[0, 7] = 0.0
    sure = sure.log_softmax(-1)
    far = torch.full((1, RECALL_K), 0.5)                 # weight exp(-100): not a match
    stored = torch.full((1, RECALL_K), 3)
    mem.lam = 0.0
    recall = lambda sims, values: mem.combine(sure, sims, values).exp()[0]
    assert torch.allclose(recall(far, stored), sure.exp()[0], atol=1e-6)          # nothing matches: the model alone
    one = far.clone()
    one[0, 0] = 1.0
    assert recall(one, stored)[3] >= 0.5                                          # one match: recited (N=1, T=1)
    split = stored.clone()
    split[0, 1] = 4
    two = one.clone()
    two[0, 1] = 1.0
    p = recall(two, split)                                                        # two matches that disagree (N=2, T=2)
    assert abs(p[3] - p[4]) < 1e-6 and 0.25 <= p[3] < 0.26 and p.argmax() == 7    # ... so the model decides
    assert recall(torch.ones(1, RECALL_K), stored)[3] >= RECALL_K / (RECALL_K + 1)   # 16 that agree: trusted
    mem.lam = 0.2

    # The tuner's formula agrees with the formula used during generation.
    mem.temperature, mem.k = 0.05, 8
    held_out = torch.tensor(tok.encode("<|endoftext|>Өгэдэй хаан 1235 онд Хархорум хотыг байгуулав. Хубилай хаан Юань улсыг тунхаглав."), dtype=torch.long)
    assert len(held_out) <= 128
    got = collect(model, mem, held_out, context=len(held_out) - 1, k=16)     # one window, as below
    with torch.no_grad():
        hh = model.forward_hidden(held_out[None, :-1])[0]
    for recall in (None, 0.005, 0.05):
        mem.recall_temperature = recall
        direct = -mem.mix(hh, model.lm_head(hh)).gather(1, held_out[1:, None]).mean().item()
        assert abs(mixture_loss(*got, k=8, temperature=0.05, lam=0.2, recall_temperature=recall) - direct) < 1e-5

    # Saving in half precision and loading again leaves the predictions unchanged.
    mem.recall_temperature = 0.005
    with tempfile.TemporaryDirectory() as tmp:
        mem.save(Path(tmp) / "memory.pt")
        loaded = Memory.load(Path(tmp) / "memory.pt")
    assert (loaded.k, loaded.temperature, loaded.lam, loaded.recall_temperature) == (8, 0.05, 0.2, 0.005)
    assert torch.allclose(loaded.mix(h, logits).exp(), mem.mix(h, logits).exp(), atol=2e-2)

    # Test prompts: one per document start, or many inside the documents; a book with no
    # separators still gives prompts.
    doc_starts = (ids == tok.eot_id).nonzero().flatten().tolist()
    assert prompt_starts(ids, tok.eot_id, 24) == doc_starts and len(doc_starts) == 4
    inside = prompt_starts(ids, tok.eot_id, 24, offset=10)
    assert len(inside) > 4 and all(any(s + 10 <= p and p + 24 <= e for s, e in zip(doc_starts, doc_starts[1:] + [len(ids)])) for p in inside)
    book = ids[ids != tok.eot_id]
    assert prompt_starts(book, tok.eot_id, 24) == [0] and len(prompt_starts(book, tok.eot_id, 24, offset=10)) >= 10
    assert len(prompt_starts(book, tok.eot_id, 24, offset=10, limit=5)) <= 5

    # An untrained model cannot continue its documents; with the memory it recites them.
    without = recitation(model, None, ids, tok.eot_id, prompt_len=8, target_len=16)
    with_mem = recitation(model, mem, ids, tok.eot_id, prompt_len=8, target_len=16)
    assert without[0] < 0.2 and with_mem == (1.0, 1.0, 4), (without, with_mem)
    print(f"[ok] memory: stored contexts retrieve themselves; mixture sums to 1; an untrained model recites "
          f"{without[0]:.0%} of its corpus alone and {with_mem[0]:.0%} with the memory")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    run_script("tokenizer.py", "self-test passed")
    run_script("model.py", "self-test passed")
    run_script("persona.py", "self-test passed")
    run_script("program_qa.py", "12/12 answers exact")
    test_muon()
    test_schedule_and_ema()
    test_hold_out_and_evaluate()
    test_prepare()
    test_training_and_resume()
    test_memory()
    print("\nall tests passed")
