"""
prepare.py — turn books into clean training text
================================================

    python prepare.py                                   # data/novels/ -> data/clean/
    python prepare.py --src data/novels --out data/clean
    python train.py --data data/clean                   # then train on the result

Every .epub and .txt file in --src becomes one clean UTF-8 .txt file in --out. A report
follows: what each book gave, what was removed, and anything that looks wrong.

What is removed, and why. The books so far all come from the e-nom.mn digital library, so
some rules name its exact phrases; the others are general.

    footnote numbers  "Шеридан1": the number of a footnote link, glued to the word before it.
                      Links whose whole text is a number or an asterisk are left out.
    footnotes         The notes section at the end ("Заметки"): glossary entries and
                      references, cut off from the text they explain.
    library text      The e-library's banner, advertisement, quality note, digitization
                      credit, download links and donation appeal (with bank accounts), and
                      the note on which printed edition was scanned.
    colophon          The printer's data after the text: editors, paper size, print run,
                      price, press address, ISBN.
    foreign lines     Lines that are mostly Latin letters: original titles, French or English
                      summaries, transliterations, bibliography.
    duplicates        A book whose text is mostly inside another book is skipped.

What is repaired:

    look-alikes       Latin letters inside a Cyrillic word ("тoгтooл" with Latin o's) become
                      the Cyrillic letters they imitate, and the other way round in a Latin
                      word. Roman numerals typed with Cyrillic Х or Ү ("ХIХ") get Latin
                      letters. Greek θ in a Cyrillic word is ө.
    spacing           Unicode normal form NFC; no-break and narrow spaces become spaces; soft
                      hyphens go; the modifier minus "˗" becomes a hyphen; at most one blank
                      line in a row.

Books keep their line breaks, so poems keep their lines.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote


# ----------------------------------------------------------------------------------------
# Reading books
# ----------------------------------------------------------------------------------------
BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "section",
              "article", "pre", "hr", "dd", "dt", "table", "ul", "ol", "header", "footer", "aside"}
SKIP_TAGS = {"script", "style", "head"}
NOTE_MARK = re.compile(r"\[?\(?(?:\d{1,4}|\*{1,3}|←\d{1,4})\)?\]?")    # what a footnote link shows: 1, [1], *, [←1]


class _HTMLText(HTMLParser):
    """The text of an (X)HTML page, with a line break at every block element.

    A link whose whole text is a footnote mark, as in <sup><a href="notes.html#note_1">1</a></sup>,
    is left out. Otherwise its number ends up glued to the word before it."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.skip = 0                           # depth inside <script>, <style>, <head>
        self.link: Optional[List[str]] = None   # text of an open link that may be a footnote mark

    def _close_link(self, keep: bool) -> None:
        if self.link is not None:
            if keep:
                self.parts.append("".join(self.link))
            self.link = None

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip += 1
        elif tag == "a" and "#" in (dict(attrs).get("href") or ""):
            self._close_link(keep=True)
            self.link = []
        elif tag in BLOCK_TAGS:
            self._close_link(keep=True)         # a link spanning blocks is not a footnote mark
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self._close_link(keep=True)
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self.skip = max(0, self.skip - 1)
        elif tag == "a" and self.link is not None:
            self._close_link(keep=not NOTE_MARK.fullmatch("".join(self.link).strip()))
        elif tag in BLOCK_TAGS:
            self._close_link(keep=True)
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            (self.link if self.link is not None else self.parts).append(data)


def html_to_text(raw: bytes) -> str:
    parser = _HTMLText()
    parser.feed(raw.decode("utf-8", errors="replace"))
    parser.close()
    parser._close_link(keep=True)
    return "".join(parser.parts)


def _local(tag: str) -> str:
    """'{http://www.idpf.org/2007/opf}item' -> 'item'."""
    return tag.rsplit("}", 1)[-1]


def read_epub(path: Path) -> str:
    """The text of an EPUB: its (X)HTML documents in reading order, which the spine of the
    package file (content.opf) lists."""
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        container = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(el.get("full-path") for el in container.iter() if _local(el.tag) == "rootfile")
        opf = ET.fromstring(z.read(opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {el.get("id"): (el.get("href") or "", el.get("media-type") or "")
                    for el in opf.iter() if _local(el.tag) == "item"}
        pages = []
        for el in opf.iter():
            if _local(el.tag) != "itemref":
                continue
            href, media = manifest.get(el.get("idref"), ("", ""))
            name = posixpath.normpath(posixpath.join(base, unquote(href.split("#")[0])))
            if "html" in media and name in names:
                pages.append(html_to_text(z.read(name)))
    return "\n".join(pages)


def read_book(path: Path) -> str:
    if path.suffix.lower() == ".epub":
        return read_epub(path)
    return path.read_text(encoding="utf-8-sig")


# ----------------------------------------------------------------------------------------
# Characters
# ----------------------------------------------------------------------------------------
CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
WORD = re.compile(r"[^\W\d_]+")
SPACES = re.compile(r"[ \t  -   　]+")

# Latin letters that look like Cyrillic ones. Lowercase y is у (the same shape); capital Y is
# Mongolian Ү; h is һ (Buryat, and the old h- of the Secret History transcriptions). The
# reverse table turns look-alikes in a Latin word back into Latin. m is the italic form of т,
# which scanners read as m; it only goes one way.
CYRILLIC_TO_LATIN = dict(zip("асеорхукһАВСЕНКМОРТХҮ", "aceopxykhABCEHKMOPTXY"))
LATIN_TO_CYRILLIC = {**{l: c for c, l in CYRILLIC_TO_LATIN.items()}, "m": "т"}
ROMAN = re.compile(r"[IVXХҮ]+")                                   # with Cyrillic Х and Ү mixed in
OTHER_LOOK_ALIKES = {"θ": "ө", "Θ": "Ө", "ɵ": "ө", "Ɵ": "Ө"}       # Greek theta, Latin barred o


def normalize(text: str) -> str:
    """NFC, one kind of space, no soft hyphens or zero-width spaces, a plain hyphen for "˗"."""
    text = unicodedata.normalize("NFC", text)
    for bad, good in (("­", ""), ("​", ""), ("﻿", ""), ("˗", "-"), ("‑", "-"), ("\r", "")):
        text = text.replace(bad, good)
    return "\n".join(SPACES.sub(" ", line).strip() for line in text.split("\n"))


def fix_word(word: str) -> str:
    """One word with both Cyrillic and Latin letters, repaired where the repair is certain.

    A Roman numeral gets Latin letters. Otherwise the word's script is decided by its
    letters that exist in one script only (г, и, ш... or d, s, t...), because a look-alike
    says nothing: "гap" is Cyrillic although two of its three letters are Latin. If only
    look-alikes are present, the more numerous script wins. The other script's letters are
    then replaced, provided every one of them has a look-alike; if not, the word is left."""
    if ROMAN.fullmatch(word):
        return word.replace("Х", "X").replace("Ү", "V")
    cyr = sum(1 for c in word if CYRILLIC.match(c) and c not in CYRILLIC_TO_LATIN)
    lat = sum(1 for c in word if LATIN.match(c) and c not in LATIN_TO_CYRILLIC)
    if cyr == lat:
        cyr, lat = len(CYRILLIC.findall(word)), len(LATIN.findall(word))
    if cyr == lat:
        return word
    table, other = (LATIN_TO_CYRILLIC, LATIN) if cyr > lat else (CYRILLIC_TO_LATIN, CYRILLIC)
    if all(c in table for c in word if other.match(c)):
        return "".join(table[c] if other.match(c) else c for c in word)
    return word


def fix_letters(line: str) -> Tuple[str, int]:
    """Repair look-alike letters in every mixed-script word. Returns the line and the number
    of words changed."""
    fixed = 0

    def repair(m: re.Match) -> str:
        nonlocal fixed
        word = m.group(0)
        new = word
        if CYRILLIC.search(word):
            new = "".join(OTHER_LOOK_ALIKES.get(c, c) for c in new)
        if CYRILLIC.search(new) and LATIN.search(new):
            new = fix_word(new)
        fixed += new != word
        return new

    return WORD.sub(repair, line), fixed


# ----------------------------------------------------------------------------------------
# Cleaning one book
# ----------------------------------------------------------------------------------------
NOTES_HEADING = re.compile(r"(?:Заметки|Примечания|Notes)\.?", re.IGNORECASE)   # headings EPUB converters add
LIBRARY_LINE = re.compile("|".join([                    # anywhere in a book
    r"Уг номыг .{1,3}Болор дуран.{1,3} шалгуураар шүүсэн болно\.?",
    r"Программтай танилцах бол сурталчилгаа дээр товшиж холбоос руу орно уу\.?",
    r"ЦАХИМ НОМЫН САН:.*",
    r".*цахим хэлбэрт хөрвүүлэв\.?",
    r"Цогт охин тэнгэр сангийн захиалгаар.*",
    r"Найзтайгаа хуваалцаарай:?",
    r".*(?:эндээс|номын сангаас) татаж уншиж болно\.?",
]), re.IGNORECASE)
SOURCE_NOTE = re.compile(r".*\b\d{4}\b.*(?:эрхлэн|хэвлэн гаргасан).*|[“\"«][^”\"»]*[”\"»] номоос\.?", re.IGNORECASE)
DONATION_START, DONATION_END = "ӨГЛӨГИЙН ЭЗНИЙ СОНОРТ", re.compile(r"Qpay|Tsogt okhin Tenger san", re.IGNORECASE)
PUBLICATION_DATA = re.compile(r"хэвлэн нийтлэлийн (?:мэдээлэл|гарчиг)|номын гарчиг\s*:?$",   # "publication data", "contents"
                              re.IGNORECASE)
SEPARATOR = re.compile(r"[⁂*\s]+")                       # "⁂", "* * *": scene breaks
RUSSIAN_WORDS = {"и", "в", "во", "на", "не", "что", "с", "со", "по", "к", "из", "у", "от", "для", "как", "это", "его",
                 "при", "же", "бы", "или", "но", "только", "был", "была", "были", "было", "этот", "эти", "так", "все",
                 "уже", "который", "которые", "где", "когда", "если", "чтобы"}      # none of them is a Mongolian word
COLOPHON = re.compile(r"Цаасны хэмжээ|Хэвлэлийн хуудас|Хэв\. ?хуудас|Хэвлэсэн тоо|^(?:Зах|За|Захиалсан)\.? ?№|"
                      r"^Х\. ?Г\. ?№|^ХГ ?№|^Үнэ \d|\bISBN\b|^©|\d (?:хувь|ширхэг) хэвлэв|(?:үйлдвэрт|комбинатад) хэвлэв|"
                      r"^Техн(?:ик|\.) ?ред|^Хянагч [А-ЯӨҮЁ]\.|^Өрөлтөд \d|^Хэвлэлтэд \d", re.IGNORECASE)
HEAD_LINES = 15           # the e-library's notes on the source edition sit in a book's first lines
TAIL_SHARE = 0.3          # footnotes and publication data are only cut from the last 30% of the text
COLOPHON_LINES = 40       # printer's lines are only looked for among the last 40 lines


def mostly_latin(line: str) -> bool:
    """A line of at least 8 letters, more than 70% of them Latin."""
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 8 and sum(1 for c in letters if LATIN.match(c)) > 0.7 * len(letters)


def russian(line: str) -> bool:
    """A Russian line: 30 letters or more, no Ө or Ү (5-7% of the letters of Mongolian), and
    Russian function words (и, в, на, что...) making up at least a tenth of its words."""
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 30 or any(c in "ӨөҮү" for c in letters):
        return False
    words = WORD.findall(line.lower())
    hits = sum(w in RUSSIAN_WORDS for w in words)
    return hits >= 2 and hits >= 0.1 * len(words)


def clean_book(text: str) -> Tuple[str, Counter]:
    """The book's text with the parts listed at the top of this file removed, and a count
    of the characters each rule removed (plus the number of repaired words)."""
    lines = [l for l in normalize(text).split("\n")]
    removed: Counter = Counter()

    def size(ls: List[str]) -> int:
        return sum(len(l) + 1 for l in ls)

    # Footnotes: from the last notes heading on, when notes follow it and are not most of the book.
    heads = [i for i, l in enumerate(lines) if NOTES_HEADING.fullmatch(l)]
    if heads and any(lines[heads[-1] + 1:]) and size(lines[heads[-1]:]) < TAIL_SHARE * size(lines):
        removed["footnotes"] += size(lines[heads[-1]:])
        lines = lines[:heads[-1]]

    # The donation appeal: from its heading to the last payment line after it.
    while DONATION_START in lines:
        start = lines.index(DONATION_START)
        ends = [i for i in range(start, min(len(lines), start + 40)) if DONATION_END.search(lines[i])]
        end = ends[-1] if ends else start
        removed["library"] += size(lines[start:end + 1])
        lines = lines[:start] + lines[end + 1:]

    # Publication data: the e-library copies the printed book's contents page and colophon
    # after the text, under headings that say so. Cut from the first one, if it is near the end.
    total = size(lines)
    for i, l in enumerate(lines):
        if PUBLICATION_DATA.search(l) and size(lines[i:]) < TAIL_SHARE * total:
            removed["colophon"] += size(lines[i:])
            lines = lines[:i]
            break

    # Printer's lines among the last lines (books without that heading).
    tail = max(0, len(lines) - COLOPHON_LINES)
    kept = []
    for i, l in enumerate(lines):
        if i >= tail and len(l) < 120 and COLOPHON.search(l):
            removed["colophon"] += len(l) + 1
        else:
            kept.append(l)
    lines = kept

    # Line by line: library text, source-edition notes at the start, foreign lines, letters.
    out, seen_text = [], 0
    for l in lines:
        if not l:
            out.append(l)
            continue
        seen_text += 1
        if LIBRARY_LINE.fullmatch(l):
            removed["library"] += len(l) + 1
        elif seen_text <= HEAD_LINES and SOURCE_NOTE.fullmatch(l):
            removed["library"] += len(l) + 1
        elif mostly_latin(l) or russian(l):
            removed["foreign"] += len(l) + 1
        else:
            l, n = fix_letters(l.replace(" (e-nom.mn)", ""))
            removed["repaired words"] += n
            out.append(l)

    # Scene-break marks left at the very start or end, or doubled where library text was cut.
    kept, last = [], None                                    # last: the previous line with text
    for l in out:
        if l and SEPARATOR.fullmatch(l) and (last is None or SEPARATOR.fullmatch(last)):
            continue
        kept.append(l)
        last = l or last
    while kept and (not kept[-1] or SEPARATOR.fullmatch(kept[-1])):
        kept.pop()
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip() + "\n"
    return text, removed


# ----------------------------------------------------------------------------------------
# The whole collection
# ----------------------------------------------------------------------------------------
def shingles(text: str, k: int = 8) -> set:
    """Hashes of every run of k words: two texts share many only if they share passages."""
    words = WORD.findall(text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(len(words) - k + 1)}


def health(text: str) -> Dict[str, float]:
    """Shares of letters by script, and characters that point to a broken conversion."""
    letters = [c for c in text if c.isalpha()]
    n = max(1, len(letters))
    return {"cyrillic": sum(1 for c in letters if CYRILLIC.match(c)) / n,
            "latin": sum(1 for c in letters if LATIN.match(c)) / n,
            "өү": sum(1 for c in letters if c in "ӨөҮү") / n,
            "broken": sum(1 for c in text if c == "�" or "" <= c <= "")}


def flags(h: Dict[str, float], n_bytes: int) -> List[str]:
    """Warnings for the report. Mongolian Cyrillic has Ө and Ү in 5-7% of its letters;
    Russian has none; legacy Mongolian fonts show up as private-use characters."""
    out = []
    if h["broken"]:
        out.append(f"{h['broken']} broken characters (legacy font or bad encoding?)")
    if h["өү"] < 0.03:
        out.append(f"only {h['өү']:.1%} Ө/Ү: not Mongolian Cyrillic?")
    if h["latin"] > 0.05:
        out.append(f"{h['latin']:.0%} Latin letters")
    if n_bytes < 5000:
        out.append("very short")
    return out


def main(argv=None) -> Dict[str, dict]:
    ap = argparse.ArgumentParser(description="Turn .epub and .txt books into clean training text.")
    ap.add_argument("--src", type=Path, default=Path("data/novels"), help="folder of .epub / .txt books")
    ap.add_argument("--out", type=Path, default=Path("data/clean"), help="folder for the clean .txt files")
    ap.add_argument("--duplicate", type=float, default=0.5,
                    help="skip a book when this share of its 8-word passages is found in other books")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    if not args.src.is_dir():
        sys.exit(f"no folder {args.src}: put the books there or pass --src")
    files = sorted(p for p in args.src.iterdir() if p.is_file())
    books: Dict[str, dict] = {}
    for path in files:
        if path.suffix.lower() not in (".epub", ".txt"):
            print(f"skipped {path.name}: only .epub and .txt are read (convert other formats first)")
            continue
        try:
            raw = read_book(path)
        except Exception as e:                     # a damaged file should not stop the others
            print(f"skipped {path.name}: could not read it ({e})")
            continue
        text, removed = clean_book(raw)
        books[path.stem] = {"raw": len(raw.encode("utf-8")), "text": text, "removed": removed}

    # Duplicates: going from the smallest book up, skip a book whose passages are mostly
    # found in the books still kept. `owners` counts the kept books that contain a passage.
    sh = {name: shingles(b["text"]) for name, b in books.items()}
    owners = Counter()
    for s in sh.values():
        owners.update(s)
    for name in sorted(books, key=lambda n: len(books[n]["text"])):
        shared = sum(1 for s in sh[name] if owners[s] > 1) / max(1, len(sh[name]))
        if shared >= args.duplicate:
            kept_books = [o for o in books if o != name and not books[o].get("duplicate_of")]
            books[name]["duplicate_of"] = max(kept_books, key=lambda o: len(sh[name] & sh[o]))
            books[name]["shared"] = shared
            owners.subtract(sh[name])

    # Write, and report.
    args.out.mkdir(parents=True, exist_ok=True)
    total_raw = total_clean = words = 0
    print(f"\n{'book':<44} {'read':>7} {'clean':>7} {'words':>8} {'Ө/Ү':>5} {'Latin':>6}   removed / notes")
    for name, b in books.items():
        target = args.out / f"{name}.txt"
        if b.get("duplicate_of"):
            if target.exists():
                target.unlink()
            print(f"{name[:44]:<44} {b['raw'] / 1e6:6.2f}M  skipped: {b['shared']:.0%} of it is in {b['duplicate_of']}")
            continue
        target.write_text(b["text"], encoding="utf-8")
        n_bytes = len(b["text"].encode("utf-8"))
        h = health(b["text"])
        b.update(clean=n_bytes, words=len(b["text"].split()), health=h)
        total_raw, total_clean, words = total_raw + b["raw"], total_clean + n_bytes, words + b["words"]
        notes = [f"{k} {v / 1e3:.1f}k" if k != "repaired words" else f"{v} repaired words"
                 for k, v in sorted(b["removed"].items()) if v]
        print(f"{name[:44]:<44} {b['raw'] / 1e6:6.2f}M {n_bytes / 1e6:6.2f}M {b['words']:>8,} {h['өү']:5.1%} {h['latin']:6.2%}   "
              + ", ".join(notes + [f"CHECK: {f}" for f in flags(h, n_bytes)]))
    kept = sum(1 for b in books.values() if not b.get("duplicate_of"))
    print(f"\n{kept} books -> {args.out}: {total_clean / 1e6:.2f} MB of text ({total_raw / 1e6:.2f} MB read), {words:,} words")
    stale = [p.name for p in sorted(args.out.glob("*.txt")) if p.stem not in books]
    if stale:
        print(f"note: {len(stale)} .txt file(s) in {args.out} are not from {args.src} and will be trained on too: "
              + ", ".join(stale))
    print(f"next: python train.py --data {args.out.as_posix()}")
    return books


if __name__ == "__main__":
    main()
