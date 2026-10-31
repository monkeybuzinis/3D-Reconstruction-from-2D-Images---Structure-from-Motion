"""
Figure and table bookkeeping for the report .docx.

Adding a figure by hand means renumbering every later caption and hunting
down every "Figure N" in the prose. That is mechanical and easy to get
wrong, so it lives here instead.

    python3 tools/report_figures.py list    documents/Khanh-Report.docx
    python3 tools/report_figures.py check   documents/Khanh-Report.docx
    python3 tools/report_figures.py renumber documents/Khanh-Report.docx

`list` prints every caption in document order with the images near it.
`check` reports duplicates, gaps and references that point nowhere.
`renumber` makes the captions 1..N in document order and rewrites every
cross-reference to match. Nothing else in the file is touched: the zip is
rebuilt entry by entry, so images, styles and relationships are copied
across byte for byte.

Citations such as "Hartley & Zisserman (2004), Figure 6.1" and
"Elkhrachy (2022), Figure 3b" are left alone -- a number followed by a
dot-digit or a letter is somebody else's figure, not ours.
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

DOC = "word/document.xml"
PARA = re.compile(r"<w:p\b.*?</w:p>", re.S)
TEXT = re.compile(r"<w:t(?: [^>]*)?>(.*?)</w:t>", re.S)
# "Figure 3." / "Table 10." at the start of a paragraph: a caption.
# "Figure #." is a placeholder: renumber assigns it the number its position earns,
# which is how a new figure gets added without touching any of the others.
CAPTION = re.compile(r"^(Figure|Table)\s+(\d+|#)\.\s")
# any "Figure 3" / "Table 10" in running text; the lookahead protects
# "Figure 6.1" (a sub-figure) and "Figure 3b" (a panel) in citations.
REF = re.compile(r"\b(Figure|Table)\s+(\d+)(?!\s*\.\s*\d)(?![0-9A-Za-z])")


def read_xml(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read(DOC).decode("utf-8")


def write_xml(src: Path, dst: Path, xml: str) -> None:
    """Copy the archive, swapping only word/document.xml."""
    with zipfile.ZipFile(src) as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in items:
            zout.writestr(info, xml.encode("utf-8") if info.filename == DOC else data)


def paragraphs(xml: str):
    for m in PARA.finditer(xml):
        yield m, "".join(TEXT.findall(m.group(0)))


def survey(xml: str):
    """Captions and images, in document order."""
    caps, imgs = [], []
    for m, text in paragraphs(xml):
        hit = CAPTION.match(text.strip())
        if hit:
            raw = hit.group(2)
            caps.append(dict(kind=hit.group(1), num=None if raw == "#" else int(raw),
                             start=m.start(), text=text.strip()))
        if "<w:drawing>" in m.group(0):
            imgs.append(m.start())
    return caps, imgs


def cmd_list(path: Path) -> int:
    xml = read_xml(path)
    caps, imgs = survey(xml)
    for kind in ("Figure", "Table"):
        rows = [c for c in caps if c["kind"] == kind]
        print(f"\n{kind}s ({len(rows)}):")
        for i, c in enumerate(rows, 1):
            shown = "#" if c["num"] is None else c["num"]
            flag = "" if c["num"] == i else f"   <- becomes {i}"
            print(f"  {c['kind']} {str(shown):<3} {c['text'][:86]}{flag}")
    print(f"\n{len(imgs)} embedded images in total.")
    return 0


def cmd_check(path: Path) -> int:
    xml = read_xml(path)
    caps, _ = survey(xml)
    bad = 0
    for kind in ("Figure", "Table"):
        rows = [c for c in caps if c["kind"] == kind]
        nums = [c["num"] for c in rows if c["num"] is not None]
        dupes = {n for n in nums if nums.count(n) > 1}
        if dupes:
            print(f"{kind}: numbers used more than once: {sorted(dupes)}")
            bad += 1
        out_of_order = [(c["num"], i) for i, c in enumerate(rows, 1)
                        if c["num"] is not None and c["num"] != i]
        if out_of_order:
            print(f"{kind}: not sequential in document order: "
                  + ", ".join(f"{n}->{i}" for n, i in out_of_order))
            bad += 1
        defined = set(nums)
        refs = set()
        for _m, text in paragraphs(xml):
            if CAPTION.match(text.strip()):
                continue
            refs |= {int(n) for k, n in REF.findall(text) if k == kind}
        dangling = sorted(refs - defined)
        if dangling:
            print(f"{kind}: referenced but never captioned: {dangling}")
            bad += 1
    print("clean" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def cmd_renumber(path: Path, out: Path | None) -> int:
    xml = read_xml(path)
    caps, _ = survey(xml)
    mapping = {}
    for kind in ("Figure", "Table"):
        rows = [c for c in caps if c["kind"] == kind]
        nums = [c["num"] for c in rows if c["num"] is not None]
        if len(set(nums)) != len(nums):
            print(f"{kind}: duplicate numbers {sorted(n for n in set(nums) if nums.count(n) > 1)}; "
                  "fix those by hand first -- references cannot be mapped unambiguously.")
            return 2
        mapping[kind] = {c["num"]: i for i, c in enumerate(rows, 1) if c["num"] is not None}
        placeholders = [i for i, c in enumerate(rows, 1) if c["num"] is None]

    moved = {k: {a: b for a, b in v.items() if a != b} for k, v in mapping.items()}
    new_caps = [c for c in caps if c["num"] is None]
    if not any(moved.values()) and not new_caps:
        print("already sequential; nothing to do")
        return 0

    def fix(body: str) -> str:
        return REF.sub(lambda m: f"{m.group(1)} {mapping[m.group(1)].get(int(m.group(2)), int(m.group(2)))}", body)

    xml = re.sub(r"(<w:t(?: [^>]*)?>)(.*?)(</w:t>)",
                 lambda m: m.group(1) + fix(m.group(2)) + m.group(3), xml, flags=re.S)

    # placeholders are filled last: "#" never matches REF, so the pass above
    # could not have renumbered them by accident.
    for kind in ("Figure", "Table"):
        rows = [c for c in survey(xml)[0] if c["kind"] == kind]
        for i, c in enumerate(rows, 1):
            if c["num"] is None:
                xml = xml.replace(f"{kind} #.", f"{kind} {i}.", 1)
                print(f"{kind} #  -> {kind} {i}")

    dst = out or path
    if dst == path:
        shutil.copy2(path, path.with_suffix(".docx.bak"))
        print(f"backup -> {path.with_suffix('.docx.bak')}")
    write_xml(path, dst, xml)
    for kind, mv in moved.items():
        if mv:
            print(f"{kind}: " + ", ".join(f"{a}->{b}" for a, b in sorted(mv.items())))
    print(f"wrote {dst}")
    return 0


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    cmd, path = sys.argv[1], Path(sys.argv[2])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "-o" else None
    if not path.is_file():
        print(f"no such file: {path}")
        return 1
    return {"list": lambda: cmd_list(path),
            "check": lambda: cmd_check(path),
            "renumber": lambda: cmd_renumber(path, out)}.get(cmd, lambda: (print(__doc__), 1)[1])()


if __name__ == "__main__":
    raise SystemExit(main())
