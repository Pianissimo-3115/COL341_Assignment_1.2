"""
Dev tool (NOT part of the submission).

Builds the submission archive exactly as the assignment specifies:

    <entry1>_<entry2>.zip
        part_a.py
        part_b.py
        part_c.py
        report.pdf

    python make_submission.py <entry1> <entry2> [out_dir]

Refuses to build if anything is missing or if a stray file would be included,
because the assignment charges 10% of the part's marks per intervention needed
at evaluation time - a missing report or a renamed file is exactly that.
"""
import io
import sys
import tokenize
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["part_a.py", "part_b.py", "part_c.py", "report.pdf"]
# Shipped with '#' comments stripped. Done here rather than in the source so the
# documented version stays in git; tokenize is used rather than a regex so a '#'
# inside a string literal is never mistaken for a comment.
STRIP_COMMENTS = ["part_a.py", "part_b.py"]


def strip_comments(src):
    """Remove '#' comments, leaving code and string literals untouched.

    Works by cutting the source lines at the comment positions tokenize
    reports, rather than re-emitting tokens, so nothing else can shift. A '#'
    inside a string is never a COMMENT token, so it survives.
    """
    cut_at = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            cut_at.setdefault(row, col)         # first comment on the line

    kept = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if lineno in cut_at:
            head = line[:cut_at[lineno]].rstrip()
            if not head:
                continue                        # whole line was a comment
            kept.append(head)
        else:
            kept.append(line.rstrip())

    out, blanks = [], 0
    for line in kept:                           # collapse runs of blank lines
        if line.strip():
            blanks = 0
            out.append(line)
        else:
            blanks += 1
            if blanks <= 2:
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        sys.exit(1)
    entry1, entry2 = sys.argv[1], sys.argv[2]
    out_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED if not (ROOT / f).exists()]
    if missing:
        print(f"ERROR: missing {', '.join(missing)}", file=sys.stderr)
        if "report.pdf" in missing:
            print("  compile it:  pdflatex report.tex", file=sys.stderr)
        sys.exit(1)

    archive = out_dir / f"{entry1}_{entry2}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in REQUIRED:
            if name in STRIP_COMMENTS:
                src = (ROOT / name).read_text(encoding="utf-8")
                stripped = strip_comments(src)
                compile(stripped, name, "exec")     # refuse to ship broken code
                z.writestr(name, stripped)
                print(f"  {name}: stripped comments "
                      f"({len(src.splitlines())} -> "
                      f"{len(stripped.splitlines())} lines)")
            else:
                z.write(ROOT / name, arcname=name)  # arcname => top level, flat

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    print(f"wrote {archive}")
    for n in names:
        print(f"  {n}")
    if sorted(names) != sorted(REQUIRED):
        print("ERROR: archive contents do not match the required four files",
              file=sys.stderr)
        sys.exit(1)
    print("\nContents match the assignment's required layout.")


if __name__ == "__main__":
    main()
