"""Check that every internal cross reference in the report resolves.

The report quotes its own sections as "Section 4.2" and "§5.3". A reference
to a section that does not exist is the kind of thing a marker notices
immediately, so this is checked mechanically rather than by eye.

Run from the repository root:  python docs/check_crossrefs.py
"""
import re
import sys
from pathlib import Path

REPO = Path("/mnt/d/Program Files/learn_torch/python-cv/Assignment2")
DOC = REPO / "progress/task6_report_draft.md"

text = DOC.read_text(encoding="utf-8")
lines = text.splitlines()

# every heading that exists
headings = {}
for i, l in enumerate(lines, 1):
    m = re.match(r"^(#{1,6})\s+(.*)$", l)
    if m:
        title = m.group(2).strip()
        key = re.sub(r"^\d+(\.\d+)*\.?\s+", "", title)
        headings.setdefault(key, []).append(i)
        headings.setdefault(title, []).append(i)

# numbered headings: "3.5 Location holdout" -> "3.5"
for i, l in enumerate(lines, 1):
    m = re.match(r"^#{1,6}\s+(\d+(?:\.\d+)*)\.?\s+", l)
    if m:
        headings.setdefault(m.group(1), []).append(i)

refs = []
for i, l in enumerate(lines, 1):
    for m in re.finditer(r"[Ss]ection (\d+(?:\.\d+)*)|§(\d+(?:\.\d+)*)", l):
        num = m.group(1) or m.group(2)
        refs.append((i, num, l))

fails = []
print(f"找到 {len(refs)} 处内部引用\n")
for i, num, ctx in refs:
    if num in headings:
        print(f"  OK    L{i:<4} Section {num:<5} -> 行 {headings[num][0]}")
    elif "experiment note" in ctx:
        # a reference to a section of progress/task3_experiment.md, not of
        # this report; the wording names the other document
        print(f"  OK    L{i:<4} Section {num:<5} -> 外部文档 progress/task3_experiment.md")
    else:
        print(f"  FAIL  L{i:<4} Section {num:<5} 不存在   | {ctx.strip()[:90]}")
        fails.append((i, num))

print("\n=== 编号小节一览 ===")
numbered = sorted(k for k in headings if re.fullmatch(r"\d+(\.\d+)*", k))
for k in numbered:
    print(f"  {k:<6} 行 {headings[k][0]}")

print(f"\n=== 汇总 ===")
if fails:
    print(f"  {len(fails)} 处引用落空: {fails}")
    raise SystemExit(1)
print("  全部引用都能落到真实小节")
