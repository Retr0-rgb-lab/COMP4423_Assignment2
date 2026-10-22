"""Final acceptance check for the two submission artefacts.

Run from the repository root:  python docs/acceptance_check.py

It checks the things that can be checked mechanically:
  - the PDF compiled without errors, has every figure, and carries the
    name, the student number and a working dataset link;
  - the report source has no placeholder left, and every internal cross
    reference resolves;
  - the code that ships has no Chinese comments left, the saved pipeline
    loads, and the application reproduces the locked test predictions;
  - the code archive contains everything the assignment asks for;
  - the two file names are exactly the ones the assignment demands.
"""
import csv
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = os.environ.get("VENV_PYTHON",
                     "/mnt/d/Program Files/learn_torch/venv/Scripts/python.exe")
PDF = REPO / "Assignment2_Report_24101322D_HUANG_Haoran.pdf"
ZIP = REPO / "Assignment2_Code_24101322D_HUANG_Haoran.zip"
MD = REPO / "progress/task6_report_draft.md"

fails = []
warns = []


def ok(label, cond, detail=""):
    print(f"  {'OK  ' if cond else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(label)


def warn(label, detail=""):
    print(f"  WARN  {label}" + (f" — {detail}" if detail else ""))
    warns.append(label)


print("=== 1. 文件命名 ===")
ok("报告文件名 = Assignment2_Report_24101322D_HUANG_Haoran.pdf", PDF.is_file())
ok("代码包名 = Assignment2_Code_24101322D_HUANG_Haoran.zip", ZIP.is_file())

print("\n=== 2. 报告源文件 ===")
md = MD.read_text(encoding="utf-8")
for pat, label in [(r"<姓名>|<学号>", "没有姓名/学号占位符"),
                   (r"PASTE_A_REAL|PASTE_THE_SAME", "没有链接占位符"),
                   (r"DRAFTING NOTES", "没有 DRAFTING NOTES 残留"),
                   (r"\bTODO\b|\bFIXME\b", "没有 TODO/FIXME")]:
    ok(label, not re.search(pat, md))
ok("姓名已填 Huang Haoran", "**Name:** Huang Haoran" in md)
ok("学号已填 24101322D", "**Student number:** 24101322D" in md)
link_m = re.search(r"(https://drive\.google\.com/drive/folders/[A-Za-z0-9_-]+)", md)
ok("报告里有 Drive 链接", link_m is not None)
if link_m:
    link = link_m.group(1)
    dl = (REPO / "DATASET_LINK.txt").read_text(encoding="utf-8")
    ok("链接与 DATASET_LINK.txt 一致", link in dl, link)

print("\n=== 3. 交叉引用与逐折声明 ===")
for script in ("check_crossrefs.py", "check_fold_claims.py"):
    r = subprocess.run([sys.executable, str(REPO / "docs" / script)],
                       capture_output=True, text=True, cwd=REPO)
    ok(f"docs/{script} 全部通过", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout else "")

print("\n=== 4. 代码注释语言 ===")
r = subprocess.run([sys.executable, "-c", """
import ast, glob, io, sys, tokenize
CJK = lambda s: any('一' <= c <= '鿿' for c in s)
bad = 0
for f in glob.glob('src/**/*.py', recursive=True):
    src = open(f, encoding='utf-8').read()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT and CJK(tok.string):
            print('comment', f, tok.start[0]); bad += 1
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            b = getattr(n, 'body', [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \\
               and isinstance(b[0].value.value, str) and CJK(b[0].value.value):
                print('docstring', f); bad += 1
print('BAD=%d' % bad)
"""], capture_output=True, text=True, cwd=REPO)
m = re.search(r"BAD=(\d+)", r.stdout)
ok("src/ 下中文注释与 docstring 为 0", m and m.group(1) == "0", r.stdout.strip()[-120:] if m else r.stderr[-200:])

print("\n=== 5. 模型与应用 ===")
r = subprocess.run([VENV, "-c", """
import csv, sys
sys.path.insert(0, '.')
import joblib
from src.task5.predict import load_pipeline, predict_rgb
from src.task3.preprocess import load_rgb
from src.task3.config import PICTURES
pipe = load_pipeline()
rows = list(csv.DictReader(open('outputs/task3/test_predictions.csv', encoding='utf-8-sig')))
idx = {p.name: p for p in PICTURES.rglob('*') if p.is_file()}
agree = sum(1 for r in rows
             if predict_rgb(pipe, load_rgb(str(idx[r['filename']]))).label_id == int(r['y_pred']))
print('DISPLAY=%s' % pipe.display_)
print('AGREE=%d/%d' % (agree, len(rows)))
"""], capture_output=True, text=True, cwd=REPO)
disp = re.search(r"DISPLAY=\{(.*?)\}", r.stdout)
agree = re.search(r"AGREE=(\d+)/(\d+)", r.stdout)
ok("保存的流水线能加载", bool(disp), disp.group(0) if disp else r.stderr[-200:])
ok("DISPLAY 是英文标签", bool(disp) and "Inside PolyU main campus" in disp.group(1))
ok("应用对全部 69 张测试图与锁定预测表一致",
   bool(agree) and agree.group(1) == agree.group(2) == "69",
   agree.group(0) if agree else r.stderr[-200:])

print("\n=== 6. PDF ===")
info = subprocess.run(["mutool", "info", str(PDF)], capture_output=True, text=True).stdout
pages = re.search(r"Pages:\s*(\d+)", info)
ok("PDF 页数已读出", bool(pages), pages.group(0) if pages else "")
import binascii
def _pdf_str(raw):
    """Decode a PDF text string: a hex string, or a literal string whose
    octal escapes hold UTF-16BE."""
    m = re.search(rb"/Title(?:<([0-9A-Fa-f]+)>|\(([^)]*)\))", raw)
    if not m:
        return ""
    if m.group(1):
        return binascii.unhexlify(m.group(1)).decode("utf-16-be", "replace").lstrip("\ufeff")
    body, out, i = m.group(2), bytearray(), 0
    while i < len(body):
        c = body[i:i + 1]
        if c == b"\\" and i + 1 < len(body):
            if body[i + 1:i + 4].isdigit():
                out.append(int(body[i + 1:i + 4], 8) & 0xFF)
                i += 4
                continue
            out += body[i + 1:i + 2]
            i += 2
            continue
        out += c
        i += 1
    return bytes(out).decode("utf-16-be", "replace").lstrip("\ufeff")


ok("PDF 有标题元数据", "COMP4423" in _pdf_str(PDF.read_bytes()),
   _pdf_str(PDF.read_bytes())[:60])
text = subprocess.run(["mutool", "draw", "-F", "txt", "-o", "-", str(PDF)],
                      capture_output=True, text=True).stdout
for n in range(1, 7):
    ok(f"PDF 里有 Figure {n}. 题注", f"Figure {n}." in text)
ok("PDF 文本层含姓名", "Huang Haoran" in text)
ok("PDF 文本层含学号", "24101322D" in text)
raw = PDF.read_bytes()
import zlib
blobs = [raw]
for mm in re.finditer(rb"stream\r?\n", raw):
    s = mm.end(); e = raw.find(b"endstream", s)
    if e > 0:
        try: blobs.append(zlib.decompress(raw[s:e]))
        except Exception: pass
allb = b"\n".join(blobs)
uris = re.findall(rb"/URI\s*\(([^)]{5,140})\)", allb)
ok("PDF 里有可点击的网盘链接", any(b"drive.google.com" in u for u in uris),
   f"{len(uris)} 个 URI 注解")
# the CJK glyphs live in a Type1 subset with a custom encoding, so the text
# layer does not map them back; check that a CJK font is embedded instead
ok("PDF 里嵌入了中文字型", any(b"/BaseFont" in b and b"gbsn" in b.lower()
   for b in blobs) or b"gbsn" in allb.lower())

print("\n=== 6b. 报告正文无乱码 ===")
# a word glued to a digit is the signature of a bad text substitution
GLUE = re.compile(r"\b(place|section|figure|table|part|line|step|item|case|kind|sort|"
                  r"mode|note|text|word|page|form|group|frame|row|column)\d")
# strip inline code spans and link targets, so that file names such as
# figure1_inside_locations_annotated.png are not read as garbled text
prose = re.sub(r"`[^`]*`", " ", md)
prose = re.sub(r"\]\([^)]*\)", "] ", prose)
glue = sorted({m.group(0) for m in GLUE.finditer(prose)})
ok("正文里没有粘连的词+数字", not glue, ", ".join(glue))

print("\n=== 7. 代码包内容 ===")
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
required = ["models/task3_pipeline.joblib", "data/annotations/dataset_summary.md",
            "data/annotations/splits.csv", "DATASET_LINK.txt", "requirements.txt",
            "README.md", "progress/task6_report_draft.md", "progress/task3_experiment.md",
            "progress/task4_evaluation.md", "progress/task1_dataset_overview.md",
            "plans/task1_3_review_2026-10-15.md", "docs/figures/task4_example_grid.png",
            "docs/figures/figure1_inside_locations_annotated.png",
            "docs/figures/task5_application.png",
            "src/task3/features.py", "src/task5/app.py", "src/task5/predict.py",
            "src/tools/build_dataset.py", "docs/make_report.py",
            # the report promises these ship with the archive
            "outputs/task3/stage_a.csv", "outputs/task3/stage_b.csv",
            "outputs/task3/selection.json", "outputs/task3/val_metrics.json",
            "outputs/task3/test_metrics.json", "outputs/task3/test_predictions.csv",
            "outputs/task3/RESULTS_PROVENANCE.txt"]
for rq in required:
    ok(f"包内含 {rq}", rq in names)
ok("包内没有 pycache", not any("__pycache__" in n for n in names))
ok("包内无 .pyc", not any(n.endswith(".pyc") for n in names))

print("\n=== 8. 版本库同步 ===")
r = subprocess.run(["git", "rev-list", "--left-right", "--count", "origin/main...HEAD"],
                   capture_output=True, text=True, cwd=REPO).stdout.strip()
ok("本地与 origin/main 同步", r == "0\t0", r)

print("\n" + "=" * 58)
if fails:
    print(f"不通过 {len(fails)} 项:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
print(f"全部通过（{len(warns)} 项提醒）")
