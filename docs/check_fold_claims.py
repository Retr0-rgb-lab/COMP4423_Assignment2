"""Check every factual claim in the report that touches the per-fold table.

Run from the repository root. It parses stage_b.csv and confirms each
statement the report makes about the five F4 classifiers, so that a claim
cannot drift away from the numbers again.
"""
import csv
import re
from pathlib import Path

REPO = Path("/mnt/d/Program Files/learn_torch/python-cv/Assignment2")
rows = {(r["feature_id"], r["clf_id"]): r
        for r in csv.DictReader(open(REPO / "outputs/task3/stage_b.csv", encoding="utf-8-sig"))}
ba = {c: [float(rows[("F4", c)][f"ba_{i}"]) for i in range(5)]
      for c in ("C1", "C2", "C3", "C4", "C5")}
doc = (REPO / "progress/task6_report_draft.md").read_text(encoding="utf-8")

fails = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        fails.append(label)


print("=== 逐折最优 ===")
for i in range(5):
    col = {c: ba[c][i] for c in ba}
    order = sorted(col.items(), key=lambda x: -x[1])
    print(f"  fold {i}: " + ", ".join(f"{c}:{v:.3f}" for c, v in order))

print("\n=== 报告里的断言 ===")
for c in ("C1", "C2", "C3", "C4", "C5"):
    r = rows[("F4", c)]
    # the per-fold table row is the one whose second cell is a number; the
    # classifier table in Section 3.3 has the same leading id but a name next
    line = None
    for l in doc.splitlines():
        if not (l.startswith(f"| {c} ") or l.startswith(f"| **{c} ")):
            continue
        cells = [x.strip().replace("*", "") for x in l.strip("|").split("|")]
        if len(cells) == 7:
            try:
                float(cells[1])
            except ValueError:
                continue
            line = cells
            break
    if line is None:
        check(f"{c} 行在逐折表里", False)
        continue
    got = [float(x) for x in line[1:7]]
    want = [round(v, 3) for v in ba[c]] + [round(float(r["ba_mean"]), 3)]
    check(f"{c} 逐折表与 stage_b.csv 一致", got == want, f"doc={got} csv={want}")

mins = {c: min(ba[c]) for c in ba}
check("C4 最低折 = 0.616", round(mins["C4"], 3) == 0.616, f"实际 {mins['C4']:.4f}")
check("C4 的最低值是五者最高", mins["C4"] == max(mins.values()),
      f"各者最低 {({c: round(v, 3) for c, v in mins.items()})}")
for c, v in (("C3", 0.510), ("C5", 0.561), ("C1", 0.534), ("C2", 0.534)):
    check(f"{c} 最低折 = {v}", round(mins[c], 3) == v, f"实际 {mins[c]:.4f}")

check("fold 0 最优是 C5 (0.870)",
      max(ba, key=lambda c: ba[c][0]) == "C5" and round(ba["C5"][0], 3) == 0.870)
check("全表最佳单折 = 0.870", round(max(max(v) for v in ba.values()), 3) == 0.870)
check("fold 1 最优是 C4 (0.790)", max(ba, key=lambda c: ba[c][1]) == "C4")
check("fold 2 最优是 C4 (0.616)", max(ba, key=lambda c: ba[c][2]) == "C4")
check("fold 3 最优是 C3 (0.783)", max(ba, key=lambda c: ba[c][3]) == "C3")
check("fold 4 最优是 C3 (0.861)", max(ba, key=lambda c: ba[c][4]) == "C3")
check("fold 4 第二名是 C4 (0.841)",
      sorted(ba, key=lambda c: -ba[c][4])[1] == "C4" and round(ba["C4"][4], 3) == 0.841)
check("fold 0 上 C4 与 C1/C2/C3 并列 0.840",
      all(round(ba[c][0], 3) == 0.840 for c in ("C1", "C2", "C3", "C4")))
check("fold 3 上 C4 排第四 (0.700)",
      round(ba["C4"][3], 3) == 0.700
      and sorted(ba, key=lambda c: -ba[c][3]).index("C4") == 3)
check("C3 在 fold 2 最低", ba["C3"][2] == min(ba["C3"]))

print("\n=== 折定义与 config.py 一致 ===")
cfg = (REPO / "src/task3/config.py").read_text(encoding="utf-8")
block = re.search(r"FOLDS = \((.*?)\n\)", cfg, re.S).group(1)
folds = re.findall(r'"inside": \(([^)]*)\), "outside": \(([^)]*)\)', block)
ids = [(re.findall(r"[A-Z]+-\d+", ins), re.findall(r"[A-Z]+-\d+", out))
       for ins, out in folds]
expect = [(["IN-01"], ["OUT-01"]), (["IN-02"], ["OUT-02"]), (["IN-04"], ["OUT-03"]),
          (["IN-06"], ["OUT-04"]), (["IN-01"], ["OUT-05", "OUT-08"])]
for i, (real, exp) in enumerate(zip(ids, expect)):
    check(f"fold {i} 折定义与 config.py 一致", real == exp, f"config: {real}")

sec42 = [l for l in doc.splitlines() if "They are the folds of Section 3.5" in l]
if sec42:
    phrases = ["IN-01 and OUT-01", "IN-02 and OUT-02", "IN-04 and OUT-03",
               "IN-06 and OUT-04", "IN-01 together with OUT-05 and OUT-08"]
    for phrase in phrases:
        check(f"§4.2 写了「{phrase}」", phrase in sec42[0])

print("\n=== 汇总 ===")
if fails:
    print(f"  {len(fails)} 项不通过:")
    for f in fails:
        print("   -", f)
    raise SystemExit(1)
print("  全部通过")
