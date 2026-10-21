# COMP4423 Assignment 2

PolyU 主校区场景照片二分类：inside = 1，outside = 0。截止 2026-10-28。提交报告 PDF 和代码 ZIP。

## 规则

1. 每个 task 的思路、过程和结果写在 `progress/`，文件名像 `task3_experiment.md`。报告只从这些记录取数。实验跑完把结果写回同一份文件。`outputs/` 和 `models/` 被 git 忽略，不能当作唯一记录。
2. 审核报告、缺陷清单和整改顺序写在 `plans/`，文件名带审核日期，像 `task1_3_review_2026-10-15.md`。后续整改在同一份文件追加新日期小节，不覆盖旧结论。
3. 一个物理场所是一个 `group_id`，也是 train / val / test 的划分单元。不改 `data/annotations/splits.csv`，除非用户明确要求。图片按文件名在 `data/pictures/` 里解析。
4. 分类器只用像素。文件名、文件夹名、EXIF、GPS 只用来核对标签和找路径。Task 3 不用神经网络，也不用 CNN 特征。
5. 不按测试集改模型或重选特征。测试集只评估已经锁定的一条流水线。新的一轮实验追加到对应 `progress/` 记录的新日期小节，不覆盖已有结果。
6. Task 5 的本地程序必须调用 Task 3 保存的 `Pipeline` 和 `src/task3/preprocess.py` 里的 `resize_and_crop`，不能另写预处理，也不能从路径推断类别。
7. 依赖钉进 `requirements.txt`。`models/task3_pipeline.joblib`、`data/annotations/dataset_summary.md` 和数据集链接 `DATASET_LINK.txt` 必须随 ZIP 提交，用 `git add -f`。文件按 `Assignment2_Report_<学号>_<姓名>.pdf` 和 `Assignment2_Code_<学号>_<姓名>.zip` 命名，命名错误记零分。
8. **代码注释一律用英文。** 仓库里所有 Python 文件的 `#` 注释和 docstring 全部写成英文，评卷人按国际读者看待这份代码。中文只允许出现在三处：数据集与模型的**数据字符串**（场所名、场景类别词表、group_id、特征与分类器显示名、`config.py` 的 `DISPLAY` 标签）、**写进产物文件的内容**（`build_dataset.py` 生成的 `dataset_summary.md`、`make_task4_figures.py` 画进 PNG 的中文标注）、以及 `progress/`、`plans/` 里的中文文档。命令行提示与报错信息（`print`、`SystemExit`、`argparse`）也用英文。不要改任何标识符（类名、函数名、变量名）——`models/task3_pipeline.joblib` 按模块路径引用 `src/task3/` 的自定义类，改名会让模型加载失败。

## 记录

| 范围 | 文件 |
|---|---|
| Task 1–2 类别、采集、标注、划分 | `progress/task1_dataset_overview.md` |
| Task 3 特征、分类器、实测 | `progress/task3_experiment.md`（第 8 节是 2026-10-14 的结果） |
| Task 4 测试集评估与错例 | `progress/task4_evaluation.md` |
| Task 5 本地选图程序 | `progress/task5_app.md`（代码在 `src/task5/`） |
| Task 6 报告草稿 | `progress/task6_report_draft.md` |
| 报告排版与出图 | `docs/make_report.py`、`docs/report_template.tex` |
| Task 1–3 审核与整改 | `plans/task1_3_review_2026-10-15.md` |
| 数据脚本 | `src/tools/` |
| 训练代码 | `src/task3/` |

当前锁定的模型是 HOG+LBP + 随机森林，见 `progress/task3_experiment.md` 第 8 节。测试集准确率 0.594、inside 召回 0.357，按地点组拆开是 OUT-06 0.950 / OUT-07 1.000 对 IN-03 0.360 / IN-07 0.353，原因分析见审核报告第 4 节。

## 报告命令

报告正文是 `progress/task6_report_draft.md`，PDF 由它生成，不手写第二份。在仓库根目录：

```
python docs/make_report.py
pdflatex Assignment2_Report_24101322D_HUANG_Haoran.tex
```

改了 `progress/task6_report_draft.md` 之后要重跑这两条，PDF 才和正文一致。`docs/make_report.py` 里有三处后处理：中文招牌包进 CJK 环境、长文件名允许断行、手写题注合并进图题注。

## 训练命令

用 `D:\Program Files\learn_torch\venv`，在仓库根目录：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```

## 本地程序命令

```
python -m src.task5.app                          # 打开选图窗口
python -m src.task5.app --image path\to\photo.jpg # 无图形界面时打印结果
```

程序读取 `models/task3_pipeline.joblib`，预处理调用 `src/task3/preprocess.py` 的 `resize_and_crop`，只用像素。
