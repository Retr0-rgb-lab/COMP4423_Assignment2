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

## 记录

| 范围 | 文件 |
|---|---|
| Task 1–2 类别、采集、标注、划分 | `progress/task1_dataset_overview.md` |
| Task 3 特征、分类器、实测 | `progress/task3_experiment.md`（第 8 节是 2026-10-14 的结果） |
| Task 1–3 审核与整改 | `plans/task1_3_review_2026-10-15.md` |
| 数据脚本 | `src/tools/` |
| 训练代码 | `src/task3/` |

当前锁定的模型是 HOG+LBP + 随机森林，见 `progress/task3_experiment.md` 第 8 节。测试集准确率 0.594、inside 召回 0.357，按地点组拆开是 OUT-06 0.950 / OUT-07 1.000 对 IN-03 0.360 / IN-07 0.353，原因分析见审核报告第 4 节。

## 训练命令

用 `D:\Program Files\learn_torch\venv`，在仓库根目录：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```
