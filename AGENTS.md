# COMP4423 Assignment 2

PolyU 主校区场景照片二分类：inside = 1，outside = 0。截止 2026-10-28。提交报告 PDF 和代码 ZIP。

## 规则

1. 每个 task 的思路、过程和结果写在 `plans/`，文件名像 `task3_experiment.md`。报告只从这些记录取数。实验跑完把结果写回同一份文件。`outputs/` 和 `models/` 被 git 忽略，不能当作唯一记录。
2. 一个物理场所是一个 `group_id`，也是 train / val / test 的划分单元。不改 `data/annotations/splits.csv`，除非用户明确要求。图片按文件名在 `data/pictures/` 里解析。
3. 分类器只用像素。文件名、文件夹名、EXIF、GPS 只用来核对标签和找路径。Task 3 不用神经网络，也不用 CNN 特征。
4. 不按测试集改模型或重选特征。测试集只评估已经锁定的一条流水线。新的一轮实验追加到对应 `plans/` 记录的新日期小节，不覆盖已有结果。
5. Task 5 的本地程序必须调用 Task 3 保存的 `Pipeline` 和 `src/task3/preprocess.py` 里的 `resize_and_crop`，不能另写预处理，也不能从路径推断类别。

## 记录

| 范围 | 文件 |
|---|---|
| Task 1–2 类别、采集、标注、划分 | `plans/task1_dataset_overview.md` |
| Task 3 特征、分类器、实测 | `plans/task3_experiment.md`（第 8 节是 2026-10-14 的结果） |
| 数据脚本 | `src/tools/` |
| 训练代码 | `src/task3/` |

当前锁定的模型是 HOG+LBP + 随机森林，见实验记录第 8 节。

## 训练命令

用 `D:\Program Files\learn_torch\venv`，在仓库根目录：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```
