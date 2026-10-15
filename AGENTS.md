# COMP4423 Assignment 2

判断一张场景照片拍在 PolyU 主校区内还是校外。数据说明在 `plans/task1_dataset_overview.md`。

## 规则

1. 每个 task 的思路、过程和结果写在 `plans/` 里，文件名像 `task3_experiment.md`。报告只从这些记录取数。实验跑完把结果写回同一份文件；`outputs/` 被 git 忽略，不能当作唯一记录。
2. 分类器只用像素。文件名、文件夹名、EXIF、GPS 不进特征。Task 3 不用神经网络，也不用 CNN 特征。
3. 不按测试集改模型或重选特征。测试集只评估已经锁定的一条流水线。
4. 不改 `data/annotations/splits.csv` 的划分，除非用户明确要求。地点组是划分单元。

## 何时读

- 改采集、标注或划分：`plans/task1_dataset_overview.md`，脚本在 `src/tools/`
- 改训练或特征：`plans/task3_experiment.md`，再读 `src/task3/AGENTS.md`

## 训练命令

用 `D:\Program Files\learn_torch\venv`，在仓库根目录：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```
