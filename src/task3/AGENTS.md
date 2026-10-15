# Task 3

改特征、分类器或折之前，先读 `plans/task3_experiment.md`。第 8 节是已经跑完的结果。

锁定的流水线是 F4（HOG+LBP）+ C4（随机森林）。不要因为 `outputs/task3/test_metrics.json` 改这组选择。新的一轮实验把结果追加到该文件的新日期小节，不要覆盖第 8 节。

仓库根目录、`D:\Program Files\learn_torch\venv`：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```

`preprocess.resize_and_crop` 是唯一的缩放入口。Task 5 调用保存的 `models/task3_pipeline.joblib`，不要另写预处理。
