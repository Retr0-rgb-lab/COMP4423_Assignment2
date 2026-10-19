# COMP4423 Assignment 2

判断一张场景照片拍在 PolyU 主校区内（inside）还是校外（outside）。

环境：`D:\Program Files\learn_torch\venv`（Python 3.12）。依赖见 `requirements.txt`。其中必须安装 `opencv-contrib-python`，不要改成 `opencv-python`。

在仓库根目录：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```

第一条比较六种特征，第二条在胜出的特征上比较分类器，第三条在全部训练集上重拟合锁定的流水线，并只评估一次测试集。已有结果若与 `splits.csv` 或 `src/task3/` 的代码不一致，续跑会退出，确认后加 `--force`。

划分表是 `data/annotations/splits.csv`。逐类逐 split 的计数在 `data/annotations/dataset_summary.md`。采集与标注说明在 `progress/task1_dataset_overview.md`。特征、分类器和实测在 `progress/task3_experiment.md`。

锁定的模型是 `models/task3_pipeline.joblib`（HOG+LBP + 随机森林）。预测只使用像素，经 `src/task3/preprocess.py` 的 `resize_and_crop`。

## Task 5 本地程序

在仓库根目录打开选图窗口：

```
python -m src.task5.app
```

点 **Select image…**（或按 Ctrl+O），选一张场景照片。窗口左侧显示这张图，右侧给出 **Inside PolyU main campus** 或 **Outside PolyU main campus**，并画出两类概率。模型输入是同一张图的 256×256 中心裁剪。

程序读取上面的 `task3_pipeline.joblib`，预处理调用 `resize_and_crop`。预测只用像素。文件名、文件夹、EXIF、GPS 不参与判别，也不会在这里重新训练。

没有图形界面时，用同一条流水线打印一张图：

```
python -m src.task5.app --image path\to\photo.jpg
```

Windows 官方 Python 自带 Tk。只要窗口、不要控制台时，用 `pythonw -m src.task5.app`。Ubuntu 缺少 Tk 时安装 `python3-tk`。不新增依赖。设计和验证记录在 `progress/task5_app.md`。
