# Task 5 — 本地分类程序

**日期**：2026-10-19
**课程**：COMP4423 Computer Vision 2026/27 Semester 1
**对应评分**：Task 5 Application（20）

作业要求一个本地程序：用户选择或上传一张图，窗口显示这张图，并返回 **Inside PolyU main campus** 或 **Outside PolyU main campus**。必须加载 Task 3 已保存的分类器，预处理与评估相同，只用像素，并写明如何启动。

本轮没有改 `data/annotations/splits.csv`，没有重训，没有改特征或分类器的选择。锁定模型仍是 `models/task3_pipeline.joblib`（HOG+LBP + 随机森林）。

---

## 1. 界面

桌面程序用标准库 Tk，不往 `requirements.txt` 加包。Pillow、joblib、OpenCV、scikit-learn 都是 Task 3 已有的依赖。

窗口里：

1. **Select image…** 打开文件对话框（Ctrl+O 同样）。可选 jpg、jpeg、png、bmp、tif、tiff、webp。
2. 左侧显示用户选中的原图，并写上原始宽高。
3. 右侧大字给出 `Pipeline.display_` 里的标签，即上面那两句英文。
4. 随机森林有 `predict_proba`，所以同时画出 inside / outside 的概率。
5. 小图是送进模型的 256×256 中心裁剪，来自 `resize_and_crop` 的返回值。

文件名只显示在按钮旁边，告诉用户当前是哪一张。判别函数不接收路径。

没有显示器时，同一条读图、预处理和 Pipeline 可以只打印结果：

```
python -m src.task5.app --image path/to/photo.jpg
```

---

## 2. 怎样接上 Task 3

代码在 `src/task5/predict.py` 和 `src/task5/app.py`。调用顺序和 `src/task3/evaluate.py` 一致：先裁好，再 `predict`。

```
load_rgb(path)                         # src.task3.preprocess，RGB uint8
resize_and_crop(rgb)                   # 短边 256，中心裁成 256×256×3
pipeline.predict([cropped])            # 0 = outside，1 = inside
pipeline.predict_proba([cropped])      # 两类概率；估计器没有该方法时跳过
pipeline.display_[label_id]            # 保存模型时写入的英文标签
```

`joblib.load` 得到的是 sklearn `Pipeline`，两步：

| 步骤 | 对象 | 内容 |
|---|---|---|
| `features` | `ImageFeatures(feature_id="F4")` | `HogLbp` |
| `clf` | `RandomForestClassifier` | 300 棵，`max_depth=12`，`random_state=42` |

`classes_` 是 `[0, 1]`。`display_` 是 `{0: "Outside PolyU main campus", 1: "Inside PolyU main campus"}`。

`Pipeline.predict` 要的是图像列表。一个 `H×W×3` 的 ndarray 若直接传进去，会被当成逐行迭代，所以这里固定写成 `pipeline.predict([cropped])`。

`HogLbp.transform` 内部还会再调用一次 `resize_and_crop`。图已经是 256×256 时，该函数原样返回，和评估里「`load_cropped` 之后再 `predict`」的顺序相同。程序里显式调用的就是 `src.task3.preprocess.resize_and_crop`，没有另写缩放。

读图也用 `load_rgb`，不用另一套解码，也不做 EXIF 转正。训练和评估都没有转正，界面若单独转正，像素就会和模型见过的不一致。

路径、文件夹、EXIF、GPS 不进 `predict_rgb`。`load_and_predict` 只用路径做两件事：读像素，以及在界面上显示文件名。

---

## 3. 怎么运行

仓库根目录，沿用 Task 3 的环境（Windows 上是 `D:\Program Files\learn_torch\venv`）：

```
python -m src.task5.app
```

也可以 `python src/task5/app.py`。Windows 上不想留控制台时用 `pythonw -m src.task5.app`。

模型文件不在时，窗口会说明要先运行 `python -m src.task3.evaluate`。本仓库里的 `models/task3_pipeline.joblib` 已经在，这条命令会重训，Task 5 本身不会去跑它。

Ubuntu 若提示没有 `_tkinter`，安装 `python3-tk`。Windows 官方安装包自带 Tk。

---

## 4. 2026-10-19 验证

在本机用已安装的 `requirements.txt` 跑过。单张图从读入到给出标签大约 0.1 秒（含 HOG+LBP，以及 `predict` 和 `predict_proba`）。

裁剪结果都是 `(256, 256, 3)`。对裁剪结果再调用一次 `resize_and_crop`，数组不变。把未裁剪的原图直接送进 `pipeline.predict`，标签与先裁再预测相同。

同一批像素存成 `IN-99_inside_campus.jpg` 和 `OUT-99_outside_campus.jpg`，标签和概率相同。界面上文件名会变成 `OUT-99_outside.jpg`，预测句子不变。读不了的文本文件：窗口弹出错误并且保留上一次的标签；命令行退出码为 1。

`data/pictures/` 不在这次检出里，所以没有用 `splits.csv` 里的原图重放测试集。作业 PDF 第 2 页的示例图是左右并排的两张说明照片，这里按中线切开，只检查程序会给出标签、不会崩溃。这不是测试集，也不拿来改模型。

| 图 | 像素 | 程序输出 | outside | inside |
|---|---|---|---:|---:|
| 示例图左半（庭院，说明文字标为 inside） | 960×1080 | Outside PolyU main campus | 0.512 | 0.488 |
| 示例图右半（街道，说明文字标为 outside） | 960×1080 | Inside PolyU main campus | 0.460 | 0.540 |

两条都接近 0.5。这和已锁定的 HOG+LBP 在没见过的场所上偏弱一致，见 `plans/task1_3_review_2026-10-15.md` 第 4 节。Task 5 只负责把这条流水线原样接出来。

另外用噪声图、1×1、超宽图、全黑、全白、灰度 JPEG、带透明通道的 PNG，以及一张 3840×2400 的 JPEG 各跑了一次，都返回了上面两个英文标签之一。

图形界面在桌面上打开，窗口大约 880×585。选图之后画布上有原图，小图上有裁剪，预测文字是上述英文句子。命令 `python -m src.task5.app --image <文件>` 和 `python src/task5/app.py --image <文件>` 都打印同一句标签和两类概率。

## 5. 2026-10-21 补测：用测试集全量回放

第 4 节写的检查是在没有 `data/pictures/` 的检出里做的，所以当时没有用原图回放测试集。照片齐了以后补做了一次。

做法：对 `outputs/task3/test_predictions.csv` 里的 69 个文件，逐个调用 `src/task5/predict.py` 的 `predict_rgb`，也就是窗口调用的那个函数，用的是 `src/task3/preprocess.py` 的 `resize_and_crop` 和 `models/task3_pipeline.joblib`。

结果：**69 张里 69 张的标签与锁定预测表一致，没有一处不符。**

这条的意义是把「应用和评估是同一条流水线」从论证变成实测。Task 5 读的是同一个 joblib，走的是同一个 `resize_and_crop`，算的是同一个特征，得到的标签和评估表逐张相同。

顺带生成 `docs/figures/task5_application.png`：用应用自己的预测路径处理一张真实照片，画出选中的文件、模型实际收到的 256×256 中心裁剪、程序返回的句子和两类概率。选的是 `OUT-06_001_IMG_0863.jpg`，港大主楼那张红砖难负样本，锁定流水线判 outside，概率 0.745。画法见 `src/task5/make_app_figure.py`。
