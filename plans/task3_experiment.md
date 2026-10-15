# Task 3 — 特征、分类器与实验记录

计划在开跑前写定（第 1–7 节）。第 8 节是 2026-10-14 的实测，报告里的表和选择理由以第 8 节为准，不以 `outputs/` 里的文件为准（那些文件不进 git）。

**课程**：COMP4423 Computer Vision 2026/27 Semester 1
**题目**：PolyU Main Campus Image Classification
**对应评分**：Task 3 提取视觉特征并训练传统分类器（5）+ 说明算法选择、给出训练证据、代码可复现（5）
**数据**：`data/annotations/splits.csv`，图片在 `data/pictures/`
**环境**：`D:\Program Files\learn_torch\venv`（Python 3.12，scikit-learn 1.8，OpenCV contrib 5.0，scikit-image 0.26）

---

## 1. 边界

分类器只读像素。文件名（`IN-` / `OUT-`）、文件夹名、EXIF、GPS 不进特征，只用来查路径和核对标签。

不用神经网络，也不用预训练 CNN 的向量。环境里的 PyTorch、Keras、LightGBM、XGBoost 本轮不调用。评分机应能只靠 scikit-learn、OpenCV、scikit-image、numpy、Pillow、joblib 重跑。

一个物理场所是一个 `group_id`。选择模型的单位是地点组，不是单张图片。数据增强只在划分之后、只加在训练集上。Task 5 必须调用这里的同一个预处理和同一条已保存的 `Pipeline`。

---

## 2. 使用的划分

不重切。`src/tools/build_dataset.py` 已经按地点组划完，540 张，inside 270 : outside 270，图片都在，路径唯一。

| split | outside | inside |
|---|---|---|
| train | 213 张，OUT-01、OUT-02、OUT-03、OUT-04、OUT-05、OUT-08 | 190 张，IN-01、IN-02、IN-04、IN-06 |
| val | 30 张，OUT-09 | 38 张，IN-05 |
| test | 27 张，OUT-06、OUT-07 | 42 张，IN-03、IN-07 |

验证集是主入口草坪对西九海滨，视觉上好分，只用来汇报。测试集是港大红砖和港大绿地对图书馆与 CMA 庭院，锁定模型和特征之后只评估一次，不参与选择。

---

## 3. 算法

两段做，不做「全部特征 × 全部分类器」。第一段回答「哪种表示有用」，第二段回答「在这个表示上，分类器换了有没有差别」。

### 3.1 预处理（所有特征共用）

1. 读成 RGB。
2. 短边缩放到 256，保持宽高比。
3. 中心裁成 256×256。横图左右等量裁掉，竖图上下等量裁掉。
4. 输出 `uint8` 的 256×256×3。HOG / LBP / SIFT 再转灰度。颜色直方图用这张图的 HSV。

水平翻转只作为训练集增强，验证集和测试集不做。不做颜色抖动：夜景和光照是采集时故意留下的轴。

随机种子 `42`。

### 3.2 第一段：固定线性 SVM，比较特征

分类器：`StandardScaler` + `LinearSVC`（或 `SVC(kernel="linear")`），`C=1`。标准化器只在该折训练图上拟合。

| 编号 | 特征 | 设置 | 大约维度 |
|---|---|---|---|
| F1 | 颜色直方图 | HSV，H 16 × S 16 × V 8。整图 1 份 + 2×2 网格 4 份，L1 归一化后拼接 | 10240 |
| F2 | HOG | 灰度，`pixels_per_cell=(16,16)`，`cells_per_block=(2,2)`，9 个方向 | 8100 |
| F3 | LBP | 均匀模式，P=8，R=1，10 个 bin。整图 + 2×2，每块直方图 L1 归一化 | 50 |
| F4 | HOG + LBP | F2、F3 各自标准化后拼接 | 8150 |
| F5 | BoW | SIFT。每图最多 500 个关键点。k-means 词典 **256** 词。词频 L1 归一化 | 256 |
| F6 | BoW + 空间金字塔 | F5 的整图直方图，再加 2×2 四个区域的直方图 | 1280 |

词典、k-means 只在该折训练图的描述子上拟合，验证折和测试集不参与建词典。训练图约 400 张，不用 1000 词的词典。

SIFT 来自 `opencv-contrib` 的 `cv2.SIFT_create`。某一张提不出关键点时，该图记成全零直方图，并在日志里计数。

第一段按第 4 节的地点留出分数排序。取得分最高的 **两种** 特征进入第二段。并列时取维度更小的；仍并列则取表中更靠前的编号。

### 3.3 第二段：固定特征，比较分类器

对上一段留下的每种特征，训练下面的分类器。参数写死，本轮不做大网格。

| 编号 | 分类器 | 设置 | 适用 |
|---|---|---|---|
| C1 | 线性 SVM | `C=1`，特征标准化 | 全部特征 |
| C2 | 逻辑回归 | `C=1`，`lbfgs`，特征标准化 | 全部特征 |
| C3 | k-NN | `k=5`。直方图（F1、F3、F5、F6）用余弦距离；含 HOG 的特征用欧氏距离 | 全部特征 |
| C4 | 随机森林 | 300 棵，`max_depth=12`，`random_state=42` | 全部特征 |
| C5 | 直方图梯度提升 | `HistGradientBoostingClassifier`，`max_depth=6`，`random_state=42` | 全部特征 |
| C6 | 多项式朴素贝叶斯 | `MultinomialNB(alpha=1.0)`，输入为非负且做 L1 的直方图 | 只用于 F1、F3、F5、F6。F2、F4 跳过 |

作业点名的三种都在表里：C1 是 SVM，C4 是 Random Forest，C5 是 Gradient Boosting。C2、C3 是不同的判别方式（线性概率模型、按距离投票），C6 只作为直方图上的简单对照。

第二段同样用地点留出分数。取得分最高的一条（特征 + 分类器）作为最终流水线。并列时按 C2、C1、C6、C3、C4、C5 的顺序取更简单的那个。

线性 SVM 若胜出，再加 **一行** 对照，不参与上面的并列规则之外的扩表：把该特征 PCA 到 100 维（只在训练折上拟合）后跑 `SVC(kernel="rbf", C=1, gamma="scale")`。这行写进结果表，不因此再去试别的核。

### 3.4 本轮不跑的方法

| 方法 | 原因 |
|---|---|
| Extra Trees、单棵决策树 | 与随机森林同一类 |
| AdaBoost | 与梯度提升同属提升法，这批特征上不构成新的对照 |
| LDA、QDA | 高维直方图上类内协方差不稳；低维情况已被逻辑回归和朴素贝叶斯盖住 |
| 最近类中心 | 比 k-NN 更弱的原型法，信息量低 |
| RBF 以外的核（直方图相交、卡方） | 仍是 SVM 换相似度。只保留上面那一行 RBF 对照 |
| VLAD / Fisher Vector | 词袋和空间金字塔之后若仍不够，再单独立项，不放进本轮 |
| LightGBM、XGBoost | 与 C5 同类，且多一个评分机依赖 |
| 任何神经网络或 CNN 特征 | 作业明确排除 |

---

## 4. 实验流程

### 4.1 地点组留出

每个训练地点组只有一个类别，不能单独留出一组再算准确率。每一折同时留出 inside 和 outside。

OUT-05 与 OUT-08 隔路互见，正式划分里已经绑定。内层也把它们当成一个原子，要么都在训练折里，要么一起留出。

5 折，写死，不随机：

| 折 | 留出的 inside | 留出的 outside |
|---|---|---|
| 0 | IN-01 | OUT-01 |
| 1 | IN-02 | OUT-02 |
| 2 | IN-04 | OUT-03 |
| 3 | IN-06 | OUT-04 |
| 4 | IN-01 | OUT-05 与 OUT-08 |

IN-01 出现两次，因为 outside 原子有 5 个、inside 组只有 4 个。分数对 5 折取平均，不把 IN-01 的两折再合并。

每一折：

1. 在留出组之外的训练图片上拟合预处理之后的特征模型（词典、标准化、PCA）。
2. 这些图片可做水平翻转，翻转图只用于拟合，不放进留出集。
3. 在留出的地点上预测，不算翻转。
4. 记录这一折的 balanced accuracy，以及 inside、outside 各自的召回。

一条配置的选择分数 = 5 折 balanced accuracy 的平均。

### 4.2 顺序

1. 解析 `splits.csv`，确认 540 张都能找到，且没有一组跨 split。对不上就退出。
2. 跑第一段 6 条。写 `outputs/task3/stage_a.csv`。
3. 按 3.2 的规则留下 2 种特征。
4. 跑第二段。写 `outputs/task3/stage_b.csv`。若胜出者是线性 SVM，补 RBF 那一行。
5. 在全部训练集上重拟合胜出的流水线（含水平翻转），在验证集上出一次准确率、混淆矩阵、分类别精确率和召回。此分数不反过来改选择。
6. 在测试集上只评估这一次。写 `outputs/task3/test_metrics.json`，并保存每张测试图的预测，供 Task 4 挑例子。
7. `joblib` 保存整条 `Pipeline` 到 `models/task3_pipeline.joblib`。

中断后可以重跑。特征缓存放在 `cache/task3/`，键包含预处理参数和特征编号。缓存只存像素算出来的描述；词袋词典不进跨折缓存，每折单独拟合。

### 4.3 训练证据

报告里两张表，外加锁定后的验证集和测试集各一块。

第一张：6 行特征，列是 5 折平均 balanced accuracy、inside 召回、outside 召回、验证集准确率（验证集列只展示）。

第二张：胜出特征 × 实际跑过的分类器，列同上。

测试集在正文里单列：准确率、混淆矩阵、inside / outside 的精确率和召回、支持数。不把测试集列进选择表。

---

## 5. 代码框架

实现时按此接口，避免 Task 5 另写一套预处理。

```
src/task3/
  config.py       尺寸、词典、分类器参数、种子、5 折配对
  data.py         读 splits.csv，文件名 → 路径，按折取出训练图和留出图
  preprocess.py   resize_and_crop(rgb) -> 256×256×3
  features.py     每种特征一个 sklearn Transformer，fit / transform
  models.py       按编号返回分类器
  cv.py           5 折留出、记分、写 CSV
  run_stage_a.py
  run_stage_b.py
  evaluate.py     验证集汇报 + 测试集一次评估 + 保存 Pipeline
cache/task3/      已 gitignore
models/           已 gitignore
outputs/task3/    已 gitignore
```

调用关系：

```
run_stage_a
  data.load_index(splits.csv, pictures/)
  cv.evaluate(feature_id, "C1")
      preprocess.resize_and_crop
      features.build(feature_id).fit(train_images)
      models.build("C1")
  cv.write(outputs/task3/stage_a.csv)

run_stage_b
  读 stage_a.csv，取前两名特征
  对每个特征 × C1..C6（C6 仅直方图）调用 cv.evaluate
  cv.write(outputs/task3/stage_b.csv)

evaluate
  读 stage_b.csv，锁定一条
  在全部 train 上 fit Pipeline
  预测 val、test
  joblib.dump(pipeline)
```

`features.build` 的 Transformer 约定：

- `fit(images)` 只看训练图。颜色直方图和 LBP、HOG 的 `fit` 可以为空。BoW 的 `fit` 收集 SIFT 并 `MiniBatchKMeans(n_clusters=256, random_state=42)`。
- `transform(images)` 返回 `float64` 的二维数组，行是图，列是该特征的固定维度。
- 最终 `Pipeline` 的步骤顺序：`resize`（若需要包进 transformer）→ 特征 →（标准化，若该分类器要求）→ 分类器。k-NN 和朴素贝叶斯不加 `StandardScaler`。线性 SVM 和逻辑回归要加。

`data.py` 不把 `group_id` 或文件名交给 `transform`。路径解析失败时抛出，不跳过。

`config.py` 里的折配对就是 4.1 的那张表，用元组写死，不在运行时抽样。

命令（实现后）：

在仓库根目录、用 `D:\Program Files\learn_torch\venv`：

```
python -m src.task3.run_stage_a
python -m src.task3.run_stage_b
python -m src.task3.evaluate
```

三次运行读到的 `splits.csv` 不变时，CSV 和测试指标应一致。

---

## 6. 已知风险（写入报告，本轮不改划分）

- **IN-01 在训练集，IN-05 在验证集，两边都拍到创新塔和 VA 一带。** 验证集准确率会偏高。选择模型用的是训练组留出，不是这个验证集。Task 4 要拿测试集错例说明这一点，不能把验证集分数写成泛化结果。
- **测试集 outside 只有 27 张、两个地点。** OUT-06 是红砖，专门对付「见红砖即 inside」。颜色直方图若在第一段领先、在测试集掉下去，这是预期结果，保留在表里，不回头用测试集再挑特征。
- **词袋会记住建筑物的关键点。** 所以词典必须折内拟合，并且选择分数来自没见过的地点组。

---

## 7. 环境

使用 `D:\Program Files\learn_torch\venv` 的 Windows Python 3.12。已确认有 scikit-learn 1.8.0、opencv-contrib-python 5.0.0.93、scikit-image 0.26.0、numpy、Pillow、joblib。

不往这个环境追加包。SIFT 不可用时停下来改计划，不用 ORB 悄悄替换。本次运行 `cv2.SIFT_create` 可用。

---

## 8. 实验结果（2026-10-14）

按第 4 节跑完。原始表在 `outputs/task3/stage_a.csv`、`stage_b.csv`，锁定记录在 `selection.json`，验证集和测试集在 `val_metrics.json`、`test_metrics.json`。下面的数从这些文件抄出。选择只用「地点留出 BA」，验证集准确率只展示。

地点留出 BA = 5 折 balanced accuracy 的平均。inside / outside 召回是这 5 折召回的平均：该类图片里被判对的比例。

### 8.1 第一段：线性 SVM，六种特征

| 特征 | 地点留出 BA | inside 召回 | outside 召回 | 验证集准确率 |
|---|---:|---:|---:|---:|
| F4 HOG+LBP | 0.721 | 0.698 | 0.744 | 0.779 |
| F2 HOG | 0.708 | 0.677 | 0.740 | 0.750 |
| F3 LBP | 0.588 | 0.536 | 0.641 | 0.779 |
| F5 BoW | 0.566 | 0.453 | 0.679 | 0.824 |
| F6 BoW+空间金字塔 | 0.563 | 0.414 | 0.712 | 0.809 |
| F1 颜色直方图 | 0.538 | 0.495 | 0.581 | 0.544 |

进入第二段的是 F4 和 F2。

颜色直方图最低。这和采集时的判断一致：PolyU 的主色是红，但港大红砖也在数据里，只看颜色分不开。词袋的验证集最高（0.824、0.809），地点留出只有 0.56 左右。验证集是主入口草坪对西九海滨，好分；词袋又容易记住某栋楼的关键点。所以不用验证集选特征。

### 8.2 第二段：F4、F2 × 分类器

F2、F4 不是非负直方图，C6 多项式朴素贝叶斯按计划跳过。胜出者不是线性 SVM，没有加 RBF 对照行。

| 特征 | 分类器 | 地点留出 BA | inside 召回 | outside 召回 | 验证集准确率 |
|---|---|---:|---:|---:|---:|
| F4 | C4 随机森林 | **0.757** | 0.660 | 0.855 | 0.691 |
| F2 | C4 随机森林 | 0.748 | 0.649 | 0.846 | 0.676 |
| F2 | C3 k-NN | 0.740 | 0.562 | 0.918 | 0.691 |
| F4 | C3 k-NN | 0.739 | 0.556 | 0.922 | 0.706 |
| F2 | C2 逻辑回归 | 0.725 | 0.685 | 0.765 | 0.735 |
| F4 | C2 逻辑回归 | 0.722 | 0.701 | 0.744 | 0.765 |
| F4 | C5 直方图梯度提升 | 0.721 | 0.648 | 0.793 | 0.603 |
| F4 | C1 线性 SVM | 0.721 | 0.698 | 0.744 | 0.779 |
| F2 | C1 线性 SVM | 0.708 | 0.677 | 0.740 | 0.750 |
| F2 | C5 直方图梯度提升 | 0.701 | 0.626 | 0.775 | 0.676 |

锁定 **F4 HOG+LBP + C4 随机森林**（300 棵，`max_depth=12`）。它的地点留出 BA 最高。k-NN 的 outside 召回更高，但 inside 召回只有 0.56，折 2（留出 IN-04 与 OUT-03）几乎把校内都判成校外。随机森林在这一折上是 F4 里最高的（BA 0.616）。

同一套 F4 特征上，线性 SVM 的验证集准确率是 0.779，高于随机森林的 0.691。仍然选随机森林，因为选择分数是地点留出，不是验证集。

### 8.3 锁定之后的一次评估

在全部训练集上重拟合（含水平翻转），再预测。模型：`models/task3_pipeline.joblib`。

验证集 68 张（IN-05 38，OUT-09 30）。准确率 0.691，balanced accuracy 0.710。

|  | 预测校外 | 预测校内 |
|---|---:|---:|
| 实际校外 | 26 | 4 |
| 实际校内 | 17 | 21 |

outside：精确率 0.605，召回 0.867，F1 0.712。  
inside：精确率 0.840，召回 0.553，F1 0.667。

测试集 69 张（IN-03 25 + IN-07 17 = 42，OUT-06 20 + OUT-07 7 = 27）。只评这一次。准确率 0.594，balanced accuracy 0.660。

|  | 预测校外 | 预测校内 |
|---|---:|---:|
| 实际校外 | 26 | 1 |
| 实际校内 | 27 | 15 |

outside：精确率 0.491，召回 0.963，F1 0.650。  
inside：精确率 0.938，召回 0.357，F1 0.517。

测试集上模型偏向预测校外：27 张校外里错 1 张，42 张校内里错 27 张。逐张预测在 `outputs/task3/test_predictions.csv`，留给 Task 4。这个结果不拿回去改特征或分类器。

### 8.4 写报告时可以用的选择理由

1. 先比特征、再比分类器。400 张训练图上，表示比分类器拉开的差距更大：颜色到 HOG+LBP，地点留出从 0.538 到 0.721；同一特征上五个分类器只从大约 0.70 到 0.757。
2. 不用颜色。主校区的红砖会让「偏红 = inside」得分，港大红砖就是用来堵这条捷径的。颜色直方图的地点留出最低。
3. 不用词袋当最终特征。验证集好看，换地点就掉。词典只在训练折上拟合，仍然记住房子的局部样子。
4. HOG+LBP 略高于单独 HOG（0.721 对 0.708），形状和纹理一起用。
5. 分类器用随机森林，不是因为验证集最高，而是地点留出最高，并且没有像 k-NN 那样在留出的校内地点上召回崩掉。作业点名的 SVM 和梯度提升都跑了，分数低于随机森林。
6. 测试集准确率 0.594 是锁定后的一次结果，不是挑选依据。校内召回低，留到错误分析，不回头重选。
