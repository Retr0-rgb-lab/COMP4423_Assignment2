# 数据集清洗与划分结果

> 由 `src/tools/build_dataset.py` 生成，请勿手工编辑。

## 清洗规则

- 只保留 `status = include`；剔除 exclude 16 张、review 0 张
- 单一地点组上限 50 张（组内等距取样，保留视点跨度）
- 类别对齐到较小的一侧
- **地点组为不可分割单元**：同一组全部照片进入同一 split
- 互见组绑定同 split：OUT-05 / OUT-08（隔路可互相看到）

## 每类每 split 的图片数与地点组数

| split | 类别 | 图片数 | 地点组数 | 地点组 |
|---|---|---:|---:|---|
| train | outside | 213 | 6 | OUT-01, OUT-02, OUT-03, OUT-04, OUT-05, OUT-08 |
| train | inside | 190 | 4 | IN-01, IN-02, IN-04, IN-06 |
| val | outside | 30 | 1 | OUT-09 |
| val | inside | 38 | 1 | IN-05 |
| test | outside | 27 | 2 | OUT-06, OUT-07 |
| test | inside | 42 | 2 | IN-03, IN-07 |

## 各地点组明细

| group_id | 场所 | 类别 | 原始 | 入库 | split |
|---|---|---|---:|---:|---|
| OUT-01 | 佐敦 Jordan | outside | 82 | 50 | train |
| OUT-02 | 港大 HKU | outside | 57 | 50 | train |
| OUT-03 | 九龙公园 Kowloon Park | outside | 72 | 32 | train |
| OUT-04 | 圆方 Elements | outside | 62 | 30 | train |
| OUT-05 | 西九文化區 West Kowloon | outside | 21 | 21 | train |
| OUT-06 | 港大 HKU – Main Building | outside | 20 | 20 | test |
| OUT-07 | 港大 HKU – 校园绿地 | outside | 7 | 7 | test |
| OUT-08 | 西九文化區 – Art Park | outside | 38 | 30 | train |
| OUT-09 | 西九文化區 – 維港海濱 | outside | 53 | 30 | val |
| IN-01 | PolyU 走道（J 區 — VA 區） | inside | 71 | 50 | train |
| IN-02 | PolyU P/Q 區藻井走道 | inside | 62 | 50 | train |
| IN-03 | PolyU 大學圖書館與中央平台 | inside | 25 | 25 | test |
| IN-04 | PolyU FSCH / VS 區 / 泳池 | inside | 53 | 50 | train |
| IN-05 | PolyU 校門與主入口 | inside | 38 | 38 | val |
| IN-06 | PolyU 樓群庭院與 A 座入口 | inside | 40 | 40 | train |
| IN-07 | PolyU CMA BUILDING 庭院 | inside | 17 | 17 | test |

## 汇总

- inside 270 张 · outside 270 张
- 地点组 16 个（inside 7 · outside 9）
- 总计 540 张
