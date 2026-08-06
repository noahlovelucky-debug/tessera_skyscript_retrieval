# 检索模型对比结果

生成日期：2026-08-06。所有新 checkpoint 均未覆盖已有模型或原始数据。

## 实验范围

SkyScript/TESSERA 对照使用固定的空间切分：训练 `397,178`、验证 `45,299`、闭集测试 `39,344`（`5,586` 个 title 查询）、OOV 测试 `2,793`。title 是训练监督，不是输入图像的完整多对象标注。

| 系统 | 训练数据与视觉输入 | 检索表示 | 训练状态 |
|---|---|---|---|
| 高分门控 `gated_coarse_v3` | SkyScript 高分图 + TESSERA v1 | 高分全局 768 + 高分 `8x1024`，文本门控融合 | 已完成，全量 |
| 高分无门控 `latent_v2` | 同上 | 高分全局 768 + 高分 `8x1024`，固定 0.65/0.35 融合；另输出 TESSERA 分支 | 已完成，全量 |
| 纯 TESSERA+bbox MLP | TESSERA v1 + bbox + title；不读取高分图特征 | `2816+7 -> 768 -> 1024 -> 384`，文本 `768 -> 384` | 已完成，30 epoch，全量 |
| v4.2 多层 Transformer | 独立 Baidu POI S1/S2 token 数据 | `S1/S2 [16,16,384] -> [16,16,768] -> 4-block Transformer -> CLS 384` | 已完成，早停于 22 epoch |

纯 TESSERA 的 7 个 bbox 特征为经纬度的 sin/cos 和物理宽、高、面积的 log 值，并仅以训练切分统计量标准化。它包含地理先验，因此应只作为“v1 向量 + box + label”的消融，不应视为纯视觉模型。

## SkyScript 自动指标

以下相关性由单个 `title_id` 定义，会低估“图中含有查询物但 title 标为另一物”的结果；Luna 视觉审核在下一节单列。

| 系统与检索分支 | P@10 | nDCG@10 | P@100 | nDCG@100 | title mAP |
|---|---:|---:|---:|---:|---:|
| 高分门控，高分候选 | **4.02%** | **8.02%** | **2.14%** | **15.28%** | - |
| 高分无门控，高分 `8x1024` 候选 | 4.00% | 7.91% | 2.11% | 15.04% | - |
| 高分无门控，TESSERA 分支 | 2.21% | 4.27% | 1.21% | 8.07% | 3.42% |
| 纯 TESSERA+bbox MLP | 1.88% | 3.59% | 1.01% | 6.60% | 2.82% |

结论：title 指标下，门控对高分主分支有小幅增益；TESSERA+bbox MLP 比旧单层 TESSERA adapter 更强，但仍低于深层 `latent_v2` TESSERA 分支。它不支持“只用 v1+box 可以替代高分路径”的结论。

来源：

- `artifacts/runs/gated_coarse_v3/evaluation.json`
- `artifacts/runs/latent_v2/evaluation.json`
- `artifacts/runs/tessera_box_v1/evaluation.json`

## Luna Top-10 视觉审核

审核模型为 `gpt-5.6-luna`。固定查询为 `river`、`school`、`farmland`；每个系统每个查询审核其闭集测试候选的前 10 张高分 RGB 图，共 120 张。Luna 只收到查询词和图像像素，不收到 title、OSM 标签、坐标或检索分数。

`视觉折扣相关性` 是布尔视觉相关性按 `1/log2(rank+1)` 加权后的归一化分数，作用等同于二值 nDCG@10 的排序版本。

| 系统 | River P@10 | School P@10 | Farmland P@10 | 平均视觉 P@10 | 平均视觉折扣相关性 |
|---|---:|---:|---:|---:|---:|
| 高分门控 | 90% | 80% | 60% | 76.7% | 76.5% |
| 高分无门控，`8x1024` | **100%** | **100%** | 80% | **93.3%** | **94.9%** |
| `latent_v2` TESSERA 分支 | 50% | 90% | **90%** | 76.7% | 77.1% |
| 纯 TESSERA+bbox MLP | 60% | 40% | 80% | 60.0% | 61.2% |

这份审核表是当前最贴近“文本搜到包含目标的高清图”目标的证据，但样本只有 3 个查询，不能报告为全类别总体性能。它说明两点：

1. 自动 title 指标确实漏记了一部分视觉正确候选，尤其是图像包含多个地物时。
2. 在这三个查询上，固定局部/全局融合的高分无门控模型优于当前文本门控；下一步应在更多查询上复核，而不是仅凭该小样本删除门控。

审计产物：

- `artifacts/audits/luna_model_compare_v2/summary.json`
- `artifacts/audits/luna_model_compare_v2/candidates_luna_judged.csv`
- `artifacts/audits/luna_model_compare_v2/luna_raw_responses.json`

## 扩展 Luna 12 类审核

对当前视觉最优的无门控高分 `latent_v2`，固定 12 个开放词汇查询，每类审核闭集候选 Top-10 的实际高分图，共 120 张。绿色边框为 Luna 判定相关，红色边框为不相关。

| 查询 | 正确 / 10 | 视觉 P@10 | 视觉 nDCG@10 |
|---|---:|---:|---:|
| river | 10 / 10 | 100% | 100.0% |
| school | 10 / 10 | 100% | 100.0% |
| farmland | 8 / 10 | 80% | 84.8% |
| hospital | 10 / 10 | 100% | 100.0% |
| airport | 9 / 10 | 90% | 89.0% |
| bridge | 10 / 10 | 100% | 100.0% |
| railway | 10 / 10 | 100% | 100.0% |
| swimming pool | 10 / 10 | 100% | 100.0% |
| golf course | 10 / 10 | 100% | 100.0% |
| parking lot | 10 / 10 | 100% | 100.0% |
| industrial building | 10 / 10 | 100% | 100.0% |
| cemetery | 10 / 10 | 100% | 100.0% |
| **平均** | **9.75 / 10** | **97.5%** | **97.8%** |

这组扩展审核中较弱的是农田和机场。农田的第 6、7 名为住宅庭院草地而非耕地，说明模型仍会把相邻的绿色覆盖误认为农田。它不是完整类覆盖或人工金标准，不能单独作为商用结论。

产物：

- `artifacts/audits/luna_latent_v2_extended_v1/summary.json`
- `artifacts/audits/luna_latent_v2_extended_v1/candidates_luna_judged.csv`
- `artifacts/audits/luna_latent_v2_extended_v1/visualizations/`

## 全部 Title 类

数据集没有一个小型、互斥的预定义类别表。它以英文细粒度 title 作为训练标签，共 `25,220` 个 title；例如不同类型的桥、道路、球场、屋顶和建筑会被分成不同 title。完整清单及各切分样本数见 `artifacts/audits/result/title_inventory.csv`。

## TESSERA v4.2 MLP 与 Transformer

这一组使用的是 `/data/code/tessera/tessera-vlm/patch_vit_retrieval` 的 Baidu POI 数据，而非上述 52 万 SkyScript 样本。候选、标签、S1/S2 token 均不同，不能与上表的 Luna 视觉 P@10 横向比较。它回答的是更窄的问题：在相同 v4.2 数据、损失、切分和文本塔下，以多层 Transformer 替换图像 MLP/attention-pool 能否改善检索。

| 闭集锁定测试，847 POI | 旧 v4.2 MLP 全局 | 新 4-block Transformer 全局 | 旧 v4.2 MLP 局部 | 新 4-block Transformer 局部 |
|---|---:|---:|---:|---:|
| mAP | 16.35% | **17.43%** | **27.13%** | 25.95% |
| Macro Category R@1 | 15.79% | **21.05%** | **39.47%** | 23.68% |
| Category R@1 | 13.11% | **29.40%** | **45.57%** | 32.70% |
| Exact token Top-1 | 4.25% | 3.78% | **4.25%** | 3.78% |

结论：Transformer 的全局粗检索提升明显，但局部检索与 token 定位下降。因此它是一个有价值的全局分支备选，不能替代旧 v4.2 的局部 MLP 主分支。Transformer 的最佳 checkpoint 为第 10 epoch，由验证集选择；未使用测试集选模型。

来源：

- 旧：`/home/star/tessera_runs/baidu_patch_vit_hierarchical_v4_2/final_test_retrieval.json`
- 新：`artifacts/runs/patch_vit_transformer_v42/final_test_metrics.json`

## 代码与产物位置

| 内容 | 路径 |
|---|---|
| 纯 TESSERA MLP 配置 | `configs/tessera_box_v1.yaml` |
| 纯 TESSERA MLP 代码 | `tessera_skyscript_retrieval/tessera_box.py` |
| 纯 TESSERA 训练产物 | `artifacts/runs/tessera_box_v1/` |
| v4.2 Transformer 补丁 | `v42_transformer_patch/` |
| v4.2 Transformer 训练产物 | `artifacts/runs/patch_vit_transformer_v42/` |
| Luna 审核脚本 | `scripts/audit_retrieval_luna.py` |

复跑 Luna 审核时，密钥仅通过环境变量传入：

```bash
cd /data/code/tessera_skyscript_retrieval
PYTHONPATH=. LUNA_API_KEY="$LUNA_API_KEY" CUDA_VISIBLE_DEVICES=0 \
python scripts/audit_retrieval_luna.py \
  --queries river school farmland --top-k 10 --judge-workers 1 \
  --output-dir artifacts/audits/luna_model_compare_v2
```
