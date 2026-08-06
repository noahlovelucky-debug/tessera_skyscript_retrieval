# TESSERA–SkyScript 开放词汇图文检索

本项目验证一个明确问题：现有 SkyScript 高分图、英文文本和同位置 TESSERA v1 Sentinel-1/2 年度表征，能否被对齐到同一语义空间，并用于英文文本搜图和坐标返回。

项目包含两套可复现实验：`configs/default.yaml` 是仅训练轻量 TESSERA MLP adapter 的基线；`configs/latent_v2.yaml` 是当前主模型，增加高分 `8×1024` 局部潜层分支和更深的 TESSERA v1 多 token adapter。两者都不微调 TESSERA v2 ViT，也不把 v1 embedding 误当成 v2 原始输入。

## 1. 数据现状

输入数据位于：

```text
/data/tessera_dataset/image-emb-text/
├── manifest_520k.csv              # sample_id, filepath, title
├── images2 ... images7/           # SkyScript高分JPG
└── tessera_chips/npy/*.npy        # H×W×128 float16 TESSERA v1特征

/data/tessera_v2_sentinel/tessera_v2/
└── meta2 ... meta7/*.pickle       # bbox、时间和OSM标签
```

原始 CSV 有 520,000 条。默认只保留真正的高分来源 `CH/US/FI/ES`，预期得到 517,669 条；`S2/L8/L9` 被排除。

`.npy` 是 Sentinel-1/2 年度时序经 TESSERA v1 编码后、按高分图 bbox 裁出的 10m 网格特征，不是原始波段：

```text
Sentinel-1/2年度时序 -> TESSERA v1 -> 全球H×W×128表征 -> 高分bbox裁片
```

准备后的 `prepared_manifest.parquet` 每行包含：

| 字段 | 含义 |
|---|---|
| `row_id`, `sample_id`, `part` | 稳定行号、OSM样本ID和分片 |
| `image_path`, `chip_path`, `metadata_path` | 高分图、TESSERA chip和pickle路径 |
| `title`, `title_id` | 英文caption和语义类别ID |
| `source`, `year` | 高分来源和TESSERA年度 |
| `bbox_*`, `center_lon`, `center_lat` | 地理范围和中心坐标 |
| `chip_h`, `chip_w` | 10m TESSERA空间网格大小 |
| `split_group`, `split` | 0.1度空间组和train/val/test/OOV切分 |

空间网格不会跨 split。OOV 标题完全不参与训练，只在测试网格中验证。

## 2. 教师模型与框架

教师是冻结的 `SkyCLIP ViT-L/14 top30pct filtered by LAION-RS`，整个 SkyCLIP 包含图像塔和文本塔。`latent_v2` 的数据流为：

```text
高分JPG [任意原始宽高]
  -> SkyCLIP预处理为224×224
  -> 冻结ViT-L/14的16×16×1024 patch tokens
  -> 自适应池化为2×4×1024
  -> 可训练2层Transformer
  -> z_high_local [8,1024]
  -> 均值残差投影到冻结教师全局向量
  -> z_high_global [768]

英文文本 -> 冻结SkyCLIP text transformer -> z_text_global [768]
  -> 可训练文本投影 -> z_text_local [1024]

TESSERA chip [H,W,128]
  -> 空间金字塔平均池化1×1/2×2/4×4 + 全局std
  -> descriptor [2816]，重排为22×128 token
  -> 128→768投影 + 可训练全局token
  -> 可训练4层Transformer
  -> z_tessera [768]
```

SkyCLIP 两个冻结教师输出已经处于共同空间。正式模型训练 `55,780,867` 个参数：TESSERA 分支 `31,619,328`，高分局部与文本投影分支 `24,161,537`，另含两个温度参数。损失同时约束：

```text
L = 1.00 * 文本-TESSERA多正样本InfoNCE
  + 0.50 * 文本-高分全局多正样本InfoNCE
  + 1.00 * 文本-高分8个局部向量MaxSim InfoNCE
  + 0.50 * 同sample高分-TESSERA余弦蒸馏
  + 0.25 * TESSERA与冻结教师的相似度分布KL蒸馏
  + 0.25 * 高分全局教师保持损失
  + 0.02 * 8个局部向量多样性正则
```

同标题的图像全部视为正样本，避免把语义相同的高分图错误当成负例。

## 3. 安装与权重

所有缓存放在本项目下，避免写满系统盘：

```bash
cd /data/code/tessera_skyscript_retrieval
bash scripts/setup_environment.sh
source .venv/bin/activate
bash scripts/fetch_skyclip.sh
```

下载脚本固定 SkyScript commit `b16d2e76c5a0cdd644e2422a4446fb092d2dc1e4`，并下载 4,486,300,088 bytes 的官方权重。

## 4. 先跑256条latent_v2 smoke test

`--limit` 必须在各阶段保持一致：

```bash
python -m tessera_skyscript_retrieval prepare \
  --config configs/latent_v2.yaml --limit 256

python -m tessera_skyscript_retrieval cache-tessera \
  --config configs/latent_v2.yaml --limit 256

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m tessera_skyscript_retrieval cache-skyclip \
  --config configs/latent_v2.yaml --limit 256

python -m tessera_skyscript_retrieval train \
  --config configs/latent_v2.yaml --limit 256

python -m tessera_skyscript_retrieval evaluate \
  --config configs/latent_v2.yaml --limit 256

python -m tessera_skyscript_retrieval build-index \
  --config configs/latent_v2.yaml --limit 256
```

如果 256 条刚好没有验证网格，改用 `--limit 4096`。

## 5. 完整训练

完整准备会覆盖 smoke manifest，因此正式运行时所有阶段都去掉 `--limit`：

```bash
python -m tessera_skyscript_retrieval prepare --config configs/latent_v2.yaml
python -m tessera_skyscript_retrieval cache-tessera --config configs/latent_v2.yaml

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tessera_skyscript_retrieval cache-skyclip --config configs/latent_v2.yaml

CUDA_VISIBLE_DEVICES=0 python -m tessera_skyscript_retrieval train --config configs/latent_v2.yaml
CUDA_VISIBLE_DEVICES=0 python -m tessera_skyscript_retrieval evaluate --config configs/latent_v2.yaml
CUDA_VISIBLE_DEVICES=0 python -m tessera_skyscript_retrieval build-index --config configs/latent_v2.yaml
```

训练只消费缓存后的固定特征，因此单张 A100 足够；8卡主要用于一次性生成 517,669 张高分图的 SkyCLIP 特征。

## 6. 文本搜索与坐标返回

```bash
python -m tessera_skyscript_retrieval search \
  --config configs/latent_v2.yaml \
  --query "quarry area" --modality both --top-k 10

python -m tessera_skyscript_retrieval visualize \
  --config configs/latent_v2.yaml \
  --query "quarry area" --top-k 5
```

可视化默认输出到 `artifacts/visualizations/`：`precision_ndcg_at_10_100.png` 是冻结教师、训练后全局向量、潜层精排和 TESSERA 路线的主指标对比，`{query}_retrieval.png` 是高分与 TESSERA 检索预览，配套 JSON 保留完整路径、分数和坐标。

`--modality` 可选：

- `highres`：先用768维全局向量从517,669条中预筛1000条，再用文本1024维向量和每幅图的8个1024维局部向量做 MaxSim 精排；最终分数为65%局部分数加35%全局分数。
- `tessera`：搜索深层 adapter 后的 Sentinel/TESSERA 768维全局向量。
- `both`：同时搜索并按 `sample_id` 去重。

结果包含相似度、命中模态、高分图路径、TESSERA chip路径、年份、bbox和中心经纬度。坐标来自 metadata，不由向量预测。

## 7. 评估指标

评估分别报告：

- 文本到高分图：冻结 SkyCLIP 教师、训练后768维全局向量和 Top-1000 后的8×1024潜层精排。
- 文本到 TESSERA：adapter 的主要效果。
- 高分图到 TESSERA：跨分辨率语义一致性。
- 闭集空间测试和未见标题 OOV 测试。
- 主指标：Precision@10、nDCG@10、Precision@100、nDCG@100。
- 辅助指标：Hit@1/5/10/100 和 macro mAP；Hit@K 表示前 K 个结果是否至少有一个相关项。
- 同 `sample_id` 精确 Recall@K，仅作为诊断，不作为主指标。

语义相关性当前按完全相同的 `title_id` 判定。Precision@K 是前 K 个结果中相关结果的平均占比；nDCG@K 同时衡量相关结果数量和排序位置，越靠前权重越高。若要评估“farmland、crop field、agricultural land 都算农田”，必须先增加概念级同义词映射或人工多标签，不能直接把当前精确标题指标解释成宽泛地物概念指标。

## 8. 当前全量实验结果

当前目录已经完成旧 MLP 基线和 `latent_v2` 两次全量运行：

| 产物 | 结果 |
|---|---|
| 有效配对 | 517,669行，25,220个唯一标题 |
| 数据切分 | train 397,178 / val 45,299 / test 39,344 / OOV test 2,793 |
| TESSERA descriptor | `(517669, 2816)` float16 |
| SkyCLIP高分特征 | `(517669, 768)` float16 |
| SkyCLIP局部token缓存 | `(517669, 8, 1024)` float16，8.8GB缓存目录 |
| latent_v2可训练参数 | 55,780,867 |
| latent_v2全量训练 | 30 epochs，约1小时25分；最佳epoch 28，验证组合分数0.070956 |
| latent_v2检索索引 | 517,669组高分/TESSERA全局向量、8×1024高分潜层及坐标，共9.5GB |

`latent_v2` 正式结果见 `artifacts/runs/latent_v2/evaluation.json`。高分精排严格按线上流程：先用全局向量预筛1000条，再以65%潜层MaxSim加35%全局分数重排。

| 切分/任务 | P@10 | nDCG@10 | P@100 | nDCG@100 | macro mAP |
|---|---:|---:|---:|---:|---:|
| closed SkyCLIP teacher | 0.0265 | 0.0523 | 0.0158 | 0.1079 | 0.0476 |
| closed text→trained highres global | 0.0364 | 0.0726 | 0.0197 | 0.1403 | 0.0616 |
| closed text→highres 8×1024 fine | **0.0400** | **0.0791** | **0.0211** | **0.1504** | - |
| closed text→TESSERA | 0.0221 | 0.0427 | 0.0121 | 0.0807 | 0.0342 |
| closed highres→TESSERA | 0.0278 | 0.0716 | 0.0128 | 0.1184 | 0.0581 |
| OOV SkyCLIP teacher | 0.4184 | 0.4422 | 0.2243 | 0.5864 | 0.4257 |
| OOV text→highres 8×1024 fine | **0.4776** | **0.5231** | 0.2209 | **0.6321** | - |
| OOV text→TESSERA | 0.3184 | 0.3557 | 0.1518 | 0.4183 | 0.2750 |
| OOV highres→TESSERA | 0.3171 | 0.3543 | 0.1476 | 0.4105 | 0.2590 |

相对旧基线，闭集高分精排的 P@10、nDCG@10、P@100、nDCG@100 分别提升 `50.78% / 51.38% / 33.81% / 39.29%`；TESSERA 全局分支分别提升 `24.65% / 26.21% / 21.70% / 26.31%`；高分到 TESSERA 跨模态检索分别提升 `86.09% / 90.52% / 66.09% / 70.03%`。OOV 高分精排的 P@100 小幅下降1.52%，其余主指标提高。

旧 MLP 基线结果见 `artifacts/runs/default/evaluation.json`：

| 切分/任务 | P@10 | nDCG@10 | P@100 | nDCG@100 | macro mAP |
|---|---:|---:|---:|---:|---:|
| closed text→highres | 0.0265 | 0.0523 | 0.0158 | 0.1079 | 0.0476 |
| closed text→TESSERA | 0.0177 | 0.0338 | 0.0100 | 0.0639 | 0.0266 |
| closed highres→TESSERA | 0.0149 | 0.0376 | 0.0077 | 0.0696 | 0.0307 |
| OOV text→highres | 0.4184 | 0.4422 | 0.2243 | 0.5864 | 0.4257 |
| OOV text→TESSERA | 0.2882 | 0.3230 | 0.1417 | 0.3870 | 0.2350 |
| OOV highres→TESSERA | 0.2263 | 0.2380 | 0.1203 | 0.3007 | 0.1645 |

OOV 测试只有76个高频未见标题和2,793个候选；闭集测试有5,586个标题和39,344个候选，两组数字不能直接横向比较。旧报告中的 `Semantic Recall@K` 实际计算的是“至少命中一个”的 Hit@K，现已纠正命名并降为辅助指标。`latent_v2` 精确同样本 highres→TESSERA 的 R@1/5/10/100 为 `0.0502/0.1454/0.2102/0.5271`，它比标题级语义指标更严格。

### 与 tessera-vlm 的同指标对照

使用 `tessera-vlm` 的 multiscale v4 checkpoint，并按同一二元相关性公式重新计算：

| 查询/候选口径 | 查询数 | 候选数 | P@10 | nDCG@10 | P@100 | nDCG@100 |
|---|---:|---:|---:|---:|---:|---:|
| query_style→POI rows | 558 | 558 | 0.4432 | 0.4679 | 0.2232 | 0.5091 |
| query_style→unique patches | 558 | 428 | 0.4724 | 0.4976 | 0.2279 | 0.5628 |
| natural 28-class→POI rows | 28 | 558 | 0.2286 | 0.2765 | 0.1175 | 0.3983 |
| natural 28-class→unique patches | 28 | 428 | 0.2571 | 0.3089 | 0.1232 | 0.4451 |

对“输入一个类别短语，找包含该地物的图”而言，最后一行最接近实际使用方式。但它是北京固定28类、仅428个候选的验证集，且验证集参与过模型选择；本项目闭集则有5,586个精确标题和39,344个候选。因此该表只能说明各自在本数据口径下的效果，不能据此直接断言某个模型更强。完整机器可读结果见 `artifacts/runs/default/tessera_vlm_comparison.json`。

CLI 每次搜索会冷加载约5GB SkyCLIP checkpoint，通常需要几十秒。在线服务应常驻模型与索引，不要为每条查询重新启动进程。

## 9. 测试

```bash
python -m pytest
```

当前12项测试覆盖可变大小 chip、2816维池化、两套 adapter 的形状与梯度、多正样本损失、精排预筛等价性、分模态搜索与去重、空间数据连接、可视化和基础命令入口。

## 10. 已知边界

- 当前只支持英文查询；中文需要额外多语文本塔或翻译层。
- TESSERA chip 是 v1 年度语义表征，不能用于微调 v2 ViT。
- 高分图与 TESSERA 按 bbox和年份对应，并非同日成像。
- 全局检索验证场景/地物语义，不证明逐像素配准或小目标定位能力。
- 高分路线虽然保留8个局部潜层，但冻结 SkyCLIP 的输入仍是224×224；原图中缩小后不足一个14×14 patch的车辆等极小目标仍可能丢失。真正的小目标检索需要多裁片或原生高分辨率视觉塔。
- `both` 模式直接比较共同空间的余弦分数；两种模态的分数分布仍可能有偏移，生产系统可在验证集上做温度校准或 rank fusion。
