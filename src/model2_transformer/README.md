# Stride Future - Tabular Transformer PoC

競馬予測 Tabular Transformer の PoC 実装。
馬ごとの過去走統計（34次元）を Self-Attention で処理し、レース内の着順を予測する。

## アーキテクチャ

```
入力: [batch, max_horses=18, 34特徴量]
  → Linear(34→64) + LayerNorm
  → TransformerEncoder(2層, 4ヘッド, d_model=64)
  → Linear(64→1) → Softmax over horses
出力: [batch, max_horses] (各馬の勝率予測)
```

## 特徴量（34次元）

### 馬の基本属性（6次元）
| # | 特徴量 | 説明 |
|---|--------|------|
| 1 | 性別 | 牡=0, 牝=1, セ=2 |
| 2 | 馬齢 | 年齢 |
| 3 | 斤量 | 負担重量(kg) |
| 4 | 馬番 | 枠順 |
| 5 | 馬体重 | 体重(kg) |
| 6 | 体重増減 | 前走比(kg) |

### レース情報（3次元）
| # | 特徴量 | 説明 |
|---|--------|------|
| 7 | 馬場状態 | 良=0, 稍重=1, 重=2, 不良=3 |
| 8 | 競馬場 | 各競馬場のID |
| 9 | 頭数 | 出走頭数 |

### 過去走統計（26次元）
| # | 特徴量 | 説明 |
|---|--------|------|
| 10-14 | 通算成績 | 走数, 勝率, 連対率, 3着内率, 平均着順 |
| 15-17 | 同距離成績 | 走数, 勝率, 平均着順 |
| 18-20 | 同競馬場成績 | 走数, 勝率, 平均着順 |
| 21-22 | 同馬場状態成績 | 走数, 平均着順 |
| 23-31 | 直近3走 | 着順×3, タイム×3, 上り×3 |
| 32-33 | 上り統計 | 平均, 最速 |
| 34 | 同距離タイム平均 | - |

※ 新馬・初条件は全て0埋め（正しく表現される）

## セットアップ

```bash
pip install torch pandas numpy
```

## 実行手順

### Step 1: 特徴量生成

```bash
# 芝1200m, 2018年以降（推奨。全量だとメモリ注意）
python 01_feature_engineering.py \
  --csv data/processed/race_result.csv \
  --distance 1200 \
  --year-from 2018-01-01 \
  --output data/poc/features_turf1200.csv

# 他の距離も可能
python 01_feature_engineering.py --distance 1600 --year-from 2018-01-01
python 01_feature_engineering.py --distance 2000 --year-from 2018-01-01
```

**処理時間目安**: 芝1200m全量で約25分（1.1秒/レース × ~1,400レース）
**メモリ**: 必要カラムのみ読み込み。8GB以上推奨。

### Step 2: モデル学習

```bash
python 02_tabular_transformer.py \
  --features data/poc/features_turf1200.csv \
  --epochs 30 \
  --batch-size 32 \
  --lr 0.001 \
  --d-model 64 \
  --nhead 4 \
  --num-layers 2 \
  --output-dir models/poc
```

**出力ファイル**:
- `models/poc/best_model.pt` - ベストモデル重み
- `models/poc/training_meta.json` - 正規化パラメータ・学習履歴

### Step 3: 結果確認

学習中にTop-1/Top-3正解率が表示される。

**ベースライン**:
- ランダム: Top1 ~7%, Top3 ~21% (14頭立て)
- 人気順: Top1 ~30%, Top3 ~65%

## ディレクトリ構成

```
stride-future/
├── data/
│   ├── processed/
│   │   └── race_result.csv    ← 元データ（要配置）
│   └── poc/
│       └── features_turf1200.csv  ← Step1で生成
├── models/
│   └── poc/
│       ├── best_model.pt
│       └── training_meta.json
├── 01_feature_engineering.py
├── 02_tabular_transformer.py
└── README.md
```

## 設計メモ

### なぜ Tabular Transformer か
- **情報損失なし**: LLM言語化と違い、数値をそのまま使える
- **Self-Attention**: 馬間の相対関係（「この馬は他の馬より速い」）を直接学習
- **軽量**: パラメータ数万程度。GPUなしでも学習可能
- **Plan B保険**: 精度出なかったらLLM fine-tuneに切り替え

### 確認済みの動作
- 高松宮記念2021で特徴量生成テスト成功
  - ダノンスマッシュ: 通算20走, 勝率0.45, 同距離11走
  - レシステンシア: 通算8走, 勝率0.50, 同距離0走（初挑戦→0埋め正常動作）
- 処理速度: 100レース / 111秒

### 今後の拡張
- [ ] 距離別モデル統合
- [ ] オッズ特徴量の追加
- [ ] コーナー通過順の追加
- [ ] Attention重みの可視化（どの馬に注目しているか）
- [ ] 推論スクリプト作成
