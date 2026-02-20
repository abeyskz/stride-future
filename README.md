# 🐎 STRIDE FUTURE — AI×競馬予測プロジェクト

> **一歩先の未来を予測する知的実験**

AI・物理モデル・進化計算・東洋思想（五行）を横断して、競馬予測を  
*技術検証 × 記事連載 × 実装開発* の三位一体で実行する。

**ターゲットレース**: 2026年3月29日（日）高松宮記念（G1）

---

## 📐 4つの予測アプローチ

| # | アプローチ | ディレクトリ | 特徴 |
|---|---|---|---|
| 案1 | 物理シミュレーション（走行圧モデル） | `src/model1_physics/` | ホワイトボックス |
| 案2 | コース特性分離 Transformer | `src/model2_transformer/` | 構造化ブラックボックス |
| 案3 | 遺伝的アルゴリズム × 決定木 | `src/model3_ga/` | 創発型ルール探索 |
| 案4 | 五行説（外生特徴量） | `src/model4_gogyo/` | 東洋思想 × 統計 |

## 📊 評価指標

- **Winner-in-Top3**: 予想Top3に1着馬が含まれるか
- **NDCG@3**: Top3のランキング品質（利得: 1着=60, 2着=30, 3着=10）
- ベースライン: 人気順Top3 / ランダムTop3

---

## 📂 ディレクトリ構成

```
stride-future/
  data/
    raw/
      jra-dataset/              # ① JRA Horse Racing Dataset (1986-2021)
      horse-racing-in-japan/    # ② Horse Racing in Japan (2010-2021)
    processed/                  # 前処理済みデータ
  src/
    common/
      features/                 # 共通特徴量エンジニアリング
      evaluation/               # NDCG@3, Winner-in-Top3
      utils/                    # データローダー等
    model1_physics/             # 案1: 走行圧モデル
    model2_transformer/         # 案2: Transformer
    model3_ga/                  # 案3: GA×決定木
    model4_gogyo/               # 案4: 五行
  notebooks/                    # EDA・分析用Notebook
  docs/                         # 設計書
  docker/                       # Docker構成
```

---

## 💾 データセットのセットアップ

CSVファイルはサイズが大きいためGit管理外です。  
以下の手順でローカルに配置してください。

### ① JRA Horse Racing Dataset

1. [Kaggle](https://www.kaggle.com/datasets/takamotoki/jra-horse-racing-dataset) からダウンロード
2. 以下のファイルを `data/raw/jra-dataset/` に配置:
   - `19860105-20210731_race_result.csv`（レース結果）
   - `19860105-20210731_odds.csv`（オッズ）
   - `19860105-20210731_laptime.csv`（ラップタイム）
   - `20020615-20210731_corner_passing_order.csv`（コーナー通過順）

### ② Horse Racing in Japan

1. [Kaggle](https://www.kaggle.com/datasets/ayuser/horse-racing-in-japan) からダウンロード
2. 以下のファイルを `data/raw/horse-racing-in-japan/` に配置:
   - `2010-2021.csv`

---

## 🛠 開発環境

- Python 3.11+
- PyTorch / scikit-learn
- Docker + PostgreSQL
- GPU: RTX 5070 Ti (16GB)

---

## 📝 note連載スケジュール

| # | 公開日 | タイトル |
|---|---|---|
| 起 | 3/16 | 理論編（物理 × Transformer） |
| 承 | 3/23 | 検証編（精度検証と改良） |
| 転 | 3/30 | 創発編（GA × 五行 + 結果速報） |
| 結 | 4/6  | 総括編（全方式の結果分析） |

---

## 📅 開発スケジュール

| 期間 | 内容 |
|---|---|
| 2月下旬 | データ取得・環境構築・プロトタイプ開発 |
| 3月上旬 | 過去データによるバックテスト |
| 3月中旬 | JRA-VAN導入・最終調整 |
| 3/22〜27 | 最終予測 |
| 3/29 | 高松宮記念 本番 |
| 3/30〜4/6 | 結果分析・連載完結 |
