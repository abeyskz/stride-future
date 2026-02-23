"""
Stride Future - Tabular Transformer モデル
==========================================
馬の特徴量ベクトル（34次元）をSelf-Attentionで処理し、
レース内の馬間の相対関係を学習するTransformerモデル。

アーキテクチャ:
  入力: [batch, max_horses, 34] (馬ごとの特徴量)
  → Linear(34→64) + LayerNorm
  → TransformerEncoder(2層, 4ヘッド, dim=64)
  → Linear(64→1) → Softmax over horses
  出力: [batch, max_horses] (各馬の勝率予測)

設計思想:
  - 数値ネイティブ: テキスト化による情報損失なし
  - Self-Attention: 「馬と馬の関係」を直接学習
  - レース単位のSoftmax: 「このレース内で誰が強いか」を学習
  - パディングマスク: 頭数が異なるレースに対応
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import time
import os
import json

MAX_HORSES = 18  # 最大出走頭数

# カラム定義（01_feature_engineering.py と同期）
STAT_COLS = [
    '通算走数', '通算勝率', '通算連対率', '通算3着内率', '平均着順',
    '同距離走数', '同距離勝率', '同距離平均着順',
    '同競馬場走数', '同競馬場勝率', '同競馬場平均着順',
    '同馬場状態走数', '同馬場状態平均着順',
    '直近1走着順', '直近1走タイム', '直近1走上り',
    '直近2走着順', '直近2走タイム', '直近2走上り',
    '直近3走着順', '直近3走タイム', '直近3走上り',
    '上り平均', '上り最速', 'タイム平均_同距離',
    '平均4角位置', '直近1走4角', '直近2走4角', '直近3走4角'
]

INPUT_FEATURES = [
    '性別', '馬齢', '斤量', '馬番', '馬体重', '体重増減',
    '人気',
    '馬場状態', '競馬場', '頭数',
] + STAT_COLS


# ============================================
# Dataset
# ============================================
class RaceDataset(Dataset):
    """
    レース単位のデータセット
    各サンプル = 1レース = (特徴量マトリクス, ターゲット, マスク)
    """

    def __init__(self, feat_df, max_horses=MAX_HORSES, target='is_winner'):
        self.max_horses = max_horses
        self.target = target

        # 入力特徴量カラム
        self.feature_cols = INPUT_FEATURES

        # レースIDでグループ化
        self.race_ids = feat_df['レースID'].unique()
        self.race_groups = {rid: group for rid, group in feat_df.groupby('レースID')}

        # 特徴量の正規化パラメータを計算
        self.feat_mean = feat_df[self.feature_cols].mean().values.astype(np.float32)
        self.feat_std = feat_df[self.feature_cols].std().values.astype(np.float32)
        self.feat_std[self.feat_std == 0] = 1.0  # ゼロ除算回避

    def __len__(self):
        return len(self.race_ids)

    def __getitem__(self, idx):
        rid = self.race_ids[idx]
        race = self.race_groups[rid]

        n_horses = min(len(race), self.max_horses)

        # 特徴量マトリクス [max_horses, n_features]
        features = np.zeros((self.max_horses, len(self.feature_cols)), dtype=np.float32)
        raw = race[self.feature_cols].values[:n_horses].astype(np.float32)
        # 正規化
        raw = (raw - self.feat_mean) / self.feat_std
        features[:n_horses] = raw

        # ターゲット [max_horses]
        targets = np.zeros(self.max_horses, dtype=np.float32)
        targets[:n_horses] = race[self.target].values[:n_horses].astype(np.float32)

        # パディングマスク [max_horses] (True = パディング = 無視)
        mask = np.ones(self.max_horses, dtype=bool)
        mask[:n_horses] = False

        return (
            torch.tensor(features),
            torch.tensor(targets),
            torch.tensor(mask),
            rid
        )

    def get_norm_params(self):
        """正規化パラメータを返す（推論時に必要）"""
        return {
            'mean': self.feat_mean.tolist(),
            'std': self.feat_std.tolist(),
            'feature_cols': self.feature_cols
        }


# ============================================
# Model
# ============================================
class TabularTransformer(nn.Module):
    """
    Tabular Transformer for Horse Racing Prediction

    各馬の特徴量ベクトルをトークンとして扱い、
    Self-Attentionで馬間の相対関係を学習する。
    """

    def __init__(self, n_features, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        # 特徴量 → 埋め込み
        self.input_proj = nn.Linear(n_features, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 出力ヘッド: 各馬のスコア
        self.output_head = nn.Linear(d_model, 1)

        self.d_model = d_model

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, max_horses, n_features]
            mask: [batch, max_horses] (True = padding)
        Returns:
            scores: [batch, max_horses] (softmax済み確率)
        """
        # 特徴量射影
        h = self.input_proj(x)  # [batch, max_horses, d_model]
        h = self.layer_norm(h)

        # Transformer (Self-Attention)
        h = self.transformer(h, src_key_padding_mask=mask)  # [batch, max_horses, d_model]

        # 各馬のスコア
        scores = self.output_head(h).squeeze(-1)  # [batch, max_horses]

        # パディング位置を-infにしてSoftmax
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        probs = F.softmax(scores, dim=-1)
        return probs


# ============================================
# Training
# ============================================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for features, targets, mask, _ in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        probs = model(features, mask)

        # Cross-entropy: ターゲットは1着馬のインデックス
        # → targets内の1の位置がラベル
        # KLDivergence or simple CE
        loss = -torch.sum(targets * torch.log(probs + 1e-8)) / targets.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, device):
    """勝ち馬予測の正解率（Top-1/Top-3）を計算"""
    model.eval()
    total_loss = 0
    n_batches = 0
    correct_top1 = 0
    correct_top3 = 0
    total = 0

    with torch.no_grad():
        for features, targets, mask, _ in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            probs = model(features, mask)

            loss = -torch.sum(targets * torch.log(probs + 1e-8)) / targets.sum()
            total_loss += loss.item()
            n_batches += 1

            # Top-1: 予測1位が実際の1着か
            pred_rank = probs.argsort(dim=-1, descending=True)
            actual_winner = targets.argmax(dim=-1)

            correct_top1 += (pred_rank[:, 0] == actual_winner).sum().item()

            # Top-3: 実際の1着が予測トップ3に入っているか
            for b in range(len(targets)):
                if actual_winner[b] in pred_rank[b, :3]:
                    correct_top3 += 1
            total += len(targets)

    return {
        'loss': total_loss / n_batches,
        'top1_acc': correct_top1 / total,
        'top3_acc': correct_top3 / total,
        'n_races': total
    }


def main():
    parser = argparse.ArgumentParser(description='Stride Future Tabular Transformer 学習')
    parser.add_argument('--features', default='data/poc/features_turf1200.csv',
                        help='特徴量CSVパス')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='テストデータ割合（時系列分割）')
    parser.add_argument('--output-dir', default='models/poc',
                        help='モデル保存ディレクトリ')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # データ読み込み
    print(f"\n特徴量読み込み: {args.features}")
    feat_df = pd.read_csv(args.features)
    print(f"  {len(feat_df):,}行 × {len(feat_df.columns)}列")
    print(f"  レース数: {feat_df['レースID'].nunique()}")

    # 時系列分割（未来のレースでテスト）
    race_ids = feat_df.groupby('レースID').first().reset_index()
    # レースIDにはレース日付情報が含まれているのでソートで時系列順になる
    race_ids_sorted = sorted(feat_df['レースID'].unique())
    split_idx = int(len(race_ids_sorted) * (1 - args.test_ratio))
    train_ids = set(race_ids_sorted[:split_idx])
    test_ids = set(race_ids_sorted[split_idx:])

    train_df = feat_df[feat_df['レースID'].isin(train_ids)]
    test_df = feat_df[feat_df['レースID'].isin(test_ids)]
    print(f"\n  Train: {train_df['レースID'].nunique()}レース, {len(train_df):,}行")
    print(f"  Test:  {test_df['レースID'].nunique()}レース, {len(test_df):,}行")

    # Dataset & DataLoader
    train_dataset = RaceDataset(train_df)
    test_dataset = RaceDataset(test_df)
    # テストデータにも学習データの正規化パラメータを適用
    test_dataset.feat_mean = train_dataset.feat_mean
    test_dataset.feat_std = train_dataset.feat_std

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # モデル
    n_features = len(INPUT_FEATURES)
    model = TabularTransformer(
        n_features=n_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nモデル: {param_count:,} パラメータ")
    print(f"  入力: {n_features}次元 → d_model={args.d_model}")
    print(f"  Transformer: {args.num_layers}層, {args.nhead}ヘッド")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 学習ループ
    print(f"\n{'=' * 60}")
    print(f"学習開始 (Epochs: {args.epochs})")
    print(f"{'=' * 60}")

    best_top3 = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_metrics = evaluate(model, test_loader, device)
        scheduler.step()

        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_loss': test_metrics['loss'],
            'top1_acc': test_metrics['top1_acc'],
            'top3_acc': test_metrics['top3_acc']
        })

        # ベスト更新
        if test_metrics['top3_acc'] > best_top3:
            best_top3 = test_metrics['top3_acc']
            best_marker = ' ★'
            # モデル保存
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            best_marker = ''

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f}/{test_metrics['loss']:.4f} | "
            f"Top1: {test_metrics['top1_acc']:.3f} | "
            f"Top3: {test_metrics['top3_acc']:.3f}{best_marker} | "
            f"{elapsed:.1f}s"
        )

    # 結果サマリ
    print(f"\n{'=' * 60}")
    print(f"学習完了!")
    print(f"  Best Top3 Accuracy: {best_top3:.3f}")
    print(f"  ランダムベースライン: ~{3 / 14:.3f} (14頭立て想定)")
    print(f"  人気順ベースライン: ~0.65 (参考値)")
    print(f"{'=' * 60}")

    # 正規化パラメータとハイパラを保存
    os.makedirs(args.output_dir, exist_ok=True)
    meta = {
        'norm_params': train_dataset.get_norm_params(),
        'model_config': {
            'n_features': n_features,
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dropout': args.dropout,
            'max_horses': MAX_HORSES
        },
        'training': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'best_top3_acc': best_top3,
            'train_races': len(train_ids),
            'test_races': len(test_ids)
        },
        'history': history
    }
    with open(os.path.join(args.output_dir, 'training_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n保存先: {args.output_dir}/")
    print(f"  best_model.pt       - ベストモデル重み")
    print(f"  training_meta.json  - 正規化パラメータ・ハイパラ・学習履歴")


if __name__ == '__main__':
    main()
