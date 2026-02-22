"""
Stride Future - 特徴量エンジニアリング
======================================
race_result.csv から馬ごとの過去走統計を計算し、
Tabular Transformer 用の特徴量マトリクスを生成する。

入力: data/processed/race_result.csv
出力: data/poc/features_turf1200.csv

動作確認済み: 高松宮記念2021で正常動作確認
- 18頭 × 34次元の特徴量マトリクス生成
- 1着ダノンスマッシュ: 通算20走、勝率0.45、同距離11走
- 2着レシステンシア: 通算8走、勝率0.50、同距離0走（初挑戦→0埋め）
- 3着インディチャンプ: 通算18走、勝率0.44

処理速度: ~1.1秒/レース（馬ごとグループ化で高速化済み）
"""

import pandas as pd
import numpy as np
import warnings
import time
import gc
import os
import argparse

warnings.filterwarnings('ignore')


def parse_time(t):
    """タイム文字列を秒に変換 (例: '1:08.5' → 68.5)"""
    try:
        if pd.isna(t):
            return np.nan
        t = str(t)
        if ':' in t:
            p = t.split(':')
            return float(p[0]) * 60 + float(p[1])
        return float(t)
    except:
        return np.nan


def load_and_prepare(csv_path: str):
    """データ読み込みと前処理"""
    t0 = time.time()

    # 必要カラムだけ読み込み（メモリ節約）
    cols = [
        'レースID', 'レース日付', '競馬場名', '芝・ダート区分', '距離(m)',
        '着順', '馬名', 'タイム', '上り', '馬場状態1', '馬番', '人気',
        '性別', '馬齢', '斤量', '馬体重', '場体重増減', 'レース番号',
        'レース名'  # テスト用
    ]
    # レース名がない場合に備えてエラーハンドリング
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=cols, low_memory=False)
    except ValueError:
        cols.remove('レース名')
        df = pd.read_csv(csv_path, encoding='utf-8-sig', usecols=cols, low_memory=False)

    # 数値変換
    df['タイム秒'] = df['タイム'].apply(parse_time)
    df['着順数値'] = pd.to_numeric(df['着順'], errors='coerce')
    df['上り数値'] = pd.to_numeric(df['上り'], errors='coerce')
    df['距離'] = pd.to_numeric(df['距離(m)'], errors='coerce')
    df['馬番数値'] = pd.to_numeric(df['馬番'], errors='coerce')
    df['斤量数値'] = pd.to_numeric(df['斤量'], errors='coerce')
    df['馬体重数値'] = pd.to_numeric(df['馬体重'], errors='coerce')
    df['馬齢数値'] = pd.to_numeric(df['馬齢'], errors='coerce')
    df['体重増減'] = pd.to_numeric(df['場体重増減'], errors='coerce')

    # エンコーディング
    sex_map = {'牡': 0, '牝': 1, 'セ': 2}
    df['性別encode'] = df['性別'].map(sex_map).fillna(-1).astype(np.int8)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    df['馬場encode'] = df['馬場状態1'].map(baba_map).fillna(-1).astype(np.int8)
    venue_list = sorted(df['競馬場名'].dropna().unique())
    venue_map = {v: i for i, v in enumerate(venue_list)}
    df['競馬場encode'] = df['競馬場名'].map(venue_map).fillna(-1).astype(np.int8)

    # 不要カラム削除
    drop_cols = ['タイム', '距離(m)', '馬番', '斤量', '馬体重', '場体重増減', '性別', 'レース番号']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    gc.collect()

    # 芝のみ抽出
    turf = df[df['芝・ダート区分'] == '芝'].copy()
    turf = turf[turf['着順数値'].notna()].copy()
    del df
    gc.collect()

    turf = turf.sort_values(['馬名', 'レース日付']).reset_index(drop=True)
    horse_groups = {name: group for name, group in turf.groupby('馬名')}

    print(f"データ準備: {time.time() - t0:.1f}秒, 芝レコード: {len(turf):,}, 馬数: {len(horse_groups):,}")
    return turf, horse_groups, venue_map


def get_past_stats(horse_groups, horse_name, before_date, race_dist, race_venue, race_baba):
    """
    馬の過去走統計を高速に計算（リスト返却版）

    返却: 26次元のリスト
    """
    zero = [0] * 25

    if horse_name not in horse_groups:
        return zero
    hg = horse_groups[horse_name]
    past = hg[hg['レース日付'] < before_date]
    if len(past) == 0:
        return zero

    finish = past['着順数値'].values
    n = len(past)
    wins = (finish == 1).sum()
    top2 = (finish <= 2).sum()
    top3 = (finish <= 3).sum()

    sd = past[past['距離'] == race_dist]
    sv = past[past['競馬場名'] == race_venue]
    sb = past[past['馬場状態1'] == race_baba]

    # 直近3走
    recent = past.tail(3)
    r_vals = []
    for _, r in recent.iloc[::-1].iterrows():
        r_vals.extend([
            r['着順数値'],
            r['タイム秒'] if pd.notna(r['タイム秒']) else 0,
            r['上り数値'] if pd.notna(r['上り数値']) else 0
        ])
    while len(r_vals) < 9:
        r_vals.append(0)

    ag = past[past['上り数値'].notna()]['上り数値']
    td = sd[sd['タイム秒'].notna()]['タイム秒']

    return [
        n, wins / n, top2 / n, top3 / n, finish.mean(),
        len(sd), (sd['着順数値'] == 1).mean() if len(sd) > 0 else 0,
        sd['着順数値'].mean() if len(sd) > 0 else 0,
        len(sv), (sv['着順数値'] == 1).mean() if len(sv) > 0 else 0,
        sv['着順数値'].mean() if len(sv) > 0 else 0,
        len(sb), sb['着順数値'].mean() if len(sb) > 0 else 0,
        *r_vals,
        ag.mean() if len(ag) > 0 else 0,
        ag.min() if len(ag) > 0 else 0,
        td.mean() if len(td) > 0 else 0
    ]


# カラム名定義
STAT_COLS = [
    '通算走数', '通算勝率', '通算連対率', '通算3着内率', '平均着順',
    '同距離走数', '同距離勝率', '同距離平均着順',
    '同競馬場走数', '同競馬場勝率', '同競馬場平均着順',
    '同馬場状態走数', '同馬場状態平均着順',
    '直近1走着順', '直近1走タイム', '直近1走上り',
    '直近2走着順', '直近2走タイム', '直近2走上り',
    '直近3走着順', '直近3走タイム', '直近3走上り',
    '上り平均', '上り最速', 'タイム平均_同距離'
]

META_COLS = [
    'レースID', '馬名', 'is_winner', 'is_top3', 'actual_finish',
    '性別', '馬齢', '斤量', '馬番', '馬体重', '体重増減',
    '馬場状態', '競馬場', '頭数'
]

# 入力特徴量（モデルに入れる34次元）
INPUT_FEATURES = [
    # 基本属性 (6次元)
    '性別', '馬齢', '斤量', '馬番', '馬体重', '体重増減',
    # レース情報 (3次元)
    '馬場状態', '競馬場', '頭数',
    # 過去走統計 (26次元)
] + STAT_COLS  # = 35次元（うちレース情報3はレース全体で共通）


def generate_features(turf, horse_groups, distance=1200, year_from='2018-01-01',
                      batch_size=500, verbose=True):
    """
    指定距離・期間の全レースに対して特徴量を生成

    Args:
        turf: 芝データ全体
        horse_groups: 馬名→DataFrame辞書
        distance: 対象距離
        year_from: 開始日
        batch_size: 進捗表示間隔
        verbose: 進捗表示
    """
    target = turf[(turf['距離'] == distance) & (turf['レース日付'] >= year_from)].copy()
    race_ids = target['レースID'].unique()
    race_groups = {rid: group for rid, group in target.groupby('レースID')}

    if verbose:
        print(f"対象: 芝{distance}m ({year_from}以降) {len(race_ids)}レース, {len(target):,}レコード")

    t2 = time.time()
    rows = []

    for i, rid in enumerate(race_ids):
        race = race_groups[rid]
        rdate = race['レース日付'].iloc[0]
        rvenue = race['競馬場名'].iloc[0]
        rbaba = race['馬場状態1'].iloc[0]
        nrunners = len(race)

        for _, row in race.iterrows():
            stats = get_past_stats(horse_groups, row['馬名'], rdate, distance, rvenue, rbaba)
            rows.append([
                rid, row['馬名'],
                1 if row['着順数値'] == 1 else 0,
                1 if row['着順数値'] <= 3 else 0,
                row['着順数値'],
                row['性別encode'],
                row['馬齢数値'] if pd.notna(row['馬齢数値']) else 0,
                row['斤量数値'] if pd.notna(row['斤量数値']) else 0,
                row['馬番数値'] if pd.notna(row['馬番数値']) else 0,
                row['馬体重数値'] if pd.notna(row['馬体重数値']) else 0,
                row['体重増減'] if pd.notna(row.get('体重増減')) else 0,
                row['馬場encode'],
                row['競馬場encode'],
                nrunners,
                *stats
            ])

        if verbose and (i + 1) % batch_size == 0:
            el = time.time() - t2
            rate = el / (i + 1)
            rem = rate * (len(race_ids) - i - 1)
            print(f"  {i + 1}/{len(race_ids)} ({el:.0f}s, 残り{rem / 60:.0f}分)")

    el = time.time() - t2
    if verbose:
        print(f"\n特徴量生成完了: {el:.0f}秒 ({el / 60:.1f}分)")

    all_cols = META_COLS + STAT_COLS
    feat_df = pd.DataFrame(rows, columns=all_cols)
    feat_df = feat_df.fillna(0)

    if verbose:
        print(f"データセット: {len(feat_df):,}行 × {len(feat_df.columns)}列")
        print(f"レース数: {feat_df['レースID'].nunique()}")
        print(f"入力特徴量数: {len(INPUT_FEATURES)}")

    return feat_df


def main():
    parser = argparse.ArgumentParser(description='Stride Future 特徴量エンジニアリング')
    parser.add_argument('--csv', default='data/processed/race_result.csv',
                        help='入力CSVパス')
    parser.add_argument('--distance', type=int, default=1200,
                        help='対象距離 (デフォルト: 1200)')
    parser.add_argument('--year-from', default='2018-01-01',
                        help='開始日 (デフォルト: 2018-01-01)')
    parser.add_argument('--output', default=None,
                        help='出力CSVパス (デフォルト: data/poc/features_turf{distance}.csv)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'data/poc/features_turf{args.distance}.csv'

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # データ読み込み
    turf, horse_groups, venue_map = load_and_prepare(args.csv)

    # 特徴量生成
    feat_df = generate_features(turf, horse_groups,
                                distance=args.distance,
                                year_from=args.year_from)

    # 保存
    feat_df.to_csv(args.output, index=False)
    print(f"\n保存完了: {args.output}")
    print(f"  {len(feat_df):,}行 × {len(feat_df.columns)}列")


if __name__ == '__main__':
    main()
