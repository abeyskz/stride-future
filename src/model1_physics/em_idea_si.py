"""
EM Idea-SI Estimator
====================
全場芝1200mの走破タイムをコース補正してベイズEMを実行。
各馬の「イデア能力（レースレベル・馬場・コース差を除去した純粋な能力）」を推定する。

観測モデル:
    SI_adj[i,r] = theta_horse[i] + theta_race[r] + eps[i,r]

M-step (MAP推定):
    theta_horse[i] = (sum_r(SI_adj - theta_race) + lambda_h * mu_global) / (n_i + lambda_h)
    theta_race[r]  = sum_i(SI_adj - theta_horse) / (n_r + lambda_r)

Usage:
    python em_idea_si.py
    → /tmp/em_result.pkl にイデアSI辞書を保存
"""

import pandas as pd
import zipfile
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 定数
# ==============================
BAJU_CORRECTION = {'良': 0.0, '稍': 0.791, '重': 0.941, '不': 2.618}
BASE_KINTO  = 57.0
KINTO_FACTOR = 0.1
BASE_TIME   = 70.208   # 中京・芝・1200m・良の全期間平均走破タイム（秒）

COURSE_CORRECTION = {   # 各場所 → 中京換算補正（秒）
    '中京': 0.0,
    '中山': 0.410,
    '京都': 0.085,
    '函館': -0.712,
    '小倉': 0.371,
    '新潟': 0.003,
    '札幌': -0.249,
    '福島': -0.168,
    '阪神': -0.354,
}

LAMBDA_H = 5.0   # 馬の正則化強度
LAMBDA_R = 3.0   # レースの正則化強度
MIN_RUNS = 3     # 学習対象の最小出走数


def load_data(paths):
    dfs = []
    for fname in paths:
        with zipfile.ZipFile(fname) as z:
            with z.open(z.namelist()[0]) as f:
                dfs.append(pd.read_csv(f, encoding='utf-8-sig', low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df['年4']   = df['年'].apply(lambda x: 2000+int(x) if int(x)<=25 else 1900+int(x))
    df['馬名_s'] = df['馬名'].str.strip()
    return df


def calc_si_adj(df):
    """全場芝1200mをコース補正して SI_adj を計算"""
    df_all = df[
        (df['芝・ダ'] == '芝') &
        (df['距離'] == 1200) &
        df['走破タイム'].notna() &
        df['馬場状態'].notna() &
        df['場所'].isin(COURSE_CORRECTION.keys())
    ].copy()

    df_all['baju_corr']   = df_all['馬場状態'].str.strip().map(BAJU_CORRECTION).fillna(0.0)
    df_all['kinto_corr']  = (BASE_KINTO - df_all['斤量'].astype(float)) * KINTO_FACTOR
    df_all['course_corr'] = df_all['場所'].map(COURSE_CORRECTION)
    df_all['SI_adj'] = (BASE_TIME - (
        df_all['走破タイム']
        - df_all['baju_corr']
        - df_all['kinto_corr']
        - df_all['course_corr']
    )) * 10 + 100

    return df_all


def run_bayesian_em(df_em):
    """ベイズEM（正則化付き）でイデアSIを推定"""
    mu_global   = df_em['SI_adj'].mean()
    theta_horse = df_em.groupby('馬名_s')['SI_adj'].mean().to_dict()
    theta_race  = {r: 0.0 for r in df_em['レースID'].unique()}

    for iteration in range(200):
        # E-step: レース補正更新
        df_em['th'] = df_em['馬名_s'].map(theta_horse)
        race_grp = df_em.groupby('レースID')
        new_theta_race = (
            (race_grp['SI_adj'].sum() - race_grp['th'].sum()) /
            (race_grp['SI_adj'].count() + LAMBDA_R)
        ).to_dict()

        # M-step: 馬能力更新（MAP）
        df_em['tr'] = df_em['レースID'].map(new_theta_race)
        horse_grp = df_em.groupby('馬名_s')
        new_theta_horse = (
            (horse_grp['SI_adj'].sum() - horse_grp['tr'].sum() + LAMBDA_H * mu_global) /
            (horse_grp['SI_adj'].count() + LAMBDA_H)
        ).to_dict()

        diff = max(abs(new_theta_horse[h] - theta_horse.get(h, mu_global))
                   for h in new_theta_horse)
        theta_horse = new_theta_horse
        theta_race  = new_theta_race

        if diff < 1e-4:
            print(f"  EM収束: iter={iteration}, diff={diff:.2e}")
            break

    return theta_horse, theta_race, mu_global


def main():
    data_paths = [
        'data/raw/race_full_2001_2011.csv.zip',
        'data/raw/race_full_2012_2025.csv.zip',
    ]
    print("データ読み込み中...")
    df = load_data(data_paths)

    print("SI_adj 計算中...")
    df_all = calc_si_adj(df)
    print(f"  全場芝1200m: {len(df_all):,}レコード, {df_all['馬名_s'].nunique():,}頭")

    # 3走以上の馬に絞る
    counts = df_all.groupby('馬名_s').size()
    valid  = counts[counts >= MIN_RUNS].index
    df_em  = df_all[df_all['馬名_s'].isin(valid)].copy()
    print(f"  {MIN_RUNS}走以上: {len(valid):,}頭, {len(df_em):,}レコード")

    print("ベイズEM 実行中...")
    theta_horse, theta_race, mu_global = run_bayesian_em(df_em)

    idea_si = pd.Series(theta_horse).sort_values(ascending=False)
    print(f"\nイデアSI統計: 平均={idea_si.mean():.2f}, std={idea_si.std():.2f}")
    print(f"最高: {idea_si.index[0]} ({idea_si.iloc[0]:.2f})")

    result = {
        'theta_horse': theta_horse,
        'theta_race':  theta_race,
        'mu_global':   mu_global,
        'valid_all':   list(valid),
    }
    with open('/tmp/em_result.pkl', 'wb') as f:
        pickle.dump(result, f)
    print("\n✅ EM結果保存: /tmp/em_result.pkl")
    return result


if __name__ == '__main__':
    main()
