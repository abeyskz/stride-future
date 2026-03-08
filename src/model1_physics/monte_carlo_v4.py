"""
Monte Carlo Simulator v4 — イデアSI版
=======================================
ベイズEMで推定したイデアSIをもとにモンテカルロシミュレーションを実行。

v1〜v3との違い:
  - v1〜v3: 中京芝1200mのSI平均を直接使用（中京実績なし → FALLBACK）
  - v4: 全場芝1200mをコース補正してEMで能力推定 → ほぼ全馬にイデアSIを付与

Usage:
    # 事前に em_idea_si.py を実行して /tmp/em_result.pkl を生成しておく
    python monte_carlo_v4.py --year 2025 --race 高松宮
"""

import argparse
import pandas as pd
import zipfile
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import japanize_matplotlib
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ==============================
# 定数
# ==============================
N_SIM        = 10000
RANDOM_SEED  = 42
DEFAULT_STD  = 5.0
FALLBACK_STD = 8.0

DATA_PATHS = [
    'data/raw/race_full_2001_2011.csv.zip',
    'data/raw/race_full_2012_2025.csv.zip',
]


def load_em_result(path='/tmp/em_result.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_race_horses(year, race_name_kw):
    """指定年・レース名キーワードで出走馬と確定着順を取得"""
    dfs = []
    for fname in DATA_PATHS:
        with zipfile.ZipFile(fname) as z:
            with z.open(z.namelist()[0]) as f:
                dfs.append(pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                                       usecols=['年', '月', 'レース名', '馬名', '確定着順']))
    df = pd.concat(dfs, ignore_index=True)
    df['年4']   = df['年'].apply(lambda x: 2000+int(x) if int(x)<=25 else 1900+int(x))
    df['馬名_s'] = df['馬名'].str.strip()

    race_df = df[
        (df['年4'] == year) &
        (df['レース名'].str.contains(race_name_kw, na=False))
    ][['馬名_s', '確定着順']].drop_duplicates('馬名_s')

    return race_df['馬名_s'].tolist(), dict(zip(race_df['馬名_s'], race_df['確定着順']))


def build_horse_stats(horses, theta_horse, mu_global):
    """各馬のシミュレーション用パラメータを構築"""
    stats = {}
    for horse in horses:
        if horse in theta_horse:
            stats[horse] = {
                'mean':   theta_horse[horse],
                'std':    DEFAULT_STD,
                'source': 'EM',
            }
        else:
            stats[horse] = {
                'mean':   mu_global,
                'std':    FALLBACK_STD,
                'source': 'FALLBACK',
            }
    return stats


def run_simulation(horses, horse_stats):
    np.random.seed(RANDOM_SEED)
    sim_results = {h: [] for h in horses}
    for _ in range(N_SIM):
        scores = {h: np.random.normal(horse_stats[h]['mean'], horse_stats[h]['std'])
                  for h in horses}
        for rank, (horse, _) in enumerate(
                sorted(scores.items(), key=lambda x: -x[1]), 1):
            sim_results[horse].append(rank)
    return sim_results


def summarize(horses, sim_results, horse_stats, actual_map):
    rows = []
    for horse in horses:
        ranks = np.array(sim_results[horse])
        rows.append({
            '馬名':       horse,
            '優勝確率':   round((ranks == 1).mean() * 100, 1),
            '3着内確率':  round((ranks <= 3).mean() * 100, 1),
            '平均予測着順': round(ranks.mean(), 1),
            '実際':       actual_map.get(horse, '?'),
            'ideaSI':    round(horse_stats[horse]['mean'], 1),
            'source':    horse_stats[horse]['source'],
        })
    return pd.DataFrame(rows).sort_values('優勝確率', ascending=False)


def plot(horses_sorted, sim_results, horse_stats, actual_map, out_path):
    if not PLOT_AVAILABLE:
        print("matplotlib/japanize_matplotlib が未インストール。グラフをスキップ。")
        return

    n = len(horses_sorted)
    cols = 6
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(22, 4.5 * rows_n))
    if rows_n == 1:
        axes = [axes]
    fig.patch.set_facecolor('#0f1117')
    fig.suptitle('モンテカルロ着順分布（イデア馬SI版）',
                 fontsize=14, fontweight='bold', color='white', y=0.99)

    colors = plt.cm.RdYlGn_r(np.linspace(0.05, 0.85, n))

    for idx, horse in enumerate(horses_sorted):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor('#1a1d27')
        ranks = np.array(sim_results[horse])
        actual_rank = actual_map.get(horse)
        src = horse_stats[horse]['source']

        counts, _ = np.histogram(ranks, bins=range(1, n + 2))
        bar_color = colors[idx] if src == 'EM' else '#666677'
        ax.bar(range(1, n + 1), counts / N_SIM * 100,
               color=bar_color, edgecolor='#0f1117', linewidth=0.4, width=0.85)

        if actual_rank:
            ax.axvline(x=actual_rank, color='#00cfff', linewidth=2.0,
                       linestyle='--', label=f'実際 {actual_rank}着')
        ax.axvline(x=np.mean(ranks), color='#ff6b6b', linewidth=1.5, alpha=0.9)

        win  = (ranks == 1).mean() * 100
        top3 = (ranks <= 3).mean() * 100
        si_s = f"ideaSI:{horse_stats[horse]['mean']:.0f}" if src == 'EM' else 'FALLBACK'
        ax.set_title(f"{horse}\n1着:{win:.0f}%  3着内:{top3:.0f}%\n{si_s}",
                     fontsize=7.5, pad=3, color='white' if src == 'EM' else '#888899')
        ax.set_xlim(0.5, n + 0.5)
        ax.set_xlabel('着順', fontsize=6.5, color='#aaaaaa')
        ax.tick_params(labelsize=6, colors='#aaaaaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        if actual_rank:
            ax.legend(fontsize=6, loc='upper right',
                      facecolor='#1a1d27', edgecolor='#444', labelcolor='white')

    # 余ったセルを非表示
    for idx in range(n, rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"✅ グラフ保存: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',  type=int,  default=2025)
    parser.add_argument('--race',  type=str,  default='高松宮')
    parser.add_argument('--em',    type=str,  default='/tmp/em_result.pkl')
    parser.add_argument('--out',   type=str,  default='results/monte_carlo_v4.png')
    args = parser.parse_args()

    print(f"EMロード中: {args.em}")
    em = load_em_result(args.em)
    theta_horse = em['theta_horse']
    mu_global   = em['mu_global']

    print(f"出走馬取得: {args.year}年 {args.race}")
    horses, actual_map = get_race_horses(args.year, args.race)
    print(f"  {len(horses)}頭")

    horse_stats = build_horse_stats(horses, theta_horse, mu_global)
    sim_results = run_simulation(horses, horse_stats)
    df_s = summarize(horses, sim_results, horse_stats, actual_map)

    print("\n=== モンテカルロ v4（イデアSI版）予測結果 ===")
    print(df_s.to_string(index=False))

    import os
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    plot(df_s['馬名'].tolist(), sim_results, horse_stats, actual_map, args.out)
    return df_s


if __name__ == '__main__':
    main()
