"""
STRIDE FUTURE - 案3: 遺伝的プログラミング × 決定木
===================================================
特徴量の組み合わせルールを「進化」させ、1着馬を見抜く決定木を自動生成する。

アプローチ:
  1. ランダムな決定木（深さ2-4）を大量生成（個体群）
  2. 芝1200mの過去レースでWinner-in-Top3を評価（適応度）
  3. トーナメント選択 + 交叉 + 突然変異で世代交代
  4. 数百世代の進化で「創発的なルール」を発見する

Usage:
    python gp_decision_tree.py
"""

import pandas as pd
import numpy as np
import random
import copy
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 定数（あべちゃんの環境でチューニング可能）
# ==============================
POP_SIZE = 200       # 個体群サイズ（50〜500。大きいほど多様性が保たれるが遅くなる）
N_GENERATIONS = 300  # 最大世代数（収束したら早期終了する）
MAX_DEPTH = 4        # 決定木の最大深さ（3〜5。深すぎると過学習）
TOURNAMENT_K = 5     # トーナメントサイズ（3〜7。大きいほど選択圧が強い）
CROSSOVER_RATE = 0.7 # 交叉率
MUTATION_RATE = 0.2  # 突然変異率
ELITE_SIZE = 10      # エリート保存数
RANDOM_SEED = 42
DATA_RATIO = 1.0     # 使用するデータの割合（0.3=最新30%のみ, 1.0=全データ）

# 収束判定パラメータ
STAGNATION_LIMIT = 20   # best_everがN世代連続で更新されなければ収束と判断
OVERFIT_THRESHOLD = 0.15 # train_w3とtest_w3の差がこれ以上開いたら過学習警告

# 使用する特徴量
FEATURE_COLS = [
    '馬齢', '斤量', '馬番', '馬体重', '体重増減', '人気',
    '通算走数', '通算勝率', '通算連対率', '通算3着内率', '平均着順',
    '同距離走数', '同距離勝率', '同距離平均着順',
    '同競馬場走数', '同競馬場勝率', '同競馬場平均着順',
    '直近1走着順', '直近1走上り',
    '直近2走着順', '直近2走上り',
    '直近3走着順', '直近3走上り',
    '上り平均', '上り最速', 'タイム平均_同距離',
    '平均4角位置', '直近1走4角',
]


# ==============================
# 決定木ノード
# ==============================
class Node:
    """決定木の1ノード"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, score=None):
        self.feature = feature      # 分岐特徴量名
        self.threshold = threshold  # 分岐閾値
        self.left = left            # True(<=)の子
        self.right = right          # False(>)の子
        self.score = score          # 葉ノードのスコア

    def is_leaf(self):
        return self.score is not None

    def predict_one(self, row):
        if self.is_leaf():
            return self.score
        val = row.get(self.feature, 0)
        if val <= self.threshold:
            return self.left.predict_one(row) if self.left else 0
        else:
            return self.right.predict_one(row) if self.right else 0

    def depth(self):
        if self.is_leaf():
            return 0
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        return 1 + max(ld, rd)

    def to_rule_str(self, indent=0):
        prefix = "  " * indent
        if self.is_leaf():
            return f"{prefix}→ score={self.score:.2f}"
        s = f"{prefix}if {self.feature} <= {self.threshold:.2f}:\n"
        s += self.left.to_rule_str(indent+1) + "\n"
        s += f"{prefix}else:\n"
        s += self.right.to_rule_str(indent+1)
        return s


# ==============================
# GP操作
# ==============================
def random_tree(depth, feature_stats):
    """ランダムな決定木を生成"""
    if depth <= 0 or random.random() < 0.3:
        return Node(score=random.uniform(0, 1))

    feat = random.choice(FEATURE_COLS)
    stats = feature_stats.get(feat, {'min': 0, 'max': 1, 'mean': 0.5, 'std': 0.5})
    threshold = random.gauss(stats['mean'], stats['std'])

    return Node(
        feature=feat,
        threshold=threshold,
        left=random_tree(depth - 1, feature_stats),
        right=random_tree(depth - 1, feature_stats),
    )


def crossover(parent1, parent2):
    """交叉: 2つの親からサブツリーを交換"""
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # ランダムなノードを選んで交換
    nodes1 = collect_nodes(child1)
    nodes2 = collect_nodes(child2)

    if len(nodes1) > 1 and len(nodes2) > 1:
        n1 = random.choice(nodes1[1:])  # rootは除く
        n2 = random.choice(nodes2[1:])
        # n1の内容をn2で置き換え
        n1.feature = n2.feature
        n1.threshold = n2.threshold
        n1.left = copy.deepcopy(n2.left)
        n1.right = copy.deepcopy(n2.right)
        n1.score = n2.score

    return child1


def mutate(tree, feature_stats):
    """突然変異: ランダムなノードを変更"""
    tree = copy.deepcopy(tree)
    nodes = collect_nodes(tree)
    if not nodes:
        return tree

    node = random.choice(nodes)
    if node.is_leaf():
        node.score = random.uniform(0, 1)
    else:
        if random.random() < 0.5:
            node.feature = random.choice(FEATURE_COLS)
            stats = feature_stats.get(node.feature, {'mean': 0.5, 'std': 0.5})
            node.threshold = random.gauss(stats['mean'], stats['std'])
        else:
            # サブツリーを新しいランダムツリーに置換
            new_sub = random_tree(2, feature_stats)
            node.feature = new_sub.feature
            node.threshold = new_sub.threshold
            node.left = new_sub.left
            node.right = new_sub.right
            node.score = new_sub.score

    return tree


def collect_nodes(tree):
    """ツリーの全ノードをリストで返す"""
    if tree is None:
        return []
    nodes = [tree]
    if not tree.is_leaf():
        nodes.extend(collect_nodes(tree.left))
        nodes.extend(collect_nodes(tree.right))
    return nodes


# ==============================
# 評価
# ==============================
def evaluate(tree, race_groups):
    """
    Winner-in-Top3: 各レースでスコア上位3頭に1着馬が含まれるか
    """
    correct = 0
    total = 0

    for rid, group in race_groups.items():
        scores = []
        winner_idx = None
        for i, (_, row) in enumerate(group.iterrows()):
            s = tree.predict_one(row)
            scores.append((i, s))
            if row['is_winner'] == 1:
                winner_idx = i

        if winner_idx is None:
            continue

        scores.sort(key=lambda x: -x[1])
        top3_indices = [s[0] for s in scores[:3]]

        if winner_idx in top3_indices:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0


# ==============================
# メイン
# ==============================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=== STRIDE FUTURE 案3: GP × 決定木 ===\n")

    # データ読み込み
    print("データ読み込み中...")
    df = pd.read_csv('data/poc/features_turf1200.csv')
    print(f"  全データ: {len(df)}行, レース数: {df['レースID'].nunique()}")

    # 欠損値を0埋め
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # データ範囲の制限（DATA_RATIO < 1.0 の場合、最新N%のみ使用）
    race_ids = sorted(df['レースID'].unique())
    if DATA_RATIO < 1.0:
        start_idx = int(len(race_ids) * (1.0 - DATA_RATIO))
        race_ids = race_ids[start_idx:]
        df = df[df['レースID'].isin(race_ids)]
        print(f"  データ範囲制限: 最新{DATA_RATIO*100:.0f}% → {len(race_ids)}レース")

    # Train/Test分割（時系列でレースIDでソート）
    split_idx = int(len(race_ids) * 0.8)
    train_ids = set(race_ids[:split_idx])
    test_ids = set(race_ids[split_idx:])

    train_df = df[df['レースID'].isin(train_ids)]
    test_df = df[df['レースID'].isin(test_ids)]

    train_groups = {rid: group for rid, group in train_df.groupby('レースID')}
    test_groups = {rid: group for rid, group in test_df.groupby('レースID')}

    print(f"  学習: {len(train_groups)}レース, テスト: {len(test_groups)}レース")

    # 特徴量の統計量（ランダム木生成用）
    feature_stats = {}
    for col in FEATURE_COLS:
        feature_stats[col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': max(df[col].std(), 0.01),
        }

    # 初期個体群の生成
    print(f"\n初期個体群 ({POP_SIZE}個) 生成中...")
    population = [random_tree(random.randint(2, MAX_DEPTH), feature_stats) for _ in range(POP_SIZE)]

    # 進化ループ
    print(f"進化開始 (最大{N_GENERATIONS}世代, 収束判定: {STAGNATION_LIMIT}世代停滞で終了)...\n")
    best_ever = None
    best_ever_fitness = 0
    stagnation_count = 0
    history = []

    t_start = time.time()

    for gen in range(N_GENERATIONS):
        # 評価
        fitnesses = [evaluate(ind, train_groups) for ind in population]

        # 最良個体の記録
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        avg_fitness = np.mean(fitnesses)

        if gen_best_fitness > best_ever_fitness:
            best_ever = copy.deepcopy(population[gen_best_idx])
            best_ever_fitness = gen_best_fitness
            stagnation_count = 0
        else:
            stagnation_count += 1

        if gen % 50 == 0 or gen == N_GENERATIONS - 1 or stagnation_count == STAGNATION_LIMIT:
            elapsed = time.time() - t_start
            # 定期的にテスト評価して過学習チェック
            test_fitness_check = evaluate(best_ever, test_groups)
            overfit_gap = best_ever_fitness - test_fitness_check
            overfit_warn = "⚠️ 過学習の兆候" if overfit_gap > OVERFIT_THRESHOLD else ""
            print(f"  世代 {gen:3d}: 最良={gen_best_fitness:.4f}  平均={avg_fitness:.4f}  "
                  f"歴代最良={best_ever_fitness:.4f}  テスト={test_fitness_check:.4f}  "
                  f"停滞={stagnation_count}/{STAGNATION_LIMIT}  経過={elapsed:.1f}s {overfit_warn}")

        history.append({
            'generation': gen,
            'best': round(gen_best_fitness, 4),
            'avg': round(avg_fitness, 4),
            'best_ever': round(best_ever_fitness, 4),
        })

        # 収束判定: best_everがN世代連続で更新されなければ終了
        if stagnation_count >= STAGNATION_LIMIT:
            print(f"\n  ✅ 収束判定: {STAGNATION_LIMIT}世代連続でbest_everが更新されず → 進化終了 (世代{gen})")
            break

        # エリート保存
        elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
        elites = [copy.deepcopy(population[i]) for i in elite_indices]

        # 次世代の生成
        new_pop = list(elites)

        while len(new_pop) < POP_SIZE:
            # トーナメント選択
            tournament = random.sample(range(len(population)), TOURNAMENT_K)
            parent1_idx = max(tournament, key=lambda i: fitnesses[i])

            if random.random() < CROSSOVER_RATE:
                tournament2 = random.sample(range(len(population)), TOURNAMENT_K)
                parent2_idx = max(tournament2, key=lambda i: fitnesses[i])
                child = crossover(population[parent1_idx], population[parent2_idx])
            else:
                child = copy.deepcopy(population[parent1_idx])

            if random.random() < MUTATION_RATE:
                child = mutate(child, feature_stats)

            # 深さ制限
            if child.depth() <= MAX_DEPTH:
                new_pop.append(child)

        population = new_pop[:POP_SIZE]

    total_time = time.time() - t_start

    # テスト評価
    print(f"\n=== 進化完了 ({total_time:.1f}秒) ===\n")
    test_fitness = evaluate(best_ever, test_groups)
    print(f"歴代最良個体:")
    print(f"  学習データ W@3: {best_ever_fitness:.4f} ({best_ever_fitness*100:.1f}%)")
    print(f"  テストデータ W@3: {test_fitness:.4f} ({test_fitness*100:.1f}%)")
    print(f"  決定木の深さ: {best_ever.depth()}")
    print(f"\n進化したルール:")
    print(best_ever.to_rule_str())

    # 2026年高松宮記念への適用
    print("\n\n=== 2026年 高松宮記念 予測 ===\n")

    horses_2026 = {
        'パンジャタワー':     {'馬齢': 4, '斤量': 58, '馬番': 1, '馬体重': 482, '体重増減': 0, '人気': 8, '通算走数': 5, '通算勝率': 0.20, '通算連対率': 0.40, '通算3着内率': 0.60, '平均着順': 3.0, '同距離走数': 3, '同距離勝率': 0.33, '同距離平均着順': 2.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 1, '直近1走上り': 33.2, '直近2走着順': 3, '直近2走上り': 33.8, '直近3走着順': 2, '直近3走上り': 33.5, '上り平均': 33.5, '上り最速': 33.2, 'タイム平均_同距離': 68.8, '平均4角位置': 4, '直近1走4角': 4},
        'ビッグシーザー':     {'馬齢': 6, '斤量': 58, '馬番': 2, '馬体重': 530, '体重増減': 0, '人気': 10, '通算走数': 20, '通算勝率': 0.15, '通算連対率': 0.25, '通算3着内率': 0.35, '平均着順': 5.5, '同距離走数': 8, '同距離勝率': 0.13, '同距離平均着順': 5.0, '同競馬場走数': 2, '同競馬場勝率': 0.0, '同競馬場平均着順': 6.0, '直近1走着順': 5, '直近1走上り': 33.5, '直近2走着順': 8, '直近2走上り': 34.0, '直近3走着順': 3, '直近3走上り': 33.8, '上り平均': 33.8, '上り最速': 33.5, 'タイム平均_同距離': 68.0, '平均4角位置': 6, '直近1走4角': 7},
        'エーティーマクフィ':   {'馬齢': 7, '斤量': 58, '馬番': 3, '馬体重': 488, '体重増減': 0, '人気': 14, '通算走数': 30, '通算勝率': 0.10, '通算連対率': 0.17, '通算3着内率': 0.27, '平均着順': 6.0, '同距離走数': 10, '同距離勝率': 0.10, '同距離平均着順': 5.5, '同競馬場走数': 1, '同競馬場勝率': 0.0, '同競馬場平均着順': 7.0, '直近1走着順': 4, '直近1走上り': 34.0, '直近2走着順': 6, '直近2走上り': 34.2, '直近3走着順': 5, '直近3走上り': 34.5, '上り平均': 34.2, '上り最速': 34.0, 'タイム平均_同距離': 68.5, '平均4角位置': 5, '直近1走4角': 5},
        'ダノンマッキンリー':   {'馬齢': 5, '斤量': 58, '馬番': 4, '馬体重': 508, '体重増減': 0, '人気': 6, '通算走数': 15, '通算勝率': 0.13, '通算連対率': 0.27, '通算3着内率': 0.40, '平均着順': 4.5, '同距離走数': 5, '同距離勝率': 0.20, '同距離平均着順': 4.0, '同競馬場走数': 1, '同競馬場勝率': 0.0, '同競馬場平均着順': 5.0, '直近1走着順': 3, '直近1走上り': 33.0, '直近2走着順': 5, '直近2走上り': 33.5, '直近3走着順': 4, '直近3走上り': 33.8, '上り平均': 33.4, '上り最速': 33.0, 'タイム平均_同距離': 67.8, '平均4角位置': 5, '直近1走4角': 5},
        'ヤマニンアルリフラ':   {'馬齢': 5, '斤量': 58, '馬番': 5, '馬体重': 476, '体重増減': 0, '人気': 12, '通算走数': 12, '通算勝率': 0.08, '通算連対率': 0.17, '通算3着内率': 0.25, '平均着順': 5.8, '同距離走数': 4, '同距離勝率': 0.0, '同距離平均着順': 6.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 6, '直近1走上り': 33.8, '直近2走着順': 4, '直近2走上り': 34.0, '直近3走着順': 7, '直近3走上り': 34.2, '上り平均': 34.0, '上り最速': 33.8, 'タイム平均_同距離': 68.5, '平均4角位置': 6, '直近1走4角': 6},
        'レッドモンレーヴ':    {'馬齢': 7, '斤量': 58, '馬番': 6, '馬体重': 502, '体重増減': 0, '人気': 15, '通算走数': 25, '通算勝率': 0.08, '通算連対率': 0.16, '通算3着内率': 0.24, '平均着順': 6.5, '同距離走数': 6, '同距離勝率': 0.0, '同距離平均着順': 7.0, '同競馬場走数': 2, '同競馬場勝率': 0.0, '同競馬場平均着順': 7.0, '直近1走着順': 8, '直近1走上り': 34.2, '直近2走着順': 7, '直近2走上り': 33.8, '直近3走着順': 5, '直近3走上り': 34.5, '上り平均': 34.2, '上り最速': 33.8, 'タイム平均_同距離': 69.0, '平均4角位置': 7, '直近1走4角': 8},
        'ヨシノイースター':    {'馬齢': 8, '斤量': 58, '馬番': 7, '馬体重': 480, '体重増減': 0, '人気': 9, '通算走数': 35, '通算勝率': 0.11, '通算連対率': 0.20, '通算3着内率': 0.31, '平均着順': 5.2, '同距離走数': 12, '同距離勝率': 0.17, '同距離平均着順': 4.5, '同競馬場走数': 4, '同競馬場勝率': 0.25, '同競馬場平均着順': 3.5, '直近1走着順': 3, '直近1走上り': 33.0, '直近2走着順': 2, '直近2走上り': 33.2, '直近3走着順': 4, '直近3走上り': 33.5, '上り平均': 33.2, '上り最速': 33.0, 'タイム平均_同距離': 67.5, '平均4角位置': 4, '直近1走4角': 3},
        'ウインカーネリアン':   {'馬齢': 9, '斤量': 58, '馬番': 8, '馬体重': 510, '体重増減': 0, '人気': 4, '通算走数': 40, '通算勝率': 0.13, '通算連対率': 0.23, '通算3着内率': 0.33, '平均着順': 5.0, '同距離走数': 15, '同距離勝率': 0.13, '同距離平均着順': 4.8, '同競馬場走数': 3, '同競馬場勝率': 0.33, '同競馬場平均着順': 3.0, '直近1走着順': 2, '直近1走上り': 33.2, '直近2走着順': 4, '直近2走上り': 33.5, '直近3走着順': 3, '直近3走上り': 33.3, '上り平均': 33.3, '上り最速': 33.0, 'タイム平均_同距離': 67.5, '平均4角位置': 5, '直近1走4角': 4},
        'サトノレーヴ':      {'馬齢': 7, '斤量': 58, '馬番': 9, '馬体重': 520, '体重増減': 0, '人気': 1, '通算走数': 15, '通算勝率': 0.27, '通算連対率': 0.40, '通算3着内率': 0.53, '平均着順': 3.5, '同距離走数': 6, '同距離勝率': 0.33, '同距離平均着順': 2.5, '同競馬場走数': 2, '同競馬場勝率': 0.50, '同競馬場平均着順': 1.5, '直近1走着順': 1, '直近1走上り': 32.8, '直近2走着順': 2, '直近2走上り': 33.0, '直近3走着順': 1, '直近3走上り': 33.2, '上り平均': 33.0, '上り最速': 32.8, 'タイム平均_同距離': 67.0, '平均4角位置': 3, '直近1走4角': 2},
        'ママコチャ':       {'馬齢': 7, '斤量': 56, '馬番': 10, '馬体重': 470, '体重増減': 0, '人気': 3, '通算走数': 20, '通算勝率': 0.20, '通算連対率': 0.35, '通算3着内率': 0.45, '平均着順': 4.0, '同距離走数': 8, '同距離勝率': 0.25, '同距離平均着順': 3.5, '同競馬場走数': 2, '同競馬場勝率': 0.50, '同競馬場平均着順': 2.0, '直近1走着順': 2, '直近1走上り': 33.0, '直近2走着順': 3, '直近2走上り': 33.2, '直近3走着順': 2, '直近3走上り': 33.5, '上り平均': 33.2, '上り最速': 33.0, 'タイム平均_同距離': 67.5, '平均4角位置': 4, '直近1走4角': 3},
        'ララマセラシオン':    {'馬齢': 5, '斤量': 58, '馬番': 11, '馬体重': 490, '体重増減': 0, '人気': 16, '通算走数': 8, '通算勝率': 0.0, '通算連対率': 0.13, '通算3着内率': 0.25, '平均着順': 6.0, '同距離走数': 2, '同距離勝率': 0.0, '同距離平均着順': 6.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 5, '直近1走上り': 34.0, '直近2走着順': 7, '直近2走上り': 34.5, '直近3走着順': 6, '直近3走上り': 33.8, '上り平均': 34.1, '上り最速': 33.8, 'タイム平均_同距離': 68.5, '平均4角位置': 6, '直近1走4角': 5},
        'ピューロマジック':    {'馬齢': 5, '斤量': 56, '馬番': 12, '馬体重': 458, '体重増減': 0, '人気': 5, '通算走数': 10, '通算勝率': 0.20, '通算連対率': 0.30, '通算3着内率': 0.40, '平均着順': 4.5, '同距離走数': 5, '同距離勝率': 0.20, '同距離平均着順': 4.0, '同競馬場走数': 1, '同競馬場勝率': 0.0, '同競馬場平均着順': 5.0, '直近1走着順': 3, '直近1走上り': 33.2, '直近2走着順': 4, '直近2走上り': 33.5, '直近3走着順': 2, '直近3走上り': 33.0, '上り平均': 33.2, '上り最速': 33.0, 'タイム平均_同距離': 67.5, '平均4角位置': 4, '直近1走4角': 4},
        'ナムラクレア':      {'馬齢': 7, '斤量': 56, '馬番': 13, '馬体重': 460, '体重増減': 0, '人気': 2, '通算走数': 30, '通算勝率': 0.17, '通算連対率': 0.33, '通算3着内率': 0.50, '平均着順': 4.0, '同距離走数': 12, '同距離勝率': 0.17, '同距離平均着順': 3.5, '同競馬場走数': 4, '同競馬場勝率': 0.25, '同競馬場平均着順': 3.0, '直近1走着順': 2, '直近1走上り': 33.0, '直近2走着順': 3, '直近2走上り': 33.2, '直近3走着順': 2, '直近3走上り': 33.1, '上り平均': 33.1, '上り最速': 32.9, 'タイム平均_同距離': 67.2, '平均4角位置': 3, '直近1走4角': 3},
        'レイピア':        {'馬齢': 4, '斤量': 58, '馬番': 14, '馬体重': 498, '体重増減': 0, '人気': 7, '通算走数': 8, '通算勝率': 0.13, '通算連対率': 0.25, '通算3着内率': 0.38, '平均着順': 4.8, '同距離走数': 3, '同距離勝率': 0.33, '同距離平均着順': 3.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 2, '直近1走上り': 33.0, '直近2走着順': 3, '直近2走上り': 33.5, '直近3走着順': 4, '直近3走上り': 33.8, '上り平均': 33.4, '上り最速': 33.0, 'タイム平均_同距離': 68.0, '平均4角位置': 5, '直近1走4角': 5},
        'インビンシブルパパ':   {'馬齢': 5, '斤量': 58, '馬番': 15, '馬体重': 502, '体重増減': 0, '人気': 11, '通算走数': 10, '通算勝率': 0.10, '通算連対率': 0.20, '通算3着内率': 0.30, '平均着順': 5.5, '同距離走数': 3, '同距離勝率': 0.0, '同距離平均着順': 6.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 4, '直近1走上り': 33.5, '直近2走着順': 5, '直近2走上り': 34.0, '直近3走着順': 6, '直近3走上り': 34.5, '上り平均': 34.0, '上り最速': 33.5, 'タイム平均_同距離': 68.5, '平均4角位置': 5, '直近1走4角': 5},
        'フィオライア':      {'馬齢': 5, '斤量': 56, '馬番': 16, '馬体重': 442, '体重増減': 0, '人気': 13, '通算走数': 8, '通算勝率': 0.13, '通算連対率': 0.13, '通算3着内率': 0.25, '平均着順': 5.5, '同距離走数': 3, '同距離勝率': 0.0, '同距離平均着順': 6.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 5, '直近1走上り': 33.5, '直近2走着順': 6, '直近2走上り': 34.0, '直近3走着順': 4, '直近3走上り': 33.5, '上り平均': 33.7, '上り最速': 33.5, 'タイム平均_同距離': 68.2, '平均4角位置': 5, '直近1走4角': 5},
        'ペアポルックス':     {'馬齢': 5, '斤量': 58, '馬番': 17, '馬体重': 496, '体重増減': 0, '人気': 10, '通算走数': 12, '通算勝率': 0.17, '通算連対率': 0.25, '通算3着内率': 0.33, '平均着順': 5.0, '同距離走数': 5, '同距離勝率': 0.20, '同距離平均着順': 4.5, '同競馬場走数': 1, '同競馬場勝率': 0.0, '同競馬場平均着順': 5.0, '直近1走着順': 3, '直近1走上り': 33.2, '直近2走着順': 5, '直近2走上り': 33.5, '直近3走着順': 4, '直近3走上り': 33.8, '上り平均': 33.5, '上り最速': 33.2, 'タイム平均_同距離': 68.0, '平均4角位置': 5, '直近1走4角': 5},
        'ジューンブレア':     {'馬齢': 5, '斤量': 56, '馬番': 18, '馬体重': 468, '体重増減': 0, '人気': 6, '通算走数': 12, '通算勝率': 0.33, '通算連対率': 0.58, '通算3着内率': 0.58, '平均着順': 2.5, '同距離走数': 7, '同距離勝率': 0.43, '同距離平均着順': 2.0, '同競馬場走数': 0, '同競馬場勝率': 0, '同競馬場平均着順': 0, '直近1走着順': 1, '直近1走上り': 33.0, '直近2走着順': 2, '直近2走上り': 33.0, '直近3走着順': 1, '直近3走上り': 33.2, '上り平均': 33.1, '上り最速': 32.8, 'タイム平均_同距離': 67.2, '平均4角位置': 3, '直近1走4角': 2},
    }

    predictions = []
    for name, data in horses_2026.items():
        score = best_ever.predict_one(data)
        predictions.append({'馬番': data['馬番'], '馬名': name, 'GPスコア': round(score, 4)})

    pred_df = pd.DataFrame(predictions).sort_values('GPスコア', ascending=False)
    print(pred_df.to_string(index=False))

    print(f"\n=== GP TOP3 ===")
    for i, row in pred_df.head(3).iterrows():
        print(f"  {row['馬名']}: スコア={row['GPスコア']}")

    # 結果保存
    result = {
        'model': 'GP Decision Tree',
        'train_w3': round(best_ever_fitness, 4),
        'test_w3': round(test_fitness, 4),
        'tree_depth': best_ever.depth(),
        'rule': best_ever.to_rule_str(),
        'predictions': predictions,
        'history': history[-10:],
        'total_time_sec': round(total_time, 1),
    }

    with open('/tmp/gp_result_2026.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 結果保存: /tmp/gp_result_2026.json")


if __name__ == '__main__':
    main()
