"""
race.csv → 物理計算用フルカラム版
カラムを減らさず、期間で2分割して出力

  race_full_2001_2011.csv  (2001〜2011年)
  race_full_2012_2025.csv  (2012〜2025年) ← 高松宮記念の改修後

使い方:
  race.csv と同じフォルダに置いて実行
  python make_race_full.py
"""
import pandas as pd
import chardet
import os

input_file = 'race.csv'

# =============================
# エンコーディング検出
# =============================
print("エンコーディング検出中...")
with open(input_file, 'rb') as f:
    raw = f.read(100000)
enc = chardet.detect(raw)['encoding'] or 'cp932'
print(f"エンコーディング: {enc}")

# =============================
# 読み込み
# =============================
print("読み込み中（少し時間かかります）...")
df = pd.read_csv(input_file, encoding=enc, low_memory=False)
print(f"元データ: {len(df):,}行 × {len(df.columns)}カラム")
print(f"元ファイルサイズ: {os.path.getsize(input_file) / 1024 / 1024:.1f}MB")

# =============================
# 年の正規化
# =============================
df['年_int'] = pd.to_numeric(df['年'], errors='coerce')
print(f"\n年の範囲（2桁）: {df['年_int'].min()} 〜 {df['年_int'].max()}")

# 2001年以降に絞る（年2桁で1以上 = 2001年以降）
df = df[df['年_int'] >= 1].copy()
print(f"2001年以降: {len(df):,}行")

# =============================
# 異常着順除外
# =============================
df = df[pd.to_numeric(df['確定着順'], errors='coerce').notna()].copy()
df['確定着順'] = df['確定着順'].astype(int)
print(f"異常着順除外後: {len(df):,}行")

# =============================
# 枠番計算（馬番から）
# =============================
def bango_to_wakuban(bango):
    try:
        b = int(bango)
        if b <= 0:  return None
        if b <= 14: return (b + 1) // 2
        else:       return 8
    except:
        return None

df['枠番'] = df['馬番'].apply(bango_to_wakuban)

# =============================
# 保持するカラム
# =============================
keep_cols = [
    # 基本情報
    '年', '月', '日', '回次', '場所', '日次', 'レース番号',
    'レース名', 'クラスコード', 'レースID',
    # コース情報
    '芝・ダ', 'コースコード', '距離', '馬場状態',
    # 馬情報
    '馬名', '性別', '年齢', '血統登録番号', '生年月日',
    '父馬名', '母馬名', '母の父馬名', '毛色',
    '馬体重',
    # 騎手・調教師
    '騎手名', '騎手コード', '斤量',
    '調教師', '調教師コード', '所属地',
    # レース結果
    '頭数', '馬番', '枠番',
    '確定着順', '入線着順', '異常コード',
    '人気順', '単勝オッズ',
    # タイム系
    '走破タイム', '走破時計', '着差タイム',
    '上がり3Fタイム',
    # 位置取り
    '通過順1', '通過順2', '通過順3', '通過順4',
    # ペース指数
    'PCI',
    # 賞金
    '賞金',
    # オーナー等（あれば）
    '現馬主名', '生産者名',
]

available = [c for c in keep_cols if c in df.columns]
missing   = [c for c in keep_cols if c not in df.columns]
if missing:
    print(f"\n⚠️  見つからなかったカラム（スキップ）: {missing}")

df_out = df[available].copy()
print(f"\n出力カラム数: {len(available)}")

# =============================
# 期間で分割して保存
# =============================
# 2001〜2011年（年2桁: 1〜11）
df_old = df_out[df['年_int'] <= 11].copy()
# 2012〜2025年（年2桁: 12〜25）
df_new = df_out[df['年_int'] >= 12].copy()

print(f"\n2001〜2011年: {len(df_old):,}行")
print(f"2012〜2025年: {len(df_new):,}行")

out1 = 'race_full_2001_2011.csv'
out2 = 'race_full_2012_2025.csv'

print(f"\n保存中: {out1} ...")
df_old.to_csv(out1, index=False, encoding='utf-8-sig')
size1 = os.path.getsize(out1) / 1024 / 1024
print(f"  → {size1:.1f}MB")

print(f"保存中: {out2} ...")
df_new.to_csv(out2, index=False, encoding='utf-8-sig')
size2 = os.path.getsize(out2) / 1024 / 1024
print(f"  → {size2:.1f}MB")

print(f"""
✅ 完了！

出力ファイル:
  {out1}  ({size1:.1f}MB)  → 物理計算の学習補助用
  {out2}  ({size2:.1f}MB)  → 高松宮記念の主力データ

使い方のヒント:
  - 高松宮記念の分析は race_full_2012_2025.csv をメインに使う
  - 2012〜2022: 学習用（11回分）
  - 2023〜2024: 検証用（2回分）
  - 2025:       最終評価用（触らない）
""")
