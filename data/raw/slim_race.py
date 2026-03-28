"""
race.csv 軽量化スクリプト
五行モデル検証に必要なカラムだけ残して圧縮する
2001年以降のデータに絞る
"""
import pandas as pd
import chardet
import os

input_file = 'race.csv'
output_file = 'race_slim.csv'

print("エンコーディング検出中...")
with open(input_file, 'rb') as f:
    raw = f.read(100000)
enc = chardet.detect(raw)['encoding'] or 'shift_jis'
print(f"エンコーディング: {enc}")

print("読み込み中（少し時間かかります）...")
df = pd.read_csv(input_file, encoding=enc, low_memory=False)
print(f"元データ: {len(df):,}行 × {len(df.columns)}カラム")
print(f"元ファイルサイズ: {os.path.getsize(input_file) / 1024 / 1024:.1f}MB")

# 年の確認と2001年以降に絞る
print(f"\n年カラムのサンプル: {df['年'].head(5).tolist()}")
print(f"年の範囲: {df['年'].min()} 〜 {df['年'].max()}")

df['年_int'] = pd.to_numeric(df['年'], errors='coerce')
before = len(df)
df = df[df['年_int'] >= 1]  # 年2桁で01以降 = 2001年以降
print(f"\n2001年以降に絞った後: {len(df):,}行（{before - len(df):,}行削除）")

# 必要カラムだけ残す
keep_cols = [
    '年', '月', '日',
    '場所',
    'レースID',
    '芝・ダ',
    '距離',
    '馬名',
    '騎手名',
    '騎手コード',
    '頭数',
    '馬番',
    '確定着順',
    '人気順',
    '血統登録番号',
]

available = [c for c in keep_cols if c in df.columns]
missing   = [c for c in keep_cols if c not in df.columns]
if missing:
    print(f"⚠️ 見つからなかったカラム: {missing}")

df_slim = df[available].copy()

# 枠番計算
def bango_to_wakuban(bango):
    try:
        b = int(bango)
        if b <= 0:  return None
        if b <= 14: return (b + 1) // 2
        else:       return 8
    except:
        return None

df_slim['枠番'] = df_slim['馬番'].apply(bango_to_wakuban)

# 異常着順除外
df_slim = df_slim[pd.to_numeric(df_slim['確定着順'], errors='coerce').notna()]
df_slim['確定着順'] = df_slim['確定着順'].astype(int)
print(f"異常着順除外後: {len(df_slim):,}行")

# 保存
df_slim.to_csv(output_file, index=False, encoding='utf-8-sig')
size_mb = os.path.getsize(output_file) / 1024 / 1024
print(f"\n✅ 完了！")
print(f"出力ファイル: {output_file}")
print(f"出力サイズ: {size_mb:.1f}MB")
