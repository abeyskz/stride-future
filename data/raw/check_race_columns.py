"""
race.csv の全カラムを確認して、
物理計算に使えるカラムを特定するスクリプト
"""
import pandas as pd
import chardet
import os

input_file = 'race.csv'

print("エンコーディング検出中...")
with open(input_file, 'rb') as f:
    raw = f.read(100000)
enc = chardet.detect(raw)['encoding'] or 'shift_jis'
print(f"エンコーディング: {enc}")

print("先頭5行だけ読み込み...")
df = pd.read_csv(input_file, encoding=enc, low_memory=False, nrows=5)

print(f"\n全カラム一覧（{len(df.columns)}個）:")
for i, col in enumerate(df.columns):
    print(f"  {i+1:3d}. {col}")

print("\nサンプル値（1行目）:")
for col in df.columns:
    print(f"  {col}: {df[col].iloc[0]}")

