import matplotlib
matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# FRED APIのキー設定
fred = Fred(api_key='73c23de34da9dc98199b57d5979386d7')

# スペインのGDPデータ
print("スペインのデータを取得中...")
spain_gdp = fred.get_series('CLVMNACSCAB1GQES')
spain_gdp = spain_gdp.dropna()
print(f"スペインのデータ期間: {spain_gdp.index[0]} から {spain_gdp.index[-1]}")

# 日本のGDPデータ
print("日本のデータを取得中...")
japan_gdp = fred.get_series('JPNRGDPEXP')
japan_gdp = japan_gdp.dropna()
print(f"日本のデータ期間: {japan_gdp.index[0]} から {japan_gdp.index[-1]}")

# 共通の期間でデータを整える
common_start = max(spain_gdp.index.min(), japan_gdp.index.min())
common_end = min(spain_gdp.index.max(), japan_gdp.index.max())

spain_gdp_aligned = spain_gdp[common_start:common_end]
japan_gdp_aligned = japan_gdp[common_start:common_end]

print(f"\n共通分析期間: {common_start} から {common_end}")
print(f"スペインのデータ点数: {len(spain_gdp_aligned)}")
print(f"日本のデータ点数: {len(japan_gdp_aligned)}")

# 対数変換
log_spain_gdp = np.log(spain_gdp_aligned)
log_japan_gdp = np.log(japan_gdp_aligned)

# HPフィルター
print("\nHP-filterを適用中...")
spain_trend, spain_cycle = hpfilter(log_spain_gdp, lamb=1600)
japan_trend, japan_cycle = hpfilter(log_japan_gdp, lamb=1600)

# 統計分析
print("\n" + "="*50)
print("ステップ4：統計分析")
print("="*50)

spain_cycle_std = spain_cycle.std()
japan_cycle_std = japan_cycle.std()

print(f"スペインの循環変動成分の標準偏差: {spain_cycle_std:.4f}")
print(f"日本の循環変動成分の標準偏差: {japan_cycle_std:.4f}")

std_ratio = spain_cycle_std / japan_cycle_std
print(f"\n標準偏差の比率（スペイン/日本）: {std_ratio:.3f}")

if std_ratio > 1:
    print(f"→ スペインの景気変動は日本より {std_ratio:.2f}倍大きい")
else:
    print(f"→ 日本の景気変動はスペインより {1/std_ratio:.2f}倍大きい")

correlation = spain_cycle.corr(japan_cycle)
print(f"\nスペインと日本の循環変動成分の相関係数: {correlation:.4f}")

if correlation > 0.7:
    correlation_strength = "非常に強い正の相関"
elif correlation > 0.5:
    correlation_strength = "強い正の相関"
elif correlation > 0.3:
    correlation_strength = "中程度の正の相関"
elif correlation > 0.1:
    correlation_strength = "弱い正の相関"
elif correlation > -0.1:
    correlation_strength = "ほとんど相関なし"
elif correlation > -0.3:
    correlation_strength = "弱い負の相関"
else:
    correlation_strength = "中程度以上の負の相関"

print(f"→ {correlation_strength}")

# グラフ作成
print("\n" + "="*50)
print("ステップ5：グラフ作成")
print("="*50)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(spain_cycle.index, spain_cycle, label='スペイン', color='red', linewidth=2)
plt.plot(japan_cycle.index, japan_cycle, label='日本', color='blue', linewidth=2)
plt.title('循環変動成分の比較（スペイン vs 日本）', fontsize=12, fontweight='bold')
plt.ylabel('循環変動成分')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.scatter(spain_cycle, japan_cycle, alpha=0.6, color='purple', s=30)
plt.xlabel('スペインの循環変動成分')
plt.ylabel('日本の循環変動成分')
plt.title(f'相関関係\n(相関係数: {correlation:.3f})', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
z = np.polyfit(spain_cycle, japan_cycle, 1)
p = np.poly1d(z)
plt.plot(spain_cycle, p(spain_cycle), "r--", alpha=0.8, linewidth=2)

plt.subplot(2, 2, 3)
plt.plot(spain_trend.index, spain_trend, label='スペイン（トレンド）', color='lightcoral', linewidth=2)
plt.plot(japan_trend.index, japan_trend, label='日本（トレンド）', color='lightblue', linewidth=2)
plt.title('トレンド成分の比較（参考）', fontsize=12, fontweight='bold')
plt.ylabel('対数実質GDP（トレンド）')
plt.xlabel('年')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
countries = ['スペイン', '日本']
std_values = [spain_cycle_std, japan_cycle_std]
bars = plt.bar(countries, std_values, color=['red', 'blue'], alpha=0.7)
plt.title('循環変動成分の標準偏差比較', fontsize=12, fontweight='bold')
plt.ylabel('標準偏差')
for bar, value in zip(bars, std_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
             f'{value:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# サマリー
print("\n" + "="*60)
print("                   分析結果サマリー")
print("="*60)
print(f"分析対象: スペイン vs 日本")
print(f"分析期間: {common_start.strftime('%Y-%m')} ～ {common_end.strftime('%Y-%m')}")
print(f"データ点数: {len(spain_cycle)}期間\n")

print("【標準偏差による変動の大きさ比較】")
print(f"  スペイン: {spain_cycle_std:.4f}")
print(f"  日本:     {japan_cycle_std:.4f}")
print(f"  比率:     {std_ratio:.3f} (スペイン/日本)\n")

print("【相関分析】")
print(f"  相関係数: {correlation:.4f}")
print(f"  相関の強さ: {correlation_strength}\n")

print("【経済学的解釈】")
if abs(correlation) > 0.5:
    print("  ✓ 両国の景気循環は高い同調性を示している")
elif abs(correlation) > 0.3:
    print("  ✓ 両国の景気循環は中程度の同調性を示している")
else:
    print("  ✓ 両国の景気循環の同調性は低い")

if std_ratio > 1.2:
    print(f"  ✓ スペインの方が景気変動が大きく、より不安定")
elif std_ratio < 0.8:
    print(f"  ✓ 日本の方が景気変動が大きく、より不安定")
else:
    print(f"  ✓ 両国の景気変動の大きさは似ている")

print(f"\nプログラム完了：統計分析とグラフ作成が正常に実行されました。")
