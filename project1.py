import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.filters.hp_filter import hpfilter

# FRED APIキーを設定（自分のAPIキーに置き換えてください）
fred = Fred(api_key='73c23de34da9dc98199b57d5979386d7')

# 日本のGDPデータを取得（四半期データ）
gdp_data = fred.get_series('CLVMNACSCAB1GQES')
gdp_data = gdp_data.dropna()

# 対数変換
log_gdp = np.log(gdp_data)

# HPフィルターを適用 (異なるλ値)
lambdas = [10, 100, 1600]
trends = {}
cycles = {}

for lam in lambdas:
    cycle, trend = hpfilter(log_gdp, lam)
    trends[lam] = trend
    cycles[lam] = cycle

# グラフ1：元のデータとトレンド成分の比較
plt.figure(figsize=(12, 6))
plt.plot(log_gdp, label='Log GDP')
for lam in lambdas:
    plt.plot(trends[lam], label=f'Trend (λ = {lam})')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Log GDP')
plt.title('Original Log GDP and Trend Components (Spain)')
plt.show()

# グラフ2：循環成分の比較
plt.figure(figsize=(12, 6))
for lam in lambdas:
    plt.plot(cycles[lam], label=f'Cycle (λ = {lam})')

plt.legend()
plt.xlabel('Date')
plt.ylabel('Cycle Component')
plt.title('Cycle Components (United States)')
plt.show()