import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')


##### 1. DATA DOWNLOAD & PREPARATION

start, end, tickers = '2007-01-03', '2025-11-19', ['SPY', 'IEI', 'GLD', '^VIX']
raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Adj Close'] if 'Adj Close' in raw.columns.levels[0] else raw['Close']
else:
    prices = raw
prices = prices[['SPY', 'IEI', 'GLD', '^VIX']].ffill().dropna().rename(columns={'^VIX': 'VIX'})

rf_raw = yf.download('^IRX', start=start, end=end, progress=False)
rf_close = rf_raw['Close'].squeeze() if isinstance(rf_raw.columns, pd.MultiIndex) else rf_raw['Close']
daily_rf = rf_close.iloc[:, 0] if isinstance(rf_close, pd.DataFrame) else rf_close
daily_rf = daily_rf.reindex(prices.index, method='nearest') / 100 / 252


##### 2. STRATEGY LOGIC

returns = prices[['SPY', 'IEI', 'GLD']].pct_change().dropna()
vix_lag = prices['VIX'].shift(1).reindex(returns.index)

conds = [(vix_lag <= 15), (vix_lag > 15) & (vix_lag <= 25), (vix_lag > 25)]
w_spy = np.select(conds, [0.90, 0.20, 0.00], 0.20)
w_iei = np.select(conds, [0.10, 0.60, 0.80], 0.60)
w_gld = np.select(conds, [0.00, 0.20, 0.20], 0.20)

strat_ret_gross = (returns['SPY'] * w_spy + returns['IEI'] * w_iei + returns['GLD'] * w_gld).to_frame('SVAS (Gross)')
regime = pd.Series(np.select(conds, [0, 1, 2], 1), index=returns.index)

daily_weights = pd.DataFrame({'SPY': w_spy, 'IEI': w_iei, 'GLD': w_gld}, index=returns.index)
abs_diff = daily_weights.diff().abs().fillna(0)
cost_dict = {'SPY': 0.00025, 'IEI': 0.00030, 'GLD': 0.00030}
costs = abs_diff.mul(pd.Series(cost_dict)).sum(axis=1)
strat_ret_net = strat_ret_gross['SVAS (Gross)'] - costs
strat_ret_net = strat_ret_net.to_frame('SVAS (Net)')

benches = pd.DataFrame({
    '40E/60B':    0.60 * returns['SPY'] + 0.40 * returns['IEI'],
    '100% SPY': returns['SPY'],
    '100% IEI': returns['IEI']
})


##### 3. METRICS

def get_metrics(s, name):
    s = s.squeeze().dropna()
    rf = daily_rf.reindex(s.index, method='nearest').fillna(0).squeeze()
    cagr = (1 + s).prod() ** (252/len(s)) - 1
    vol = s.std() * np.sqrt(252)
    excess_ret = s - rf
    sharpe = (excess_ret.mean() / excess_ret.std()) * np.sqrt(252)
    downside_ret = excess_ret[excess_ret < 0]
    sortino = (excess_ret.mean() * np.sqrt(252)) / downside_ret.std() if downside_ret.std() > 0 else np.nan
    var_95 = s.quantile(0.05)
    cum = (1 + s).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return [cagr, vol, sharpe, sortino, max_dd, calmar, var_95]

metrics_df = pd.DataFrame([get_metrics(strat_ret_gross, 'SVAS (Gross)')] +
                          [get_metrics(strat_ret_net, 'SVAS (Net)')] +
                          [get_metrics(benches[c], c) for c in benches.columns],
                          columns=['CAGR', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'VaR 95%'],
                          index=['SVAS (Gross)', 'SVAS (Net)'] + list(benches.columns))

strategies = metrics_df.loc[['SVAS (Gross)', 'SVAS (Net)']]
others = metrics_df.drop(['SVAS (Gross)', 'SVAS (Net)']).sort_values('Sharpe Ratio', ascending=False)
final_metrics = pd.concat([strategies, others])


##### 4. PLOTTING

colors = {
    'vix_net':   '#003758', 
    'vix_gross': '#006064', 
    'spy':       'gray',
    'low': 'palegreen', 'mid': 'lemonchiffon', 'high': 'salmon'
}

def clean_plot(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_xlabel("")

# SET FONT TO TIMES NEW ROMAN & STYLE
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False

# DOUBLE THE FONT SIZE
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 34

# --- GRAPH 1: SVAS NET ONLY ---
plt.figure(figsize=(16, 10))
ax1 = plt.gca()

# Data: Rebased to 100
net_rebased = (1 + strat_ret_net).cumprod() * 100

ax1.plot(net_rebased, lw=1.5, c=colors['vix_net'], label='SVAS (Net)')

# Styling Primary Axis
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.set_xlim(prices.index[0], prices.index[-1])
ax1.set_xlabel("")

# Y-Axis Label
ax1.set_ylabel('Index Level (Rebased to 100)', fontweight='bold', fontname='DejaVu Serif')

# Secondary Axis
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)

y1_min, y1_max = ax1.get_ylim()
ax2.set_ylim((y1_min/100)-1, (y1_max/100)-1)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_ylabel('Performance since inception (%)', fontweight='bold', fontname='DejaVu Serif', rotation=270, labelpad=40)

plt.title('SVAS (Net) Performance', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'])
lines, labels = ax1.get_legend_handles_labels()
ax1.legend(lines, labels, loc='upper left')
plt.show()

# --- GRAPH 2: TABLE ---
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

# Custom Colormap
navy_cmap = mcolors.LinearSegmentedColormap.from_list("navy_gradient", ["#89CFF0", "#003758"])

def get_color(val, col_name, data):
    s = data[col_name]
    if s.max() == s.min():
        return mcolors.to_hex(navy_cmap(0.5))

    rng = s.max() - s.min()

    # Logic: Darkest Color for "Best" metric
    if col_name in ['Volatility']:
        # Lower = Better = Darker
        norm_val = (s.max() - val) / rng
    else:
        # Higher = Better = Darker
        norm_val = (val - s.min()) / rng

    return mcolors.to_hex(navy_cmap(norm_val))

cell_text = []
cell_colours = []
header = final_metrics.columns.tolist()
row_labels = final_metrics.index.tolist()

for idx, row in final_metrics.iterrows():
    row_text = []
    row_cols = []
    for col in header:
        val = row[col]
        if col in ['CAGR', 'Volatility', 'Max Drawdown', 'VaR 95%']:
            txt = f"{val:.2%}"
        else:
            txt = f"{val:.2f}"
        row_text.append(txt)
        bg_color = get_color(val, col, final_metrics)
        row_cols.append(bg_color)
    cell_text.append(row_text)
    cell_colours.append(row_cols)

the_table = ax.table(cellText=cell_text, rowLabels=row_labels, colLabels=header,
                     cellColours=cell_colours, loc='center', cellLoc='center')

the_table.auto_set_font_size(False)
the_table.set_fontsize(20)
the_table.scale(1.2, 3)

# --- STRICT STYLING FOR HEADERS vs DATA ---
for (row, col), cell in the_table.get_celld().items():
    if row == -1 or col == -1:
        cell.get_text().set_fontweight('bold')
        cell.get_text().set_color('black')
        cell.set_facecolor('white') 
        
        

    # DATA CELLS
    else:
        cell.get_text().set_fontweight('bold')
        cell.get_text().set_color('black')
        cell.set_edgecolor('white') 

plt.title('Performance Summary', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'], pad=20)
plt.tight_layout()
plt.show()

# --- GRAPH 3: CUMULATIVE RETURNS COMPARISON ---
plt.figure(figsize=(16, 10))
ax1 = plt.gca()

# 1. Prepare Rebased Data (Index 100)
data_to_plot = {
    'SVAS (Net)': (1 + strat_ret_net['SVAS (Net)']).cumprod() * 100,
    'SVAS (Gross)': (1 + strat_ret_gross['SVAS (Gross)']).cumprod() * 100,
    '40E/60B': (1 + benches['40E/60B']).cumprod() * 100,
    '100% SPY': (1 + benches['100% SPY']).cumprod() * 100
}

# 2. Plotting on Primary Axis
ax1.plot(data_to_plot['SVAS (Net)'], lw=1.5, c=colors['vix_net'], label='SVAS (Net)')
ax1.plot(data_to_plot['SVAS (Gross)'], lw=1.5, c='#1669A4', label='SVAS (Gross)')
ax1.plot(data_to_plot['40E/60B'], lw=1.5, c='tab:purple', alpha=0.9, label='40E/60B')
ax1.plot(data_to_plot['100% SPY'], lw=1.5, alpha=0.6, c=colors['spy'], label='100% SPY')

# 3. Styling Primary Axis
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.set_xlim(prices.index[0], prices.index[-1])
ax1.set_xlabel("")

# Y-Axis Label
ax1.set_ylabel('Index Level (Rebased to 100)', fontweight='bold', fontname='DejaVu Serif')

# 4. Secondary Axis (Percentage)
ax2 = ax1.twinx()
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)

y1_min, y1_max = ax1.get_ylim()
ax2.set_ylim((y1_min/100)-1, (y1_max/100)-1)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_ylabel('Performance since inception (%)', fontweight='bold', fontname='DejaVu Serif', rotation=270, labelpad=40)

plt.title('Cumulative Returns', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'])
ax1.legend(loc='upper left')
plt.show()

# --- GRAPH 4: DRAWDOWN ---
plt.figure(figsize=(16, 8))
ax = plt.gca()
clean_plot(ax)
ax.set_xlim(prices.index[0], prices.index[-1])

dd_net = (1+strat_ret_net).cumprod() / (1+strat_ret_net).cumprod().cummax() - 1
dd_bench = (1+benches['40E/60B']).cumprod() / (1+benches['40E/60B']).cumprod().cummax() - 1

dd_net.plot(ax=ax, lw=1.25, c=colors['vix_net'], label='SVAS (Net)')
dd_bench.plot(ax=ax, lw=1.25, alpha=0.8, c='tab:purple', label='40E/60B')

plt.title('Drawdown Profile', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'])
plt.legend()
ax.set_xlabel("")
plt.show()

# --- GRAPH 5: ALLOCATION ---
fig, ax = plt.subplots(figsize=(16, 10))
clean_plot(ax)
ax.set_xlim(prices.index[0], prices.index[-1])

ax.plot(returns.index, prices['VIX'].reindex(returns.index), c=colors['vix_net'], lw=1, label='VIX Index')
trans = ax.get_xaxis_transform()

c_low  = 'white'
c_mid  = '#FFCDD2' 
c_high = '#EF9A9A' 

ax.fill_between(returns.index, 0, 1, where=(regime==0), color=c_low, alpha=0.6, transform=trans, lw=0)
ax.fill_between(returns.index, 0, 1, where=(regime==1), color=c_mid, alpha=0.6, transform=trans, lw=0)
ax.fill_between(returns.index, 0, 1, where=(regime==2), color=c_high, alpha=0.5, transform=trans, lw=0)

ax.axhline(15, color='#E72831', linewidth=2, alpha=0.8)
ax.axhline(25, color='#E72831', linewidth=2, alpha=0.8)

ax.set_ylim(0, prices['VIX'].max()*1.1)
ax.set_title('VIX Regime Allocation', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'], pad=20)

patches = [
    mpatches.Patch(facecolor=c_low, edgecolor='black', linewidth=1, label='Low Vol: 90% SPY / 10% IEI'),
    mpatches.Patch(facecolor=c_mid, edgecolor='black', linewidth=1, label='Mid Vol: 20% SPY / 60% IEI / 20% GLD'),
    mpatches.Patch(facecolor=c_high, edgecolor='black', linewidth=1, label='High Vol: 80% IEI / 20% GLD')
]
cutoff_line = mlines.Line2D([], [], color="#E72831", linewidth=2, label='Regime Cutoffs (15/25)')
vix_line = mlines.Line2D([],[], color=colors['vix_net'], lw=2, label='VIX Index')

ax.legend(handles=patches + [cutoff_line, vix_line], loc='upper right')
ax.set_xlabel("")
plt.show()

# --- GRAPH 6: VOLATILITY ---
plt.figure(figsize=(16, 8))
ax = plt.gca()
clean_plot(ax)
ax.set_xlim(prices.index[0], prices.index[-1])

strat_ret_net.rolling(252).std().mul(np.sqrt(252)).plot(ax=ax, lw=1.5, c=colors['vix_net'], label='SVAS (Net)')
benches['40E/60B'].rolling(252).std().mul(np.sqrt(252)).plot(ax=ax, lw=1.25, c='tab:purple', alpha=0.8, label='40E/60B')
benches['100% SPY'].rolling(252).std().mul(np.sqrt(252)).plot(ax=ax, lw=1.25, alpha=0.6, c=colors['spy'], label='100% SPY')

plt.title('Rolling 1-Year Volatility', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'])
plt.legend()
ax.set_xlabel("")
plt.show()

# --- GRAPH 7: BETA ---
plt.figure(figsize=(16, 8))
ax = plt.gca()
clean_plot(ax)
ax.set_xlim(prices.index[0], prices.index[-1])

rolling_window = 252
asset_ret = strat_ret_net['SVAS (Net)']
market_ret = returns['SPY']

rolling_cov = asset_ret.rolling(window=rolling_window).cov(market_ret)
rolling_var = market_ret.rolling(window=rolling_window).var()
rolling_beta = rolling_cov / rolling_var

rolling_beta.plot(ax=ax, lw=1, color=colors['vix_net'], label='Strategy Beta')
ax.axhline(1.0, color='gray', linewidth=2, linestyle='--', label='SPY')

ax.set_ylim(top=1.2)
plt.title('Rolling 1-Year Beta (vs SPY)', fontweight='bold', fontname='DejaVu Serif', color=colors['vix_net'])
plt.legend()
ax.set_xlabel("")
plt.show()
