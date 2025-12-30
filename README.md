# 📉 VIX Algorithmic Trading Strategy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Strategy](https://img.shields.io/badge/Strategy-Trend%20Following-orange)
![Status](https://img.shields.io/badge/Status-Academic-lightgrey)

A quantitative research project developed at **Nova School of Business & Economics**. This repository contains a backtesting engine designed to exploit inefficiencies in the volatility markets (VIX) using technical momentum indicators.

## 🧠 Project Overview

Volatility is often mean-reverting but can exhibit strong trend-following characteristics during regime shifts (e.g., market crashes). This project tests a **Moving Average Crossover** strategy on the **CBOE Volatility Index (VIX)** to identify optimal entry/exit points for long and short volatility exposure.

### **Key Objectives**
* **Signal Generation:** Automate entry/exit logic based on dynamic lookback windows (e.g., SMA 20 vs SMA 50).
* **Vectorized Backtesting:** Implement a high-speed Python engine to simulate trades over 10+ years of historical data.
* **Risk Management:** Optimize for **Sharpe Ratio** and **Maximum Drawdown (MDD)** rather than pure absolute returns.
* **Sensitivity Analysis:** Test the robustness of parameters to ensure the strategy is not overfit to historical noise.

## 🛠️ Methodology

1.  **Data Ingestion:** Historical VIX data fetched via `yfinance` (Ticker: `^VIX`).
2.  **Regime Detection:** logic identifies "Fear" (high vol) vs "Complacency" (low vol) markets.
3.  **Execution Simulation:**
    * **Long Signal:** Fast MA crosses *above* Slow MA (Vol spike expected).
    * **Short Signal:** Fast MA crosses *below* Slow MA (Vol crush expected).
4.  **Performance Metrics:**
    * Cumulative Returns vs Buy & Hold
    * Annualized Volatility
    * Sortino Ratio

## 📂 Repository Structure

```text
├── data/
│   ├── (Optional: Place CSV data here if not using API)
├── notebooks/
│   └── VIX_strategy.ipynb   # Core analysis and backtest logic
├── .gitignore
├── requirements.txt
└── README.md
