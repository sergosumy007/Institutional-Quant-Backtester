# Institutional-Grade Quant Backtester & Validation Engine

Welcome to my proprietary algorithmic trading framework. This repository demonstrates the core architecture of my vectorized backtesting and validation engine, built strictly in Python (Pandas/NumPy) for quantitative research, specifically focusing on Smart Money Concepts (SMC), Volume Profile, and Order Flow strategies.

> ⚠️ **Note:** To protect intellectual property, the actual alpha-generating strategies (`bt_*.py`) and proprietary predictive models are kept private. This public repository serves to showcase the underlying infrastructure and my capability to rapidly digitize and rigorously validate any custom trading logic.

## ⚙️ Core Engine Architecture

My framework is divided into two strict pipelines to completely eliminate look-ahead bias and prevent curve-fitting:

### Block 1: Strategy Development & In-Sample (IS) Discovery
* **Vectorized Signal Generation:** Translates complex manual setups (e.g., FVG retests, Wyckoff Springs, Volume Climax) into pure mathematical algorithms.
* **Zero Look-Ahead Guarantee:** Signals are generated strictly on bar close and executed on the next bar's open.
* **Bare Hypothesis Testing:** Every concept is first tested as a "bare hypothesis" against a random-entry baseline (`edge_gap > 0`) before any filters are applied.
* **Dynamic Risk Management:** Integrated calculation of dynamic SL/TP based on ATR, structural levels (Key Highs/Lows), and dynamic RR adjustments.

### Block 2: Institutional Validation & Out-Of-Sample (OOS) Testing
* **Strict IS/OOS Split:** Parameters are locked during Phase 1. OOS data is revealed only once during final validation.
* **9-Tier Validator (`strategy_validator.py`):** * *T1/T2:* Look-ahead and architectural integrity checks.
    * *T3/T4:* Win Rate and Profit Factor statistical significance.
    * *T5/T6:* Slippage Simulation and Commission impact (Cost Model).
    * *T7:* Monte Carlo Stress Testing (10,000 iterations).
    * *T8/T9:* Walk-Forward Analysis (WFA) and Multi-Regime testing.
* **Machine Learning Overlay:** Random Forest metadata injection for signal classification and regime filtering.

## 📊 Interactive Visualization
The engine includes a custom UI Screener (built with Plotly & Streamlit, see `app.py`) capable of real-time rendering of:
* Candlestick charts with executed trades (LONG/SHORT markers).
* Dynamic Value Area (VAH/VAL) and Point of Control (POC) profiles.
* Trade management trajectories (Entry, SL, TP execution paths).

## 💼 Freelance & Consulting
I specialize in helping traders automate their manual strategies. If you have a trading concept (especially SMC or Volume-based) and need it coded in Python, stress-tested with Monte Carlo simulations, and delivered with a comprehensive performance report, let's connect.

**Turnaround time:** Typically 48 hours for a complete digitization and validation report, utilizing advanced AI orchestration.
