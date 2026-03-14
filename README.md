
# Market Regime Detection with a 2-State Gaussian HMM

This project detects **market regimes (bull vs bear)** using a **two-state Gaussian Hidden Markov Model (HMM)** trained on daily S&P 500 returns and changes in implied volatility (VIX). The main goal is to show how an HMM can segment market dynamics into persistent latent states, and how those states can be used to build a simple regime-filtered strategy.

The work is exploratory and intentionally minimal: two features, two regimes, and a walk-forward style out-of-sample check.

## Repo structure

- [data/data_market.csv](data/data_market.csv): daily close data for `^GSPC` and `^VIX`.
- [scripts/extract.py](scripts/extract.py): downloads data from Yahoo Finance and writes a CSV.
- [scripts/notebook.ipynb](scripts/exp.ipynb): end-to-end exploratory notebook (EDA → HMM fit → plots → out-of-sample).
- [requirements.txt](requirements.txt): Python dependencies.

## Data

The dataset is sourced from Yahoo Finance via `yfinance`:

- `^GSPC` (S&P 500 index)
- `^VIX` (CBOE Volatility Index)

The workflow uses **daily close prices** and derives features from them.

To (re)download the data, run from the repo root:

```bash
python scripts/extract.py
```

This writes to `data/data_market.csv` (relative to your current working directory).

## Feature engineering

Two daily features are used:

1. **S&P 500 log return**

$$r_t = \log\left(\frac{P_t}{P_{t-1}}\right)$$

2. **VIX log difference** (log change in VIX level)

$$\Delta v_t = \log(VIX_t) - \log(VIX_{t-1})$$

Rows with NaNs induced by shifting/differencing are dropped before modeling.

Because HMM emission distributions depend on scale, features are standardized using `StandardScaler`:

$$x'_t = \frac{x_t - \mu}{\sigma}$$

## Model: 2-state Gaussian HMM

We fit a Gaussian HMM with:

- `n_components=2` (two latent regimes)
- `covariance_type="full"` so the model can learn correlation between return and volatility feature
- `n_iter=1000` for convergence headroom

Conceptually:

- Hidden state $z_t \in \{0,1\}$ follows a Markov chain (transition matrix learned from data).
- Observations $x_t$ are drawn from a Gaussian distribution conditioned on the state:

$$x_t \mid z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k)$$

## Interpreting regimes (bull vs bear)

State labels in an unsupervised HMM are arbitrary. The notebook assigns “bull” vs “bear” by inspecting the **state-conditional means in original units** (inverse-transforming the standardized means):

- A state with **positive average S&P 500 log returns** and **lower/negative VIX log changes** is treated as “bull”.
- A state with **negative average returns** and **rising VIX** is treated as “bear”.

You can also use posterior state probabilities to avoid hard assignments:

$$\gamma_t(k) = P(z_t=k \mid x_{1:T})$$

## Strategy experiment (in-sample)

The notebook compares:

- **Buy & hold**: accumulate all daily S&P 500 log returns.
- **Only-bull strategy**: accumulate returns only on days where the inferred regime is “bull”; otherwise the return is set to 0 (flat exposure).

This is a didactic regime-filtering example; it does **not** include transaction costs, slippage, or position sizing.

## Out-of-sample (OOS) walk-forward check

The OOS section performs a simple time split:

- Train: 2010-01-05 → 2024-12-31
- Test: 2025-01-01 → 2025-12-31

The HMM is fit on the training period and then used to predict regimes on the 2025 period, and the same buy & hold vs only-bull cumulative log return plot is produced.

## Reproducibility

Create a virtual environment and install dependencies:

```bash
python -m venv venv
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then open this notebook and run cells top-to-bottom:

- [scripts/notebook.ipynb](scripts/notebook.ipynb)

## Important caveats

- **Scaler leakage in OOS**: in the current OOS code, the test features are scaled with `fit_transform` on the test set. For a strict OOS setup, you should use `scaler_oos.transform(X_test)` (fit scaler only on train).
- **Look-ahead bias**: using the regime predicted from day $t$’s features to decide exposure on day $t$ can be optimistic if the decision is assumed to happen before observing close-to-close returns. A more realistic backtest shifts the signal by one day (trade on $t+1$ using information up to $t$).
- **State label instability**: the mapping “state 0 = bear, state 1 = bull” is not guaranteed across refits. Always infer labels from state statistics (means/volatility) rather than assuming indices.
- **No costs/constraints**: the strategy ignores transaction costs, taxes, borrow constraints, and financing costs.
- **Model simplicity**: a 2-state Gaussian HMM is a strong simplification; more states/features can help but also increase overfitting risk.

## What to extend next

- Add a strict, no-leakage backtest with lagged signals.
- Compare against baselines (e.g., volatility filter, moving-average regime).
- Try more features (realized volatility, drawdown, macro proxies) and more states.
- Use expanding-window refits (true walk-forward) instead of one train/test split.

