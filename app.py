import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import zipfile
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize  # installed together with scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
)

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="QUANTLAB 39 | Multi Asset Intelligence",
    page_icon="📈",
    layout="wide"
)

# Streamlit API compatibility (new versions use width="stretch")
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (1, 0)

WIDE = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}


def plot_chart(fig, key=None, **kw):
    # theme=None keeps our own dark Plotly styling (Streamlit theme was hiding titles)
    st.plotly_chart(fig, key=key, theme=None, **WIDE)

# =====================================================
# THEME
# =====================================================

st.markdown("""
<style>
.stApp { background: #0B1220; color: #F1F5F9; }
[data-testid="stHeader"] { background: #0B1220; }
[data-testid="stSidebar"] { background: #172438; }
[data-testid="stSidebar"] * { color: #E2E8F0; }
h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 750; }
p, label, li, span { color: #CBD5E1; }
[data-testid="stCaptionContainer"] * { color: #AEBBCD !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #C5D0E0 !important; }

/* Metric cards: dark card, bright readable text */
[data-testid="stMetric"] {
    background: #172438;
    border: 1px solid #2B3C54;
    padding: 16px;
    border-radius: 13px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: #94A3B8 !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: #00E5BE !important;
    font-weight: 700;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background: #00BFA6;
    color: #07111F !important;
    border: none;
    border-radius: 9px;
    font-weight: 750;
    padding: 10px 18px;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #00D6B5; color: #07111F !important;
}
.stButton > button p, .stDownloadButton > button p { color: #07111F !important; }

/* Inputs: dark boxes with light text */
[data-baseweb="select"] > div,
[data-baseweb="input"],
[data-baseweb="base-input"],
[data-testid="stNumberInput"] input {
    background-color: #0F1B2D !important;
    color: #F1F5F9 !important;
    border-color: #2B3C54 !important;
}
[data-baseweb="select"] *, [data-baseweb="input"] input,
[data-testid="stNumberInput"] input {
    color: #F1F5F9 !important;
}
[data-baseweb="select"] svg { fill: #CBD5E1 !important; }

/* Dropdown menus */
[data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background: #0F1B2D !important;
}
[data-baseweb="popover"] li, [data-baseweb="popover"] li * {
    color: #F1F5F9 !important;
}
[data-baseweb="popover"] li:hover { background: #1E3350 !important; }

/* Multiselect chips */
span[data-baseweb="tag"], div[data-baseweb="tag"] { background-color: #00BFA6 !important; }
span[data-baseweb="tag"] *, div[data-baseweb="tag"] * {
    color: #07111F !important; fill: #07111F !important; font-weight: 650;
}

/* Tabs */
.stTabs [data-baseweb="tab"] { color: #CBD5E1; }
.stTabs [aria-selected="true"] { color: #00D6A3 !important; }

/* Expander, alerts, divider */
[data-testid="stExpander"] {
    background: #172438;
    border: 1px solid #2B3C54;
    border-radius: 10px;
}
hr { border-color: #29384D; }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
ZIP_PATH = BASE / "QuantLab39_models.zip"

# =====================================================
# MULTI-ASSET UNIVERSE
# =====================================================

ASSETS = {
    "Stocks": {
        "Apple": "AAPL", "Tesla": "TSLA", "NVIDIA": "NVDA",
        "Microsoft": "MSFT", "Amazon": "AMZN", "Google": "GOOGL",
        "Meta": "META", "AMD": "AMD", "Netflix": "NFLX",
        "Broadcom": "AVGO", "Oracle": "ORCL", "JPMorgan": "JPM",
        "Walmart": "WMT", "Intel": "INTC", "Qualcomm": "QCOM",
        "IBM": "IBM", "Palantir": "PLTR", "TSMC": "TSM",
        "Alibaba": "BABA", "Reliance": "RELIANCE.NS",
        "TCS": "TCS.NS", "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS"
    },
    "Metals": {
        "Gold": "GC=F", "Silver": "SI=F", "Platinum": "PL=F",
        "Palladium": "PA=F", "Copper": "HG=F"
    },
    "Indices": {
        "S&P 500": "^GSPC", "NASDAQ Composite": "^IXIC",
        "Dow Jones": "^DJI", "Russell 2000": "^RUT",
        "NIFTY 50": "^NSEI", "SENSEX": "^BSESN",
        "FTSE 100": "^FTSE", "DAX": "^GDAXI", "Nikkei 225": "^N225"
    },
    "Forex": {
        "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X", "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
        "NZD/USD": "NZDUSD=X"
    },
    "Crypto": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD",
        "Solana": "SOL-USD", "XRP": "XRP-USD", "BNB": "BNB-USD"
    }
}

ALL_INSTRUMENTS = {n: t for cls in ASSETS.values() for n, t in cls.items()}

PERIOD_MAP = {
    "1 Month": 22, "3 Months": 66, "6 Months": 132,
    "1 Year": 252, "5 Years": 1260
}

# =====================================================
# LOAD EXISTING (PRE-TRAINED) AI MODELS
# =====================================================

@st.cache_resource
def load_models():
    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(BASE)

    configs = list(BASE.glob("**/model_config.json"))
    if not configs:
        return None, None

    config_path = configs[0]
    with open(config_path, "r") as f:
        config = json.load(f)

    return config, config_path.parent


config, model_root = load_models()


@st.cache_resource
def load_stock_model(ticker):
    folder = model_root / ticker
    model_path = folder / "ridge_model.pkl"
    scaler_path = folder / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"No trained AI model available for {ticker}. "
            "Use the ML Lab tab to train one live."
        )
    return joblib.load(model_path), joblib.load(scaler_path)


# =====================================================
# MARKET DATA
# =====================================================

@st.cache_data(ttl=900, show_spinner=False)
def get_market_data(ticker):
    df = None
    for _ in range(2):  # one retry, Yahoo sometimes rate-limits
        try:
            df = yf.download(
                ticker, period="5y", interval="1d",
                auto_adjust=True, progress=False, threads=False
            )
        except Exception:
            df = None
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        raise ValueError(
            f"No market data available for {ticker}. "
            "Yahoo Finance may be rate-limiting, try again in a minute."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col} data")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


# =====================================================
# TECHNICAL INDICATORS
# =====================================================

def add_indicators(df):
    df = df.copy()

    for n in [20, 50, 200]:
        df[f"SMA{n}"] = df["Close"].rolling(n).mean()
        df[f"EMA{n}"] = df["Close"].ewm(span=n, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Bollinger Bands
    middle = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = middle + 2 * std
    df["BB_Lower"] = middle - 2 * std

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Stochastic %K
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14).replace(0, np.nan)

    return df


@st.cache_data(ttl=900, show_spinner=False)
def get_indicators(ticker):
    return add_indicators(get_market_data(ticker))


def technical_score(latest):
    """Simple rule-based score from -5 (very bearish) to +5 (very bullish)."""
    score = 0
    reasons = []

    if latest["RSI"] < 30:
        score += 1
        reasons.append("RSI oversold (+1)")
    elif latest["RSI"] > 70:
        score -= 1
        reasons.append("RSI overbought (-1)")
    else:
        reasons.append("RSI neutral (0)")

    if latest["MACD"] > latest["Signal"]:
        score += 1
        reasons.append("MACD above signal (+1)")
    else:
        score -= 1
        reasons.append("MACD below signal (-1)")

    if latest["Close"] > latest["EMA50"]:
        score += 1
        reasons.append("Price above EMA50 (+1)")
    else:
        score -= 1
        reasons.append("Price below EMA50 (-1)")

    if latest["EMA50"] > latest["EMA200"]:
        score += 1
        reasons.append("EMA50 above EMA200 (+1)")
    else:
        score -= 1
        reasons.append("EMA50 below EMA200 (-1)")

    if latest["Close"] < latest["BB_Lower"]:
        score += 1
        reasons.append("Price below lower Bollinger band (+1)")
    elif latest["Close"] > latest["BB_Upper"]:
        score -= 1
        reasons.append("Price above upper Bollinger band (-1)")
    else:
        reasons.append("Price inside Bollinger bands (0)")

    if score >= 3:
        label = "Strong Bullish"
    elif score >= 1:
        label = "Bullish"
    elif score <= -3:
        label = "Strong Bearish"
    elif score <= -1:
        label = "Bearish"
    else:
        label = "Neutral"

    return score, label, reasons


# =====================================================
# PRE-TRAINED MODEL FEATURES (matches your saved models)
# =====================================================

PRETRAINED_FEATURES = [
    "intradaySpread", "lag1Return", "lag2Return",
    "distSMA5", "distSMA10", "distSMA20", "distEma12",
    "logVolume", "volumeRatioDev", "dailyReturn", "volatility5"
]


def create_features(df):
    df = df.copy()

    df["intradaySpread"] = (df["High"] - df["Low"]) / df["Close"]
    df["dailyReturn"] = df["Close"].pct_change()
    df["lag1Return"] = df["dailyReturn"].shift(1)
    df["lag2Return"] = df["dailyReturn"].shift(2)

    for n in [5, 10, 20]:
        df[f"distSMA{n}"] = df["Close"] / df["Close"].rolling(n).mean() - 1

    df["distEma12"] = df["Close"] / df["Close"].ewm(span=12).mean() - 1
    df["logVolume"] = np.log1p(df["Volume"])

    volume_mean = df["Volume"].rolling(20).mean()
    df["volumeRatioDev"] = df["Volume"] / volume_mean - 1
    df["volatility5"] = df["dailyReturn"].rolling(5).std()

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def predict_stock(ticker, data):
    if model_root is None:
        raise FileNotFoundError("AI model configuration is missing.")

    model, scaler = load_stock_model(ticker)

    feature_data = create_features(data)
    if feature_data.empty:
        raise ValueError("Not enough data for prediction.")

    X = feature_data[PRETRAINED_FEATURES].iloc[[-1]]
    predicted_return = float(model.predict(scaler.transform(X))[0])

    current = float(data["Close"].iloc[-1])
    predicted = current * (1 + predicted_return)

    return current, predicted, predicted_return


# =====================================================
# LIVE ML LAB (works for ANY asset class)
# =====================================================

def strip_tz(obj):
    obj = obj.copy()
    idx = obj.index
    if getattr(idx, "tz", None) is not None:
        obj.index = idx.tz_localize(None)
    return obj


# name -> (factory, kind). kind: "reg" = next-day return, "clf" = P(up)
MODEL_SPECS = {
    "Ridge": (
        lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), "reg"
    ),
    "Lasso": (
        lambda: make_pipeline(
            StandardScaler(), Lasso(alpha=1e-4, max_iter=10000)
        ), "reg"
    ),
    "Random Forest": (
        lambda: RandomForestRegressor(
            n_estimators=150, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        ), "reg"
    ),
    "Gradient Boosting": (
        lambda: GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42
        ), "reg"
    ),
    "Logistic (Up/Down)": (
        lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)
        ), "clf"
    ),
    "RF Classifier": (
        lambda: RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=20,
            n_jobs=-1, random_state=42
        ), "clf"
    ),
    "GB Classifier": (
        lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42
        ), "clf"
    ),
}


def build_ml_dataset(ind, market=None):
    """Features for next-day prediction. Uses only past information."""
    d = ind.copy()
    ret = d["Close"].pct_change()

    f = pd.DataFrame(index=d.index)
    f["ret0"] = ret
    for i in range(1, 6):
        f[f"lag{i}"] = ret.shift(i)

    f["spread"] = (d["High"] - d["Low"]) / d["Close"]
    f["gap"] = d["Open"] / d["Close"].shift() - 1

    for n in [5, 10, 20, 50]:
        f[f"distSMA{n}"] = d["Close"] / d["Close"].rolling(n).mean() - 1

    f["mom10"] = d["Close"].pct_change(10)
    f["mom20"] = d["Close"].pct_change(20)
    f["mom60"] = d["Close"].pct_change(60)
    f["vol5"] = ret.rolling(5).std()
    f["vol20"] = ret.rolling(20).std()
    f["vol_regime"] = f["vol5"] / f["vol20"].replace(0, np.nan)
    f["rsi"] = d["RSI"] / 100
    f["rsi_chg3"] = (d["RSI"] / 100).diff(3)
    f["macd_hist"] = d["MACD_Hist"] / d["Close"]
    f["bb_pos"] = (d["Close"] - d["BB_Lower"]) / (
        d["BB_Upper"] - d["BB_Lower"]
    ).replace(0, np.nan)
    f["atr_pct"] = d["ATR"] / d["Close"]
    f["stoch"] = d["Stoch_K"] / 100
    f["dist_high126"] = d["Close"] / d["Close"].rolling(126).max() - 1
    f["dow"] = d.index.dayofweek / 4

    # Volume features only if the asset really has volume (forex does not)
    if d["Volume"].sum() > 0:
        f["volume_ratio"] = np.log(
            (d["Volume"] + 1) / (d["Volume"].rolling(20).mean() + 1)
        )

    # Market context (S&P 500): how the broad market behaved
    if market is not None and len(market) > 100:
        mc = market["Close"]
        mret = mc.pct_change()
        f["mkt_ret0"] = mret.reindex(d.index, method="ffill")
        f["mkt_ret5"] = mc.pct_change(5).reindex(d.index, method="ffill")
        f["mkt_vol20"] = mret.rolling(20).std().reindex(d.index, method="ffill")

    f = f.replace([np.inf, -np.inf], np.nan).dropna()

    target = ret.shift(-1).reindex(f.index)
    latest_X = f.iloc[[-1]]
    mask = target.notna()

    return f[mask], target[mask], latest_X


def feature_importance(model, cols):
    if model is None:
        return pd.Series(dtype=float)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
    else:
        return pd.Series(dtype=float)
    s_ = pd.Series(vals, index=cols)
    total = s_.sum()
    return (s_ / total if total > 0 else s_).sort_values(ascending=False)


def fit_predict(name, Xtr, ytr, Xte):
    """Returns (model, signal). Signal is a return for regressors and
    P(up) - 0.5 for classifiers, so sign(signal) is always the direction."""
    factory, kind = MODEL_SPECS[name]
    model = factory()

    if kind == "reg":
        model.fit(Xtr, ytr)
        return model, model.predict(Xte)

    labels = (ytr > 0).astype(int)
    if labels.nunique() < 2:
        return None, np.zeros(len(Xte))
    model.fit(Xtr, labels)
    return model, model.predict_proba(Xte)[:, 1] - 0.5


def score_predictions(y_true, signal):
    y_true = pd.Series(y_true)
    signal = pd.Series(np.asarray(signal), index=y_true.index)
    dir_acc = float((np.sign(signal) == np.sign(y_true)).mean())
    up_rate = float((y_true > 0).mean())
    try:
        auc = float(roc_auc_score(y_true > 0, signal))
    except ValueError:
        auc = np.nan
    return dir_acc, max(up_rate, 1 - up_rate), auc


def walk_forward(name, X, y, n_splits=5):
    """Expanding-window validation: train on the past, test on the next block."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    parts, accs, edges = [], [], []
    for tr, te in tscv.split(X):
        _, sig = fit_predict(name, X.iloc[tr], y.iloc[tr], X.iloc[te])
        sig = pd.Series(sig, index=X.index[te])
        acc, base, _ = score_predictions(y.iloc[te], sig)
        parts.append(sig)
        accs.append(acc)
        edges.append(acc - base)
    return pd.concat(parts), accs, edges


@st.cache_data(ttl=900, show_spinner=False)
def run_ml_experiment(ticker, model_name, test_frac):
    raw = strip_tz(get_market_data(ticker))
    try:
        market = strip_tz(get_market_data("^GSPC"))
    except Exception:
        market = None

    ind = add_indicators(raw)
    X, y, latest_X = build_ml_dataset(ind, market)

    if len(X) < 300:
        raise ValueError("Not enough history to train a reliable model.")

    kind = MODEL_SPECS[model_name][1]

    # 1) Single holdout split (last test_frac of the history)
    split = int(len(X) * (1 - test_frac))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y.iloc[:split], y.iloc[split:]

    _, sig = fit_predict(model_name, Xtr, ytr, Xte)
    pred = pd.Series(sig, index=Xte.index)
    dir_acc, baseline, auc = score_predictions(yte, pred)

    if kind == "reg":
        mae = float(mean_absolute_error(yte, pred))
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        r2 = float(r2_score(yte, pred))
    else:
        mae = rmse = r2 = np.nan

    # 2) Walk-forward validation (more trustworthy than one split)
    oos, accs, edges = walk_forward(model_name, X, y)
    y_wf = y.reindex(oos.index)

    # 3) Final model on ALL data for the latest forecast
    final, fsig = fit_predict(model_name, X, y, latest_X)
    current = float(raw["Close"].iloc[-1])

    if kind == "reg":
        next_ret = float(fsig[0])
        p_up = np.nan
        next_price = current * (1 + next_ret)
    else:
        next_ret = np.nan
        p_up = float(fsig[0] + 0.5)
        next_price = np.nan

    return {
        "model": model_name,
        "kind": kind,
        "mae": mae, "rmse": rmse, "r2": r2,
        "dir_acc": dir_acc, "baseline": baseline, "auc": auc,
        "wf_acc": float(np.mean(accs)),
        "wf_std": float(np.std(accs)),
        "wf_edge": float(np.mean(edges)),
        "wf_beat": int(sum(e > 0 for e in edges)),
        "wf_folds": len(edges),
        "y_test": yte, "pred": pred,
        "y_wf": y_wf, "pred_wf": oos,
        "importance": feature_importance(final, X.columns),
        "current": current,
        "next_ret": next_ret, "p_up": p_up, "next_price": next_price,
        "train_rows": len(Xtr), "test_rows": len(Xte),
        "n_features": X.shape[1],
    }


# =====================================================
# PERFORMANCE / BACKTEST HELPERS
# =====================================================

def perf_stats(r, rf=0.0):
    r = pd.Series(r).dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    n = len(r)
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (252 / n) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252 - rf) / vol if vol > 0 else 0.0
    max_dd = (eq / eq.cummax() - 1).min()
    active = r[r != 0]
    win = float((active > 0).mean()) if len(active) else 0.0
    return {
        "Total Return": total, "CAGR": cagr, "Volatility": vol,
        "Sharpe": sharpe, "Max Drawdown": max_dd, "Win Rate": win
    }


def backtest(actual, pred, threshold, allow_short, cost_bps):
    """Position is decided at day t using prediction for day t+1. No look-ahead."""
    pos = pd.Series(0.0, index=pred.index)
    pos[pred > threshold] = 1.0
    if allow_short:
        pos[pred < -threshold] = -1.0

    trades = pos.diff().abs().fillna(pos.abs())
    cost = trades * (cost_bps / 10000)
    strat = pos * actual - cost
    return strat, pos, int((trades > 0).sum())


def style_fig(fig, height=450, title=None):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#F1F5F9", size=16)) if title else None,
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#111C2E",
        font=dict(color="#E2E8F0"),
        height=height,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(gridcolor="#22324A", zerolinecolor="#2B3C54",
                     tickfont=dict(color="#CBD5E1"))
    fig.update_yaxes(gridcolor="#22324A", zerolinecolor="#2B3C54",
                     tickfont=dict(color="#CBD5E1"))
    return fig


# =====================================================
# MARKET CHART
# =====================================================

def show_chart(data, name):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=["Candlestick & Moving Averages", "RSI (14)", "MACD"]
    )

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price",
        increasing_line_color="#00D6A3", decreasing_line_color="#FF5263"
    ), row=1, col=1)

    for col, color in [
        ("EMA20", "#00D6A3"), ("EMA50", "#F5B942"), ("EMA200", "#A78BFA"),
        ("BB_Upper", "#64748B"), ("BB_Lower", "#64748B")
    ]:
        fig.add_trace(go.Scatter(
            x=data.index, y=data[col], name=col,
            line=dict(color=color, width=1.3)
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"], name="RSI", line=dict(color="#00D6A3")
    ), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#FF5263", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00D6A3", row=2, col=1)

    fig.add_trace(go.Bar(
        x=data.index, y=data["MACD_Hist"], name="Histogram",
        marker_color=np.where(data["MACD_Hist"] >= 0, "#00D6A3", "#FF5263")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["MACD"], name="MACD", line=dict(color="#3B82F6")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Signal"], name="Signal", line=dict(color="#F5B942")
    ), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"{name} | Technical Analysis",
                   font=dict(color="#F1F5F9", size=16)),
        template="plotly_dark", paper_bgcolor="#0B1220",
        plot_bgcolor="#111C2E", font=dict(color="#E2E8F0"),
        height=820, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor="#22324A")
    fig.update_yaxes(gridcolor="#22324A")
    fig.update_annotations(font=dict(color="#CBD5E1"))
    plot_chart(fig, key="tech_chart", **WIDE)


# =====================================================
# APP HEADER
# =====================================================

st.title("📈 QUANTLAB 39")
st.caption("AI Powered Multi-Asset Market Intelligence Platform")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Market Controls")

asset_class = st.sidebar.selectbox("Select Market", list(ASSETS.keys()))
asset_names = list(ASSETS[asset_class].keys())
selected_name = st.sidebar.selectbox("Select Instrument", asset_names)
selected_ticker = ASSETS[asset_class][selected_name]

period = st.sidebar.selectbox(
    "Chart History", list(PERIOD_MAP.keys()), index=3
)

rf_rate = st.sidebar.number_input(
    "Risk-free rate (% per year)", 0.0, 15.0, 4.0, 0.25
) / 100

st.sidebar.markdown("---")
st.sidebar.caption(f"Market: {asset_class}")
st.sidebar.caption(f"Instrument: {selected_ticker}")
st.sidebar.caption("Data source: Yahoo Finance")
st.sidebar.caption("App version: v4 (15 Nov 2026)")

if st.sidebar.button("🧹 Clear cached data"):
    st.cache_data.clear()
    st.rerun()

# =====================================================
# TABS
# =====================================================

tabs = st.tabs([
    "📊 AI Predictions",
    "📈 Technical Analysis",
    "🧪 ML Lab",
    "⚠️ Risk & Monte Carlo",
    "💼 Portfolio",
    "🌍 Market Explorer",
    "ℹ️ About"
])
tab_pred, tab_tech, tab_ml, tab_risk, tab_port, tab_explore, tab_about = tabs

# =====================================================
# TAB 1 - AI PREDICTIONS (pre-trained Ridge models)
# =====================================================

with tab_pred:
    st.subheader("AI Stock Prediction Center")
    st.info(
        "Uses your saved Ridge Regression models when available. For stocks "
        "without a saved model, a Ridge model is trained live automatically. "
        "For forex, metals, indices or crypto use the 🧪 ML Lab tab."
    )

    stock_lookup = ASSETS["Stocks"]

    selected_prediction_names = st.multiselect(
        "Select stocks for AI prediction",
        list(stock_lookup.keys()),
        default=["Apple", "NVIDIA", "Microsoft"]
    )

    def run_predictions(names):
        results, errors = [], []

        with st.spinner("Running AI models..."):
            for stock_name in names:
                ticker = stock_lookup[stock_name]
                try:
                    data = get_market_data(ticker)

                    try:
                        current, predicted, ret = predict_stock(ticker, data)
                        source = "Saved Ridge"
                    except FileNotFoundError:
                        live = run_ml_experiment(ticker, "Ridge", 0.2)
                        current = live["current"]
                        predicted = live["next_price"]
                        ret = live["next_ret"]
                        source = "Live Ridge (trained now)"

                    vol = float(data["Close"].pct_change().tail(60).std())
                    z = ret / vol if vol > 0 else 0.0

                    if abs(z) < 0.05:
                        view = "Neutral"
                    elif abs(z) < 0.15:
                        view = "Slightly Bullish" if z > 0 else "Slightly Bearish"
                    else:
                        view = "Bullish" if z > 0 else "Bearish"

                    results.append({
                        "Stock": stock_name,
                        "Ticker": ticker,
                        "Current Price": current,
                        "Predicted Price": predicted,
                        "Expected Return (%)": ret * 100,
                        "Signal (σ)": z,
                        "View": view,
                        "Model Source": source,
                        "Data As Of": data.index[-1].strftime("%Y-%m-%d")
                    })
                except Exception as e:
                    errors.append(f"{stock_name}: {e}")

        st.session_state["ai_results"] = results
        st.session_state["ai_errors"] = errors

    if st.button("🚀 Generate AI Predictions", type="primary",
                 key="prediction_button"):
        run_predictions(selected_prediction_names)
    elif "ai_results" not in st.session_state and selected_prediction_names:
        run_predictions(selected_prediction_names)

    if st.session_state.get("ai_results"):
        result_df = pd.DataFrame(st.session_state["ai_results"])

        st.subheader("Prediction Summary")
        st.dataframe(
            result_df.style.format({
                "Current Price": "{:,.4f}",
                "Predicted Price": "{:,.4f}",
                "Expected Return (%)": "{:+.3f}%",
                "Signal (σ)": "{:+.2f}"
            }),
            hide_index=True,
            **WIDE
        )

        fig = go.Figure(go.Bar(
            x=result_df["Stock"],
            y=result_df["Expected Return (%)"],
            marker_color=np.where(
                result_df["Expected Return (%)"] >= 0, "#00D6A3", "#FF5263"
            ),
            text=result_df["Expected Return (%)"].map(lambda v: f"{v:+.3f}%"),
            textposition="outside",
            textfont=dict(color="#F1F5F9")
        ))
        plot_chart(
            style_fig(fig, 380, "Expected Next-Day Return (%)"),
            key="pred_bar", **WIDE
        )

        st.caption(
            "Signal (σ) = predicted return divided by the stock's recent daily "
            "volatility. Values near 0 mean the model sees no meaningful edge. "
            "Next-day moves are very hard to predict, so treat this as one input only."
        )

        st.download_button(
            "📥 Download Predictions",
            result_df.to_csv(index=False),
            file_name="quantlab39_predictions.csv",
            mime="text/csv"
        )

    for error in st.session_state.get("ai_errors", []):
        st.warning(error)

    if not selected_prediction_names:
        st.caption("Select at least one stock to continue.")

# =====================================================
# TAB 2 - TECHNICAL ANALYSIS
# =====================================================

with tab_tech:
    st.subheader("Advanced Technical Analysis")
    st.write(f"Analyzing: **{selected_name} ({selected_ticker})**")

    try:
        with st.spinner("Loading market data..."):
            full = get_indicators(selected_ticker)
        data = full.tail(PERIOD_MAP[period])
        latest = data.iloc[-1]

        a, b, c, d = st.columns(4)
        a.metric("Latest Price", f"{latest['Close']:,.4f}")
        b.metric("RSI (14)", f"{latest['RSI']:.2f}")
        c.metric("MACD", f"{latest['MACD']:.4f}")
        d.metric("ATR (14)", f"{latest['ATR']:.4f}")

        score, label, reasons = technical_score(latest)
        gauge_color = (
            "#00D6A3" if score > 0 else "#FF5263" if score < 0 else "#94A3B8"
        )

        g1, g2 = st.columns([1, 1])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={"text": f"Technical Score: {label}"},
                gauge={
                    "axis": {"range": [-5, 5]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [-5, -2], "color": "#7F1D1D"},
                        {"range": [-2, 2], "color": "#334155"},
                        {"range": [2, 5], "color": "#064E3B"},
                    ],
                }
            ))
            plot_chart(style_fig(gauge, 300), key="tech_gauge", **WIDE)

        with g2:
            st.markdown("**Why this score?**")
            for r in reasons:
                st.write(f"- {r}")
            st.caption(
                "Rule-based observation. Not a guaranteed trade signal."
            )

        show_chart(data, selected_name)

        with st.expander("📄 View / download indicator data"):
            st.dataframe(data.tail(100).round(4), **WIDE)
            st.download_button(
                "📥 Download indicators CSV",
                data.round(6).to_csv(),
                file_name=f"{selected_ticker}_indicators.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(str(e))

# =====================================================
# TAB 3 - ML LAB
# =====================================================

with tab_ml:
    st.subheader("🧪 ML Lab: Train, Compare & Backtest Models Live")
    st.caption(
        "Regression models predict the next-day return. Classifiers predict "
        "the probability that tomorrow closes up. Every model is checked on a "
        "holdout period AND with walk-forward validation (5 expanding folds)."
    )

    names_list = list(ALL_INSTRUMENTS.keys())
    default_idx = names_list.index(selected_name) if selected_name in names_list else 0

    c1, c2 = st.columns([3, 1])
    ml_name = c1.selectbox("Instrument", names_list, index=default_idx, key="ml_name")
    test_pct = c2.slider("Holdout %", 10, 40, 20, 5, key="ml_test") / 100

    ml_models = st.multiselect(
        "Models to compare", list(MODEL_SPECS.keys()),
        default=["Ridge", "Random Forest", "Logistic (Up/Down)"],
        key="ml_models"
    )

    if st.button("🧠 Train & Evaluate", type="primary", key="ml_train"):
        if not ml_models:
            st.warning("Select at least one model.")
        else:
            ticker = ALL_INSTRUMENTS[ml_name]
            res, errs = {}, []
            with st.spinner("Training models (holdout + walk-forward)..."):
                for m in ml_models:
                    try:
                        res[m] = run_ml_experiment(ticker, m, test_pct)
                    except Exception as e:
                        errs.append(f"{m}: {e}")
            st.session_state["ml_out"] = {
                "name": ml_name, "ticker": ticker, "results": res
            }
            st.session_state["ml_errs"] = errs

    for e in st.session_state.get("ml_errs", []):
        st.warning(e)

    out = st.session_state.get("ml_out")
    if out and out["results"]:
        results = out["results"]
        st.markdown(f"### Results for **{out['name']}** ({out['ticker']})")

        table = pd.DataFrame([{
            "Model": m,
            "Type": "Classifier" if r["kind"] == "clf" else "Regression",
            "Holdout Acc (%)": r["dir_acc"] * 100,
            "Baseline (%)": r["baseline"] * 100,
            "Edge (pp)": (r["dir_acc"] - r["baseline"]) * 100,
            "Walk-Fwd Acc (%)": r["wf_acc"] * 100,
            "WF Edge (pp)": r["wf_edge"] * 100,
            "Folds > Baseline": f"{r['wf_beat']}/{r['wf_folds']}",
            "AUC": r["auc"],
            "R²": r["r2"],
            "Next-Day Return (%)": r["next_ret"] * 100,
            "P(Up) (%)": r["p_up"] * 100,
        } for m, r in results.items()])

        st.dataframe(
            table.style.format({
                "Holdout Acc (%)": "{:.2f}", "Baseline (%)": "{:.2f}",
                "Edge (pp)": "{:+.2f}", "Walk-Fwd Acc (%)": "{:.2f}",
                "WF Edge (pp)": "{:+.2f}", "AUC": "{:.3f}", "R²": "{:.4f}",
                "Next-Day Return (%)": "{:+.3f}", "P(Up) (%)": "{:.1f}"
            }, na_rep="-"),
            hide_index=True,
            **WIDE
        )

        fig = go.Figure()
        fig.add_bar(x=table["Model"], y=table["Holdout Acc (%)"],
                    name="Holdout", marker_color="#00D6A3")
        fig.add_bar(x=table["Model"], y=table["Walk-Fwd Acc (%)"],
                    name="Walk-forward", marker_color="#3B82F6")
        fig.add_bar(x=table["Model"], y=table["Baseline (%)"],
                    name="Always-majority baseline", marker_color="#64748B")
        fig.update_layout(barmode="group", yaxis_range=[35, 70])
        plot_chart(
            style_fig(fig, 360, "Directional Accuracy vs Baseline"),
            key="ml_dir_bar", **WIDE
        )

        best = table.sort_values("WF Edge (pp)", ascending=False).iloc[0]
        if best["WF Edge (pp)"] > 0.5:
            auc_txt = f"{best['AUC']:.3f}" if pd.notna(best["AUC"]) else "n/a"
            st.success(
                f"Best by walk-forward: **{best['Model']}**, "
                f"{best['WF Edge (pp)']:+.2f} pp over baseline "
                f"(beat baseline in {best['Folds > Baseline']} folds, AUC {auc_txt})."
            )
        else:
            st.warning(
                "No model beats the always-majority baseline consistently on "
                "this asset. For daily data this is a normal, honest result. "
                "Try another asset or a different model type."
            )

        st.markdown("### 🔍 Inspect one model")
        chosen = st.selectbox("Model", list(results.keys()), key="ml_inspect")
        r = results[chosen]

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"{r['current']:,.4f}")
        if r["kind"] == "reg":
            m2.metric("Forecast Price", f"{r['next_price']:,.4f}",
                      delta=f"{r['next_ret']*100:+.2f}%")
        else:
            m2.metric("P(Up) tomorrow", f"{r['p_up']*100:.1f}%",
                      delta=f"{(r['p_up']-0.5)*100:+.1f} pp vs 50%")
        m3.metric("Train / Test rows", f"{r['train_rows']} / {r['test_rows']}")
        st.caption(
            f"{r['n_features']} features | walk-forward accuracy "
            f"{r['wf_acc']*100:.1f}% ± {r['wf_std']*100:.1f}% across folds"
        )

        i1, i2 = st.columns(2)
        with i1:
            imp = r["importance"].head(10).sort_values()
            if len(imp):
                fig = px.bar(x=imp.values, y=imp.index, orientation="h")
                fig.update_traces(marker_color="#00D6A3")
                plot_chart(
                    style_fig(fig, 380, "Top 10 Feature Importance"),
                    key="ml_imp", **WIDE
                )
        with i2:
            if r["kind"] == "reg":
                tail = 120
                fig = go.Figure()
                fig.add_scatter(x=r["y_test"].index[-tail:],
                                y=r["y_test"].iloc[-tail:] * 100,
                                name="Actual", line=dict(color="#E2E8F0", width=1))
                fig.add_scatter(x=r["pred"].index[-tail:],
                                y=r["pred"].iloc[-tail:] * 100,
                                name="Predicted", line=dict(color="#F5B942", width=2))
                plot_chart(
                    style_fig(fig, 380, "Actual vs Predicted Return (%), last 120 test days"),
                    key="ml_avp", **WIDE
                )
            else:
                try:
                    cal = pd.DataFrame({
                        "p": r["pred"] + 0.5,
                        "up": (r["y_test"] > 0).astype(float)
                    })
                    cal["bucket"] = pd.qcut(cal["p"], 5, labels=False,
                                            duplicates="drop")
                    g = cal.groupby("bucket")["up"].mean() * 100
                    fig = go.Figure(go.Bar(
                        x=[f"Q{int(i)+1}" for i in g.index], y=g.values,
                        marker_color="#00D6A3",
                        text=[f"{v:.0f}%" for v in g.values],
                        textposition="outside"
                    ))
                    fig.add_hline(y=50, line_dash="dash", line_color="#F5B942")
                    fig.update_layout(yaxis_range=[0, 100])
                    plot_chart(
                        style_fig(fig, 380, "Actual up-days by predicted P(Up) bucket (Q5 = most bullish)"),
                        key="ml_cal", **WIDE
                    )
                except Exception:
                    st.caption("Not enough variation in predictions for a calibration chart.")

        st.markdown("### 📉 Strategy Backtest")
        src = st.radio(
            "Backtest data",
            ["Holdout test period", "Walk-forward (all out-of-sample)"],
            horizontal=True, key="bt_src"
        )
        if src.startswith("Holdout"):
            y_s, p_s = r["y_test"], r["pred"]
        else:
            y_s, p_s = r["y_wf"], r["pred_wf"]

        b1, b2, b3 = st.columns(3)
        if r["kind"] == "reg":
            thr = b1.slider("Signal threshold (%)", 0.0, 1.0, 0.05, 0.05,
                            key="bt_thr") / 100
        else:
            thr = b1.slider("Min. probability edge (pp above 50%)", 0.0, 10.0,
                            2.0, 0.5, key="bt_thr_clf") / 100
        short = b2.checkbox("Allow short selling", value=False, key="bt_short")
        cost = b3.slider("Cost per trade (bps)", 0, 30, 5, key="bt_cost")

        strat, pos, n_trades = backtest(y_s, p_s, thr, short, cost)

        eq = pd.DataFrame({
            "ML Strategy": (1 + strat).cumprod(),
            "Buy & Hold": (1 + y_s).cumprod()
        })
        fig = px.line(eq, color_discrete_sequence=["#00D6A3", "#64748B"])
        plot_chart(
            style_fig(fig, 400, "Equity Curve (start = 1.0)"),
            key="bt_eq", **WIDE
        )

        s_stats = perf_stats(strat, rf_rate)
        b_stats = perf_stats(y_s, rf_rate)

        def _fmt(stats):
            out_ = {}
            for key, val in stats.items():
                out_[key] = f"{val:.2f}" if key == "Sharpe" else f"{val*100:.2f}%"
            return out_

        cmp_df = pd.DataFrame(
            {"ML Strategy": _fmt(s_stats), "Buy & Hold": _fmt(b_stats)}
        ).astype(str)
        st.dataframe(cmp_df, **WIDE)
        st.caption(f"Number of trades: {n_trades}. Past performance does not "
                   "guarantee future results.")

# =====================================================
# TAB 4 - RISK & MONTE CARLO
# =====================================================

with tab_risk:
    st.subheader("⚠️ Risk Analytics & Monte Carlo Simulation")
    st.write(f"Instrument: **{selected_name} ({selected_ticker})**")

    lookback = st.select_slider(
        "Lookback window", options=list(PERIOD_MAP.keys()),
        value="1 Year", key="risk_lb"
    )

    try:
        rdata = get_market_data(selected_ticker).tail(PERIOD_MAP[lookback] + 1)
        rets = rdata["Close"].pct_change().dropna()

        ann_vol = rets.std() * np.sqrt(252)
        ann_ret = rets.mean() * 252
        sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
        equity = (1 + rets).cumprod()
        dd = equity / equity.cummax() - 1
        var95 = np.percentile(rets, 5)
        var99 = np.percentile(rets, 1)
        cvar95 = rets[rets <= var95].mean()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Annual Volatility", f"{ann_vol*100:.2f}%")
        k2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        k3.metric("Max Drawdown", f"{dd.min()*100:.2f}%")
        k4.metric("Skew / Kurtosis", f"{rets.skew():.2f} / {rets.kurt():.2f}")

        k5, k6, k7, k8, k9 = st.columns(5)
        k5.metric("VaR 95% (1d)", f"{var95*100:.2f}%")
        k6.metric("VaR 99% (1d)", f"{var99*100:.2f}%")
        k7.metric("CVaR 95%", f"{cvar95*100:.2f}%")
        k8.metric("Best Day", f"{rets.max()*100:+.1f}%")
        k9.metric("Worst Day", f"{rets.min()*100:+.1f}%")

        r1, r2 = st.columns(2)
        with r1:
            fig = go.Figure(go.Scatter(
                x=dd.index, y=dd * 100, fill="tozeroy",
                line=dict(color="#FF5263")
            ))
            plot_chart(
                style_fig(fig, 340, "Drawdown (%)"), key="risk_dd", **WIDE
            )
        with r2:
            fig = go.Figure(go.Histogram(
                x=rets * 100, nbinsx=50, marker_color="#00D6A3"
            ))
            fig.add_vline(x=var95 * 100, line_dash="dash",
                          line_color="#FF5263", annotation_text="VaR 95%")
            plot_chart(
                style_fig(fig, 340, "Daily Return Distribution (%)"),
                key="risk_hist", **WIDE
            )

        st.markdown("### 🎲 Monte Carlo Price Simulation")
        st.caption(
            "Geometric Brownian Motion using the lookback window's mean and "
            "volatility of log returns. This is a statistical what-if, not a forecast."
        )

        s1, s2 = st.columns(2)
        horizon = s1.slider("Days ahead", 10, 252, 60, key="mc_h")
        sims = s2.slider("Simulations", 200, 3000, 1000, 100, key="mc_n")

        log_r = np.log1p(rets)
        mu, sigma = log_r.mean(), log_r.std()
        s0 = float(rdata["Close"].iloc[-1])

        rng = np.random.default_rng(42)
        z = rng.standard_normal((sims, horizon))
        paths = s0 * np.exp(np.cumsum(mu + sigma * z, axis=1))
        paths = np.hstack([np.full((sims, 1), s0), paths])

        p5, p25, p50, p75, p95 = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
        xs = np.arange(horizon + 1)

        fig = go.Figure()
        for i in range(min(30, sims)):
            fig.add_scatter(x=xs, y=paths[i], mode="lines",
                            line=dict(color="rgba(148,163,184,0.15)", width=1),
                            showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=xs, y=p95, line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=xs, y=p5, fill="tonexty",
                        fillcolor="rgba(0,214,163,0.12)", line=dict(width=0),
                        name="5-95% range")
        fig.add_scatter(x=xs, y=p75, line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=xs, y=p25, fill="tonexty",
                        fillcolor="rgba(0,214,163,0.25)", line=dict(width=0),
                        name="25-75% range")
        fig.add_scatter(x=xs, y=p50, name="Median",
                        line=dict(color="#F5B942", width=2.5))
        plot_chart(
            style_fig(fig, 430, f"{horizon}-day simulated price paths"),
            key="mc_fan", **WIDE
        )

        final = paths[:, -1]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Price", f"{np.median(final):,.4f}",
                  delta=f"{(np.median(final)/s0-1)*100:+.2f}%")
        m2.metric("5th Percentile", f"{p5[-1]:,.4f}")
        m3.metric("95th Percentile", f"{p95[-1]:,.4f}")
        m4.metric("Probability of Profit", f"{(final > s0).mean()*100:.1f}%")

    except Exception as e:
        st.error(str(e))

# =====================================================
# TAB 5 - PORTFOLIO
# =====================================================

with tab_port:
    st.subheader("💼 Portfolio Suite")
    p_calc, p_track, p_opt, p_corr = st.tabs([
        "🧮 Calculator", "📒 Position Tracker",
        "🎯 Optimizer", "🔗 Correlation"
    ])

    # ---- Calculator ----
    with p_calc:
        st.caption("Hypothetical profit or loss of one position.")

        portfolio_class = st.selectbox(
            "Asset Class", list(ASSETS.keys()), key="portfolio_class"
        )
        portfolio_name = st.selectbox(
            "Select Instrument", list(ASSETS[portfolio_class].keys()),
            key="portfolio_name"
        )
        portfolio_ticker = ASSETS[portfolio_class][portfolio_name]

        entry_price = st.number_input(
            "Your Entry Price", min_value=0.000001, value=100.0,
            format="%.6f", key="calc_entry"
        )
        quantity = st.number_input(
            "Quantity / Units", min_value=0.000001, value=1.0,
            format="%.6f", key="calc_qty"
        )

        if st.button("Calculate Portfolio", type="primary", key="portfolio_button"):
            try:
                df = get_market_data(portfolio_ticker)
                current = float(df["Close"].iloc[-1])
                invested = entry_price * quantity
                current_value = current * quantity
                profit = current_value - invested
                returns = profit / invested * 100

                a, b, c = st.columns(3)
                a.metric("Invested Value", f"{invested:,.4f}")
                b.metric("Current Value", f"{current_value:,.4f}")
                c.metric("Profit / Loss", f"{profit:+,.4f}",
                         delta=f"{returns:+.2f}%")
                st.caption(f"Instrument: {portfolio_name} ({portfolio_ticker})")
            except Exception as e:
                st.error(str(e))

    # ---- Tracker ----
    with p_track:
        st.caption("Add multiple positions and see total P/L and allocation.")

        if "positions" not in st.session_state:
            st.session_state["positions"] = []

        t1, t2, t3 = st.columns(3)
        pos_name = t1.selectbox("Instrument", list(ALL_INSTRUMENTS.keys()),
                                key="trk_name")
        pos_entry = t2.number_input("Entry price", min_value=0.000001,
                                    value=100.0, format="%.4f", key="trk_entry")
        pos_qty = t3.number_input("Quantity", min_value=0.000001,
                                  value=1.0, format="%.4f", key="trk_qty")

        bcol1, bcol2 = st.columns(2)
        if bcol1.button("➕ Add position", key="trk_add"):
            st.session_state["positions"].append({
                "Instrument": pos_name,
                "Ticker": ALL_INSTRUMENTS[pos_name],
                "Entry": pos_entry,
                "Qty": pos_qty
            })
        if bcol2.button("🗑️ Clear all", key="trk_clear"):
            st.session_state["positions"] = []

        if st.session_state["positions"]:
            rows = []
            for p in st.session_state["positions"]:
                try:
                    price = float(get_market_data(p["Ticker"])["Close"].iloc[-1])
                except Exception:
                    price = np.nan
                invested = p["Entry"] * p["Qty"]
                value = price * p["Qty"]
                rows.append({
                    "Instrument": p["Instrument"], "Ticker": p["Ticker"],
                    "Entry": p["Entry"], "Qty": p["Qty"], "Price": price,
                    "Invested": invested, "Value": value,
                    "P/L": value - invested,
                    "P/L (%)": (value / invested - 1) * 100
                })
            pdf = pd.DataFrame(rows)

            tot_inv, tot_val = pdf["Invested"].sum(), pdf["Value"].sum()
            a, b, c = st.columns(3)
            a.metric("Total Invested", f"{tot_inv:,.2f}")
            b.metric("Total Value", f"{tot_val:,.2f}")
            c.metric("Total P/L", f"{tot_val - tot_inv:+,.2f}",
                     delta=f"{(tot_val/tot_inv-1)*100:+.2f}%" if tot_inv else None)

            st.dataframe(
                pdf.style.format({
                    "Entry": "{:,.4f}", "Qty": "{:,.4f}", "Price": "{:,.4f}",
                    "Invested": "{:,.2f}", "Value": "{:,.2f}",
                    "P/L": "{:+,.2f}", "P/L (%)": "{:+.2f}%"
                }),
                **WIDE
            )

            valid = pdf.dropna(subset=["Value"])
            if not valid.empty:
                fig = px.pie(valid, names="Instrument", values="Value", hole=0.45)
                plot_chart(style_fig(fig, 360, "Allocation by Value"),
                                key="trk_pie", **WIDE)

            st.download_button("📥 Download positions", pdf.to_csv(index=False),
                               file_name="positions.csv", mime="text/csv")
            st.caption("Note: values are in each asset's own currency (USD, INR, etc.).")

    # ---- Optimizer ----
    with p_opt:
        st.caption(
            "Monte Carlo efficient frontier: thousands of random weight "
            "combinations, find max Sharpe and min volatility portfolios."
        )

        opt_sel = st.multiselect(
            "Choose 3 or more instruments",
            list(ALL_INSTRUMENTS.keys()),
            default=["Apple", "Gold", "S&P 500", "Bitcoin"],
            key="opt_sel"
        )
        n_port = st.slider("Random portfolios (for the cloud)", 1000, 20000,
                           5000, 1000, key="opt_n")
        max_w = st.slider("Max weight per asset (%)", 20, 100, 60, 5,
                          key="opt_maxw") / 100

        if st.button("🎯 Optimize Portfolio", type="primary", key="opt_btn"):
            if len(opt_sel) < 3:
                st.warning("Select at least 3 instruments.")
            else:
                try:
                    closes = {}
                    for n in opt_sel:
                        closes[n] = strip_tz(
                            get_market_data(ALL_INSTRUMENTS[n])
                        )["Close"]
                    prices = pd.concat(closes, axis=1).ffill(limit=3).dropna()
                    rets = prices.pct_change().dropna().tail(756)

                    if len(rets) < 60:
                        raise ValueError("Not enough overlapping history.")

                    mu = rets.mean().values * 252
                    cov = rets.cov().values * 252
                    k = len(opt_sel)
                    cap = max(max_w, 1.0 / k + 1e-6)

                    # Random cloud (for the picture)
                    rng = np.random.default_rng(7)
                    W = rng.dirichlet(np.ones(k), size=n_port)
                    p_ret = W @ mu
                    p_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))
                    p_sh = (p_ret - rf_rate) / p_vol

                    # Exact optimisation (SLSQP)
                    def port_vol(w):
                        return float(np.sqrt(w @ cov @ w))

                    def neg_sharpe(w):
                        v = port_vol(w)
                        return -((w @ mu - rf_rate) / v) if v > 0 else 0.0

                    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
                    bnds = [(0.0, cap)] * k
                    w0 = np.ones(k) / k

                    r_sh = minimize(neg_sharpe, w0, method="SLSQP",
                                    bounds=bnds, constraints=cons)
                    r_mv = minimize(port_vol, w0, method="SLSQP",
                                    bounds=bnds, constraints=cons)

                    def clean(w):
                        w = np.clip(w, 0, None)
                        return w / w.sum()

                    w_sh = clean(r_sh.x) if r_sh.success else W[p_sh.argmax()]
                    w_mv = clean(r_mv.x) if r_mv.success else W[p_vol.argmin()]

                    def stats(w):
                        v = port_vol(w)
                        return float(w @ mu), v, (float(w @ mu) - rf_rate) / v

                    ret_sh, vol_sh, sh_sh = stats(w_sh)
                    ret_mv, vol_mv, sh_mv = stats(w_mv)

                    fig = go.Figure()
                    fig.add_scatter(
                        x=p_vol * 100, y=p_ret * 100, mode="markers",
                        marker=dict(color=p_sh, colorscale="Viridis", size=4,
                                    colorbar=dict(title="Sharpe", x=1.02)),
                        name="Random portfolios"
                    )
                    fig.add_scatter(
                        x=[vol_sh * 100], y=[ret_sh * 100], mode="markers",
                        name="Max Sharpe",
                        marker=dict(color="#F5B942", size=20, symbol="star",
                                    line=dict(color="#FFFFFF", width=1.5))
                    )
                    fig.add_scatter(
                        x=[vol_mv * 100], y=[ret_mv * 100], mode="markers",
                        name="Min Volatility",
                        marker=dict(color="#FF5263", size=16, symbol="diamond",
                                    line=dict(color="#FFFFFF", width=1.5))
                    )
                    fig.update_layout(
                        xaxis_title="Volatility (%)", yaxis_title="Return (%)",
                        legend=dict(orientation="h", yanchor="top", y=-0.18,
                                    xanchor="left", x=0)
                    )
                    plot_chart(
                        style_fig(fig, 500, "Efficient Frontier"),
                        key="opt_frontier", **WIDE
                    )

                    w_df = pd.DataFrame({
                        "Instrument": opt_sel,
                        "Max Sharpe (%)": w_sh * 100,
                        "Min Volatility (%)": w_mv * 100
                    })
                    st.dataframe(
                        w_df.style.format({
                            "Max Sharpe (%)": "{:.1f}",
                            "Min Volatility (%)": "{:.1f}"
                        }),
                        hide_index=True,
                        **WIDE
                    )

                    st.markdown("**Max Sharpe portfolio**")
                    a1, a2, a3 = st.columns(3)
                    a1.metric("Return", f"{ret_sh*100:.1f}%")
                    a2.metric("Volatility", f"{vol_sh*100:.1f}%")
                    a3.metric("Sharpe", f"{sh_sh:.2f}")

                    st.markdown("**Min Volatility portfolio**")
                    b1_, b2_, b3_ = st.columns(3)
                    b1_.metric("Return", f"{ret_mv*100:.1f}%")
                    b2_.metric("Volatility", f"{vol_mv*100:.1f}%")
                    b3_.metric("Sharpe", f"{sh_mv:.2f}")

                    st.caption(
                        "Weights come from a real optimiser (SLSQP) with your max-weight "
                        "limit; the dots are random portfolios for context. Based on up to "
                        "3 years of overlapping history, so future results can differ."
                    )
                except Exception as e:
                    st.error(str(e))

    # ---- Correlation ----
    with p_corr:
        corr_sel = st.multiselect(
            "Select instruments",
            list(ALL_INSTRUMENTS.keys()),
            default=["Apple", "NVIDIA", "Gold", "S&P 500", "Bitcoin"],
            key="corr_sel"
        )
        if len(corr_sel) >= 2:
            try:
                closes = {
                    n: strip_tz(get_market_data(ALL_INSTRUMENTS[n]))["Close"]
                    for n in corr_sel
                }
                prices = pd.concat(closes, axis=1).ffill(limit=3).dropna()
                corr = prices.pct_change().dropna().tail(252).corr()
                fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                                color_continuous_scale="RdYlGn")
                plot_chart(
                    style_fig(fig, 520, "Return Correlation (last ~1 year)"),
                    key="corr_heat", **WIDE
                )
                st.caption("Low or negative correlation = better diversification.")
            except Exception as e:
                st.error(str(e))
        else:
            st.caption("Select at least 2 instruments.")

# =====================================================
# TAB 6 - MARKET EXPLORER
# =====================================================

with tab_explore:
    st.subheader("🌍 Multi-Asset Market Explorer")
    st.write("Explore stocks, metals, indices, forex and crypto.")

    explorer_class = st.selectbox(
        "Choose Asset Class", list(ASSETS.keys()), key="explorer_class"
    )
    explorer_names = list(ASSETS[explorer_class].keys())
    explorer_selection = st.multiselect(
        "Select Instruments", explorer_names,
        default=explorer_names[:3], key="explorer_selection"
    )

    if st.button("🔄 Load Market Data", key="explorer_button"):
        market_results, market_errors, closes = [], [], {}

        with st.spinner("Fetching market data..."):
            for name in explorer_selection:
                ticker = ASSETS[explorer_class][name]
                try:
                    df = get_market_data(ticker)
                    cl = df["Close"]
                    last = float(cl.iloc[-1])

                    def chg(n):
                        return (last / float(cl.iloc[-1 - n]) - 1) * 100 \
                            if len(cl) > n else np.nan

                    this_year = cl[cl.index.year == cl.index[-1].year]
                    ytd = (last / float(this_year.iloc[0]) - 1) * 100 \
                        if len(this_year) else np.nan

                    hi52 = float(cl.tail(252).max())
                    lo52 = float(cl.tail(252).min())
                    rsi = float(add_indicators(df)["RSI"].iloc[-1])

                    market_results.append({
                        "Instrument": name, "Ticker": ticker,
                        "Last Price": last,
                        "1D (%)": chg(1), "1W (%)": chg(5),
                        "1M (%)": chg(21), "YTD (%)": ytd,
                        "From 52W High (%)": (last / hi52 - 1) * 100,
                        "52W Range Position (%)":
                            (last - lo52) / (hi52 - lo52) * 100 if hi52 > lo52 else np.nan,
                        "RSI": rsi
                    })
                    closes[name] = strip_tz(cl).tail(252)
                except Exception as e:
                    market_errors.append(f"{name}: {e}")

        st.session_state["market_results"] = market_results
        st.session_state["market_errors"] = market_errors
        st.session_state["market_closes"] = closes
        st.session_state["market_class"] = explorer_class

    same_class = st.session_state.get("market_class") == explorer_class
    if not same_class:
        st.caption("Press **Load Market Data** to see this asset class.")

    if same_class and st.session_state.get("market_results"):
        market_df = pd.DataFrame(st.session_state["market_results"])

        st.dataframe(
            market_df.style.format({
                "Last Price": "{:,.4f}", "1D (%)": "{:+.2f}%",
                "1W (%)": "{:+.2f}%", "1M (%)": "{:+.2f}%",
                "YTD (%)": "{:+.2f}%", "From 52W High (%)": "{:+.2f}%",
                "52W Range Position (%)": "{:.0f}%", "RSI": "{:.1f}"
            }),
            **WIDE
        )

        closes = st.session_state.get("market_closes", {})
        if len(closes) >= 1:
            reb = pd.concat(closes, axis=1).ffill().dropna()
            if not reb.empty:
                reb = reb / reb.iloc[0] * 100
                fig = px.line(reb)
                plot_chart(
                    style_fig(fig, 420, "Performance Comparison (rebased to 100, 1 year)"),
                    key="explorer_reb", **WIDE
                )

        st.download_button("📥 Download table",
                           market_df.to_csv(index=False),
                           file_name="market_explorer.csv", mime="text/csv")

    for error in (st.session_state.get("market_errors", []) if same_class else []):
        st.warning(error)

# =====================================================
# TAB 7 - ABOUT
# =====================================================

with tab_about:
    st.subheader("About QUANTLAB 39")

    st.markdown("""
    ### AI Powered Multi-Asset Intelligence Platform

    **Supported Markets**
    - US and Indian Stocks
    - Precious and Industrial Metals
    - Global Stock Indices
    - Major Forex Pairs
    - Major Cryptocurrencies

    **Machine Learning**
    - Saved Ridge Regression stock models (next-day return)
    - ML Lab: live training for any asset, 4 regression + 3 up/down classifier models
    - Holdout split AND walk-forward validation, AUC, directional accuracy vs baseline
    - Features include momentum, volatility regime, RSI/MACD, S&P 500 market context
    - Feature importance and ML strategy backtesting with trading costs

    **Risk & Portfolio**
    - Volatility, Sharpe, drawdown, VaR, CVaR
    - Monte Carlo (Geometric Brownian Motion) simulation
    - Position tracker, efficient frontier optimizer, correlation heatmap

    **Technical Indicators**
    - RSI (14), MACD, Stochastic %K
    - EMA / SMA 20 / 50 / 200
    - Bollinger Bands, ATR (14)
    - Rule-based technical score

    **Technology**
    - Python, Streamlit, Pandas, NumPy
    - Scikit-learn, Plotly, yfinance, joblib

    **Important Disclaimer**

    This platform is an educational project.
    Market prices may be delayed or unavailable.
    AI predictions, backtests and technical indicators are not
    financial advice or guaranteed trading signals.
    """)

st.markdown("---")
st.caption(
    "QUANTLAB 39 | Multi-Asset Intelligence Platform | Educational Use Only"
)
