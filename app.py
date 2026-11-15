import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import zipfile
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="QUANTLAB 39 | Multi Asset Intelligence",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# 60% DARK / 40% LIGHT THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0B1220;
    color: #F1F5F9;
}

[data-testid="stHeader"] {
    background: #0B1220;
}

[data-testid="stSidebar"] {
    background: #172438;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

h1, h2, h3 {
    color: #F8FAFC !important;
    font-weight: 750;
}

p, label {
    color: #CBD5E1;
}

[data-testid="stMetric"] {
    background: #F1F5F9;
    border: 1px solid #DCE4ED;
    padding: 17px;
    border-radius: 13px;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #0F766E !important;
}

[data-testid="stMetricDelta"] {
    color: #059669 !important;
}

.stButton > button {
    background: #00BFA6;
    color: #07111F;
    border: none;
    border-radius: 9px;
    font-weight: 750;
    padding: 10px 18px;
}

.stButton > button:hover {
    background: #00D6B5;
    color: #07111F;
}

[data-testid="stDataFrame"] {
    background: #F1F5F9;
    border-radius: 12px;
}

.stTabs [aria-selected="true"] {
    color: #00D6A3 !important;
}

.stTabs [data-baseweb="tab"] {
    color: #CBD5E1;
}

[data-testid="stExpander"] {
    background: #172438;
    border: 1px solid #2B3C54;
    border-radius: 10px;
}

hr {
    border-color: #29384D;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stNumberInput"] > div > div {
    background: #F1F5F9;
    color: #172B4D;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
ZIP_PATH = BASE / "QuantLab39_models.zip"

# =====================================================
# MULTI-ASSET UNIVERSE
# =====================================================

ASSETS = {

    "Stocks": {
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "AMD": "AMD",
        "Netflix": "NFLX",
        "Broadcom": "AVGO",
        "Oracle": "ORCL",
        "JPMorgan": "JPM",
        "Walmart": "WMT",
        "Intel": "INTC",
        "Qualcomm": "QCOM",
        "IBM": "IBM",
        "Palantir": "PLTR",
        "TSMC": "TSM",
        "Alibaba": "BABA",
        "Reliance": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS"
    },

    "Metals": {
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Platinum": "PL=F",
        "Palladium": "PA=F",
        "Copper": "HG=F"
    },

    "Indices": {
        "S&P 500": "^GSPC",
        "NASDAQ Composite": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225"
    },

    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "USDCAD=X",
        "USD/CHF": "USDCHF=X",
        "NZD/USD": "NZDUSD=X"
    }
}

STOCKS = list(ASSETS["Stocks"].values())

# =====================================================
# LOAD EXISTING AI MODELS
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

# =====================================================
# MARKET DATA
# =====================================================

@st.cache_data(ttl=300)
def get_market_data(ticker):

    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"No market data available for {ticker}"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing {col} data")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    return df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna()


# =====================================================
# TECHNICAL INDICATORS
# =====================================================

def add_indicators(df):

    df = df.copy()

    # Moving averages
    for n in [20, 50, 200]:
        df[f"SMA{n}"] = df["Close"].rolling(n).mean()
        df[f"EMA{n}"] = df["Close"].ewm(
            span=n, adjust=False
        ).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26, adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Bollinger Bands
    middle = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()

    df["BB_Upper"] = middle + (2 * std)
    df["BB_Lower"] = middle - (2 * std)

    # ATR
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] - df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] - df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    return df


# =====================================================
# MODEL FEATURE ENGINEERING
# =====================================================

def create_features(df):

    df = df.copy()

    df["intradaySpread"] = (
        (df["High"] - df["Low"]) / df["Close"]
    )

    df["dailyReturn"] = df["Close"].pct_change()
    df["lag1Return"] = df["dailyReturn"].shift(1)
    df["lag2Return"] = df["dailyReturn"].shift(2)

    for n in [5, 10, 20]:
        df[f"distSMA{n}"] = (
            df["Close"] /
            df["Close"].rolling(n).mean() - 1
        )

    df["distEma12"] = (
        df["Close"] /
        df["Close"].ewm(span=12).mean() - 1
    )

    df["logVolume"] = np.log1p(df["Volume"])

    volume_mean = df["Volume"].rolling(20).mean()

    df["volumeRatioDev"] = (
        df["Volume"] / volume_mean - 1
    )

    df["volatility5"] = (
        df["dailyReturn"].rolling(5).std()
    )

    return df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()


# =====================================================
# AI STOCK PREDICTION
# =====================================================

def predict_stock(ticker, data):

    if model_root is None:
        raise FileNotFoundError(
            "AI model configuration is missing."
        )

    folder = model_root / ticker

    model_path = folder / "ridge_model.pkl"
    scaler_path = folder / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"No trained AI model available for {ticker}. "
            "This asset needs its own trained model."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features = [
        "intradaySpread",
        "lag1Return",
        "lag2Return",
        "distSMA5",
        "distSMA10",
        "distSMA20",
        "distEma12",
        "logVolume",
        "volumeRatioDev",
        "dailyReturn",
        "volatility5"
    ]

    feature_data = create_features(data)

    if feature_data.empty:
        raise ValueError("Not enough data for prediction.")

    X = feature_data[features].iloc[[-1]]

    X_scaled = scaler.transform(X)

    predicted_return = float(
        model.predict(X_scaled)[0]
    )

    current = float(data["Close"].iloc[-1])

    predicted = current * (1 + predicted_return)

    return current, predicted, predicted_return


# =====================================================
# MARKET CHART
# =====================================================

def show_chart(data, name):

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=[
            "Candlestick & Moving Averages",
            "RSI (14)",
            "MACD"
        ]
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#00D6A3",
            decreasing_line_color="#FF5263"
        ),
        row=1, col=1
    )

    for col, color in [
        ("EMA20", "#00D6A3"),
        ("EMA50", "#F5B942"),
        ("EMA200", "#A78BFA"),
        ("BB_Upper", "#64748B"),
        ("BB_Lower", "#64748B")
    ]:

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                line=dict(color=color, width=1.3)
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line=dict(color="#00D6A3")
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="#FF5263",
        row=2, col=1
    )

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="#00D6A3",
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD",
            line=dict(color="#3B82F6")
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Signal"],
            name="Signal",
            line=dict(color="#F5B942")
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=f"{name} | Technical Analysis",
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#111C2E",
        font=dict(color="#E2E8F0"),
        height=820,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================================
# APP HEADER
# =====================================================

st.title("📈 QUANTLAB 39")

st.caption(
    "AI Powered Multi-Asset Market Intelligence Platform"
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Market Controls")

asset_class = st.sidebar.selectbox(
    "Select Market",
    list(ASSETS.keys())
)

asset_names = list(ASSETS[asset_class].keys())

selected_name = st.sidebar.selectbox(
    "Select Instrument",
    asset_names
)

selected_ticker = ASSETS[asset_class][selected_name]

period = st.sidebar.selectbox(
    "Chart History",
    ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"],
    index=3
)

period_map = {
    "1 Month": 22,
    "3 Months": 66,
    "6 Months": 132,
    "1 Year": 252,
    "5 Years": 1260
}

st.sidebar.markdown("---")

st.sidebar.caption(
    f"Market: {asset_class}"
)

st.sidebar.caption(
    f"Instrument: {selected_ticker}"
)

st.sidebar.caption(
    "Data source: Yahoo Finance"
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 AI Predictions",
    "📈 Technical Analysis",
    "🌍 Market Explorer",
    "💰 Portfolio Calculator",
    "ℹ️ About"
])


# =====================================================
# TAB 1 - AI PREDICTIONS
# =====================================================

with tab1:

    st.subheader("AI Stock Prediction Center")

    st.info(
        "AI predictions currently use your existing trained "
        "Ridge Regression stock models. Other asset classes "
        "require separately trained models."
    )

    stock_names = list(ASSETS["Stocks"].keys())

    stock_lookup = ASSETS["Stocks"]

    selected_prediction_names = st.multiselect(
        "Select stocks for AI prediction",
        stock_names,
        default=["Apple", "NVIDIA", "Microsoft"]
    )

    if st.button(
        "🚀 Generate AI Predictions",
        type="primary",
        key="prediction_button"
    ):

        results = []
        errors = []

        with st.spinner("Running AI models..."):

            for stock_name in selected_prediction_names:

                ticker = stock_lookup[stock_name]

                try:

                    data = get_market_data(ticker)

                    current, predicted, ret = predict_stock(
                        ticker, data
                    )

                    results.append({
                        "Stock": stock_name,
                        "Ticker": ticker,
                        "Current Price": current,
                        "Predicted Price": predicted,
                        "Expected Return (%)": ret * 100
                    })

                except Exception as e:

                    errors.append(
                        f"{stock_name}: {str(e)}"
                    )

        st.session_state["ai_results"] = results

        st.session_state["ai_errors"] = errors

    if st.session_state.get("ai_results"):

        result_df = pd.DataFrame(
            st.session_state["ai_results"]
        )

        st.subheader("Prediction Summary")

        st.dataframe(
            result_df.style.format({
                "Current Price": "{:,.4f}",
                "Predicted Price": "{:,.4f}",
                "Expected Return (%)": "{:+.2f}%"
            }),
            use_container_width=True
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

with tab2:

    st.subheader("Advanced Technical Analysis")

    st.write(
        f"Analyzing: **{selected_name} ({selected_ticker})**"
    )

    if st.button(
        "Analyze Market",
        type="primary",
        key="technical_button"
    ):

        try:

            data = get_market_data(selected_ticker)

            data = add_indicators(data)

            data = data.tail(
                period_map[period]
            )

            st.session_state["technical_data"] = data

            st.session_state["technical_name"] = selected_name

            st.session_state["technical_ticker"] = selected_ticker

        except Exception as e:

            st.error(str(e))

    if st.session_state.get("technical_data") is not None:

        data = st.session_state["technical_data"]

        latest = data.iloc[-1]

        a, b, c, d = st.columns(4)

        a.metric(
            "Latest Price",
            f"{latest['Close']:,.4f}"
        )

        b.metric(
            "RSI (14)",
            f"{latest['RSI']:.2f}"
        )

        c.metric(
            "MACD",
            f"{latest['MACD']:.4f}"
        )

        d.metric(
            "ATR (14)",
            f"{latest['ATR']:.4f}"
        )

        st.subheader("Indicator Interpretation")

        if latest["RSI"] > 70:

            st.warning(
                "RSI is above 70: potentially overbought."
            )

        elif latest["RSI"] < 30:

            st.success(
                "RSI is below 30: potentially oversold."
            )

        else:

            st.info(
                "RSI is in the neutral range."
            )

        if latest["MACD"] > latest["Signal"]:

            st.success(
                "MACD is above its signal line."
            )

        else:

            st.warning(
                "MACD is below its signal line."
            )

        show_chart(
            data,
            st.session_state["technical_name"]
        )

        st.caption(
            "Indicators are technical observations, not guaranteed trade signals."
        )


# =====================================================
# TAB 3 - MARKET EXPLORER
# =====================================================

with tab3:

    st.subheader("🌍 Multi-Asset Market Explorer")

    st.write(
        "Explore stocks, metals, indices and forex instruments."
    )

    explorer_class = st.selectbox(
        "Choose Asset Class",
        list(ASSETS.keys()),
        key="explorer_class"
    )

    explorer_names = list(
        ASSETS[explorer_class].keys()
    )

    explorer_selection = st.multiselect(
        "Select Instruments",
        explorer_names,
        default=explorer_names[:3],
        key="explorer_selection"
    )

    if st.button(
        "🔄 Load Market Data",
        key="explorer_button"
    ):

        market_results = []
        market_errors = []

        with st.spinner("Fetching market data..."):

            for name in explorer_selection:

                ticker = ASSETS[explorer_class][name]

                try:

                    df = get_market_data(ticker)

                    current = float(df["Close"].iloc[-1])

                    previous = float(df["Close"].iloc[-2])

                    change = (
                        (current - previous) / previous
                    ) * 100 if previous else 0

                    market_results.append({
                        "Instrument": name,
                        "Ticker": ticker,
                        "Last Price": current,
                        "Daily Change (%)": change
                    })

                except Exception as e:

                    market_errors.append(
                        f"{name}: {str(e)}"
                    )

        st.session_state["market_results"] = market_results

        st.session_state["market_errors"] = market_errors

    if st.session_state.get("market_results"):

        market_df = pd.DataFrame(
            st.session_state["market_results"]
        )

        st.dataframe(
            market_df.style.format({
                "Last Price": "{:,.4f}",
                "Daily Change (%)": "{:+.2f}%"
            }),
            use_container_width=True
        )

    for error in st.session_state.get("market_errors", []):
        st.warning(error)


# =====================================================
# TAB 4 - PORTFOLIO CALCULATOR
# =====================================================

with tab4:

    st.subheader("💰 Portfolio Calculator")

    st.caption(
        "Calculate the current value and hypothetical "
        "profit or loss of an asset position."
    )

    portfolio_class = st.selectbox(
        "Asset Class",
        list(ASSETS.keys()),
        key="portfolio_class"
    )

    portfolio_names = list(
        ASSETS[portfolio_class].keys()
    )

    portfolio_name = st.selectbox(
        "Select Instrument",
        portfolio_names,
        key="portfolio_name"
    )

    portfolio_ticker = ASSETS[portfolio_class][portfolio_name]

    entry_price = st.number_input(
        "Your Entry Price",
        min_value=0.000001,
        value=100.0,
        format="%.6f"
    )

    quantity = st.number_input(
        "Quantity / Units",
        min_value=0.000001,
        value=1.0,
        format="%.6f"
    )

    if st.button(
        "Calculate Portfolio",
        type="primary",
        key="portfolio_button"
    ):

        try:

            df = get_market_data(portfolio_ticker)

            current = float(df["Close"].iloc[-1])

            invested = entry_price * quantity

            current_value = current * quantity

            profit = current_value - invested

            returns = (
                profit / invested
            ) * 100

            a, b, c = st.columns(3)

            a.metric(
                "Invested Value",
                f"{invested:,.4f}"
            )

            b.metric(
                "Current Value",
                f"{current_value:,.4f}"
            )

            c.metric(
                "Profit / Loss",
                f"{profit:+,.4f}",
                delta=f"{returns:+.2f}%"
            )

            st.caption(
                f"Instrument: {portfolio_name} ({portfolio_ticker})"
            )

        except Exception as e:

            st.error(str(e))


# =====================================================
# TAB 5 - ABOUT
# =====================================================

with tab5:

    st.subheader("About QUANTLAB 39")

    st.markdown("""
    ### AI Powered Multi-Asset Intelligence Platform

    **Supported Markets**
    - US and Indian Stocks
    - Precious and Industrial Metals
    - Global Stock Indices
    - Major Forex Pairs

    **AI Prediction**
    - Ridge Regression
    - Existing trained stock models
    - Next Trading Day Return Estimation

    **Technical Indicators**
    - RSI (14)
    - MACD
    - EMA 20 / 50 / 200
    - SMA 20 / 50 / 200
    - Bollinger Bands
    - ATR (14)

    **Market Data**
    - Yahoo Finance

    **Technology**
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly

    **Important Disclaimer**

    This platform is an educational project.
    Market prices may be delayed or unavailable.
    AI predictions and technical indicators are not
    financial advice or guaranteed trading signals.
    """)

st.markdown("---")

st.caption(
    "QUANTLAB 39 | Multi-Asset Intelligence Platform | "
    "Educational Use Only"
)
