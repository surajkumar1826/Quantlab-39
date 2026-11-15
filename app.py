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

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="QUANTLAB 39 | AI Stock Intelligence",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080D18, #101B2D);
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background: #111C2E;
    border-right: 1px solid #263750;
}

h1, h2, h3 {
    color: #00D4AA !important;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(90deg, #3978F6, #00BFA6);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 700;
}

.stButton > button:hover {
    background: #00D4AA;
    color: #080D18;
    border: none;
}

[data-testid="stMetric"] {
    background: #15243A;
    border: 1px solid #263B56;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #00D4AA;
}

[data-testid="stDataFrame"] {
    border: 1px solid #263B56;
    border-radius: 12px;
}

.stTabs [aria-selected="true"] {
    color: #00D4AA !important;
}

hr {
    border-color: #263750;
}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
ZIP_PATH = BASE / "QuantLab39_models.zip"

STOCKS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

# ---------------- LOAD MODELS ----------------

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

# ---------------- MARKET DATA ----------------

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    df = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No stock data available.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# ---------------- INDICATORS ----------------

def add_indicators(df):
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    return df

# ---------------- MODEL FEATURES ----------------

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
            df["Close"] / df["Close"].rolling(n).mean() - 1
        )

    df["distEma12"] = (
        df["Close"] /
        df["Close"].ewm(span=12).mean() - 1
    )

    df["logVolume"] = np.log1p(df["Volume"])

    df["volumeRatioDev"] = (
        df["Volume"] / df["Volume"].rolling(20).mean() - 1
    )

    df["volatility5"] = df["dailyReturn"].rolling(5).std()

    return df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

# ---------------- PREDICTION ----------------

def predict_stock(ticker, data):
    folder = model_root / ticker

    model_path = folder / "ridge_model.pkl"
    scaler_path = folder / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model not found for {ticker}"
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

    X = feature_data[features].iloc[[-1]]
    X_scaled = scaler.transform(X)

    predicted_return = float(model.predict(X_scaled)[0])

    current = float(data["Close"].iloc[-1])
    predicted = current * (1 + predicted_return)

    return current, predicted, predicted_return

# ---------------- CHARTS ----------------

def show_chart(data, ticker):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            "Price & Moving Averages",
            "RSI (14)",
            "MACD"
        )
    )

    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price"
        ),
        row=1, col=1
    )

    for col, color in [
        ("SMA20", "#55d6be"),
        ("SMA50", "#ffbd59"),
        ("SMA200", "#e879f9")
    ]:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                name=col,
                line=dict(color=color, width=1.5)
            ),
            row=1, col=1
        )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["RSI"],
            name="RSI",
            line=dict(color="#55d6be")
        ),
        row=2, col=1
    )

    fig.add_hline(
        y=70, line_dash="dash",
        line_color="red", row=2, col=1
    )

    fig.add_hline(
        y=30, line_dash="dash",
        line_color="green", row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["MACD"],
            name="MACD"
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Signal"],
            name="Signal"
        ),
        row=3, col=1
    )

    fig.update_layout(
        title=f"{ticker} Technical Analysis",
        template="plotly_dark",
        height=800,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- APP HEADER ----------------

st.title("📈 QUANTLAB 39")
st.caption("AI Powered Stock Intelligence Platform")

if config is None:
    st.error("Model configuration not found.")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("⚙️ Controls")

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Single Stock", "Multiple Stocks"]
)

if mode == "Single Stock":
    selected_stocks = [
        st.sidebar.selectbox("Select Stock", STOCKS)
    ]
else:
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        STOCKS,
        default=["AAPL", "NVDA", "MSFT"]
    )

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
st.sidebar.caption("Prediction Model: Ridge Regression")
st.sidebar.caption("Forecast: Next Trading Day")

# ---------------- TABS ----------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predictions",
    "📈 Technical Analysis",
    "💰 Portfolio Calculator",
    "ℹ️ About"
])

# ---------------- PREDICTIONS ----------------

with tab1:
    st.subheader("AI Stock Predictions")

    if st.button("🚀 Generate Prediction", type="primary"):
        if not selected_stocks:
            st.warning("Select at least one stock.")
        else:
            results = []
            chart_data = {}

            with st.spinner("Analyzing market data..."):
                for ticker in selected_stocks:
                    try:
                        data = get_stock_data(ticker)

                        current, predicted, ret = predict_stock(
                            ticker, data
                        )

                        results.append({
                            "Ticker": ticker,
                            "Current Price": current,
                            "Predicted Price": predicted,
                            "Expected Return (%)": ret * 100
                        })

                        chart_data[ticker] = data

                    except Exception as e:
                        st.error(f"{ticker}: {e}")

            st.session_state["results"] = results
            st.session_state["chart_data"] = chart_data

    if st.session_state.get("results"):
        result_df = pd.DataFrame(
            st.session_state["results"]
        )

        st.subheader("Prediction Summary")

        st.dataframe(
            result_df.style.format({
                "Current Price": "${:,.2f}",
                "Predicted Price": "${:,.2f}",
                "Expected Return (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

        st.download_button(
            "📥 Download CSV",
            result_df.to_csv(index=False),
            "quantlab39_predictions.csv",
            "text/csv"
        )

        for _, row in result_df.iterrows():
            ticker = row["Ticker"]

            with st.expander(
                f"{ticker} - Detailed Analysis",
                expanded=(len(result_df) == 1)
            ):
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Current Price",
                    f"${row['Current Price']:,.2f}"
                )

                c2.metric(
                    "Predicted Price",
                    f"${row['Predicted Price']:,.2f}"
                )

                c3.metric(
                    "Expected Return",
                    f"{row['Expected Return (%)']:.2f}%"
                )

                chart_data = st.session_state["chart_data"][ticker]
                show_chart(
                    add_indicators(chart_data).tail(
                        period_map[period]
                    ),
                    ticker
                )

# ---------------- TECHNICAL ANALYSIS ----------------

with tab2:
    st.subheader("Technical Analysis Center")

    technical_stock = st.selectbox(
        "Choose Stock",
        STOCKS,
        key="technical_stock"
    )

    if st.button("Analyze Stock"):
        try:
            data = add_indicators(
                get_stock_data(technical_stock)
            )

            latest = data.iloc[-1]

            a, b, c = st.columns(3)

            a.metric(
                "RSI",
                f"{latest['RSI']:.2f}"
            )

            b.metric(
                "MACD",
                f"{latest['MACD']:.2f}"
            )

            c.metric(
                "Latest Price",
                f"${latest['Close']:.2f}"
            )

            if latest["RSI"] > 70:
                st.warning("RSI indicates a potentially overbought condition.")
            elif latest["RSI"] < 30:
                st.success("RSI indicates a potentially oversold condition.")
            else:
                st.info("RSI is in the neutral range.")

            show_chart(
                data.tail(period_map[period]),
                technical_stock
            )

        except Exception as e:
            st.error(str(e))

# ---------------- PORTFOLIO CALCULATOR ----------------

with tab3:
    st.subheader("💰 Portfolio Calculator")

    st.caption(
        "Calculate the current value and hypothetical profit/loss "
        "of your stock holdings."
    )

    investment = st.number_input(
        "Investment per share ($)",
        min_value=0.01,
        value=100.0
    )

    quantity = st.number_input(
        "Number of shares",
        min_value=1,
        value=10
    )

    portfolio_stock = st.selectbox(
        "Select Stock",
        STOCKS,
        key="portfolio_stock"
    )

    if st.button("Calculate Portfolio"):
        try:
            data = get_stock_data(portfolio_stock)

            current = float(data["Close"].iloc[-1])

            invested = investment * quantity
            current_value = current * quantity
            profit = current_value - invested
            returns = (profit / invested) * 100

            a, b, c = st.columns(3)

            a.metric("Investment", f"${invested:,.2f}")
            b.metric("Current Value", f"${current_value:,.2f}")
            c.metric(
                "Profit / Loss",
                f"${profit:,.2f}",
                delta=f"{returns:.2f}%"
            )

        except Exception as e:
            st.error(str(e))

# ---------------- ABOUT ----------------

with tab4:
    st.subheader("About QUANTLAB 39")

    st.markdown("""
    ### AI Powered Stock Intelligence

    **Prediction Model**
    - Ridge Regression

    **Technical Indicators**
    - RSI (14)
    - MACD
    - SMA 20
    - SMA 50
    - SMA 200

    **Market Data**
    - Yahoo Finance

    **Technology Stack**
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly

    **Disclaimer**

    This application is an educational project.
    Predictions and indicators are not financial advice.
    """)

st.markdown("---")

st.caption(
    "QUANTLAB 39 | AI Stock Intelligence Platform | "
    "Educational purposes only."
)
