import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import zipfile
import json
from pathlib import Path
import plotly.graph_objects as go

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="QuantLab 39 | AI Stock Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #080d18;
    color: #ffffff;
}
[data-testid="stMetric"] {
    background: #151e30;
    padding: 18px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: #55d6be;
}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent
ZIP_PATH = BASE / "QuantLab39_models.zip"

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
    model_root = config_path.parent

    with open(config_path, "r") as f:
        config = json.load(f)

    return config, model_root


config, model_root = load_models()

# ---------------- FEATURE ENGINEERING ----------------

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
        df["Close"] / df["Close"].ewm(span=12).mean() - 1
    )

    df["logVolume"] = np.log1p(df["Volume"])

    df["volumeRatioDev"] = (
        df["Volume"] / df["Volume"].rolling(20).mean() - 1
    )

    df["volatility5"] = df["dailyReturn"].rolling(5).std()

    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------- PREDICTION ----------------

def predict_stock(ticker):
    folder = model_root / ticker

    model_path = folder / "ridge_model.pkl"
    scaler_path = folder / "scaler.pkl"

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Saved Ridge model not found for {ticker}"
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    data = yf.download(
        ticker,
        period="5y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("Unable to download stock data.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[["Open", "High", "Low", "Close", "Volume"]]

    features_df = create_features(data)

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

    X = features_df[features].iloc[[-1]]
    X_scaled = scaler.transform(X)

    predicted_return = float(model.predict(X_scaled)[0])

    current_price = float(data["Close"].iloc[-1])
    predicted_price = current_price * (1 + predicted_return)

    return data, current_price, predicted_price, predicted_return


# ---------------- DASHBOARD ----------------

st.title("📈 QUANTLAB 39")
st.caption("AI Powered Multi-Stock Prediction Platform")

if config is None:
    st.error("Model configuration not found. Check the ZIP file structure.")
    st.stop()

stocks = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]

mode = st.sidebar.radio(
    "Prediction Mode",
    ["Single Stock", "Multiple Stocks"]
)

if mode == "Single Stock":
    selected = st.sidebar.selectbox("Select Stock", stocks)
    selected_stocks = [selected]
else:
    selected_stocks = st.sidebar.multiselect(
        "Select Stocks",
        stocks,
        default=["AAPL", "NVDA", "MSFT"]
    )

st.sidebar.markdown("---")
st.sidebar.caption("Model: Ridge Regression")
st.sidebar.caption("Forecast: Next Trading Day")

if st.button("🚀 Generate Prediction", type="primary"):
    if not selected_stocks:
        st.warning("Select at least one stock.")
        st.stop()

    results = []

    for ticker in selected_stocks:
        try:
            data, current, predicted, ret = predict_stock(ticker)

            results.append({
                "Ticker": ticker,
                "Current Price": current,
                "Predicted Price": predicted,
                "Expected Return (%)": ret * 100
            })

            if mode == "Single Stock":
                st.subheader(f"{ticker} Stock Analysis")

                c1, c2, c3 = st.columns(3)

                c1.metric("Current Price", f"${current:,.2f}")
                c2.metric("Predicted Price", f"${predicted:,.2f}")
                c3.metric(
                    "Expected Return",
                    f"{ret * 100:.2f}%",
                    delta=f"{ret * 100:.2f}%"
                )

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=data.index[-180:],
                    y=data["Close"].iloc[-180:],
                    name="Historical Price",
                    line=dict(color="#55d6be", width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=[data.index[-1], data.index[-1] + pd.Timedelta(days=1)],
                    y=[current, predicted],
                    name="Next Day Prediction",
                    line=dict(color="#ffbd59", width=3, dash="dash")
                ))

                fig.update_layout(
                    title=f"{ticker} Price Analysis",
                    template="plotly_dark",
                    height=450,
                    xaxis_title="Date",
                    yaxis_title="Price (USD)"
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"{ticker}: {str(e)}")

    if results:
        st.subheader("Prediction Summary")

        result_df = pd.DataFrame(results)

        st.dataframe(
            result_df.style.format({
                "Current Price": "${:,.2f}",
                "Predicted Price": "${:,.2f}",
                "Expected Return (%)": "{:.2f}%"
            }),
            use_container_width=True
        )

        st.download_button(
            "📥 Download Predictions CSV",
            result_df.to_csv(index=False),
            file_name="quantlab39_predictions.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption(
    "QuantLab 39 | AI Stock Prediction Platform | "
    "Educational purposes only. Predictions are not financial advice."
)
