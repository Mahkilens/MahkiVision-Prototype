import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Page Configuration ---
st.set_page_config(
    page_title="MahkiVision",
    page_icon="🤖",
    layout="wide"
)


# --- Data Fetching Function ---
def get_stock_data(ticker_symbol, period, interval):
    # yfinance does not support 4h intervals natively; map to 1h as the nearest equivalent
    yf_interval = "1h" if interval == "4h" else interval

    stock = yf.Ticker(ticker_symbol)

    data = stock.history(
        period=period,
        interval=yf_interval
    )

    return data

# normalize forex inputs
def normalize_inputs(user_input, asset_type):
    symbol = user_input.upper().replace("/", "").replace(" ", "")
    
    if asset_type == "Forex":
        if not symbol.endswith("=X"):
            symbol = symbol + "=X"

    return symbol

# ==============================================================
# SESSION STATE DEFAULTS
# ==============================================================
_defaults = {
    "analysis_ready":               False,
    "ai_summary":                   "",
    "trade_setup":                  "",
    "trade_journal":                [],
    # Sidebar input persistence — survive page navigation
    "asset_type":                   "Stock / ETF",
    "ticker":                       "VOO",
    "historical_range":             "1y",
    "candle_timeframe":             "1d",
    "previous_asset_type":          "Stock / ETF",
    "latest_symbol":                None,
    "latest_asset_type":            None,
    "latest_market_bias":           None,
    "latest_setup_score":           None,
    "latest_risk_level":            None,
    "latest_momentum_status":       None,
    "latest_alignment_summary":     None,
    "latest_market_context":        None,
    # Trade Readiness — shared across Dashboard, Opportunities, AI Alerts
    "latest_trade_readiness_score": None,
    "latest_trade_readiness_label": None,
    "latest_decision_action":       None,
    "latest_decision_reason":       None,
    "latest_what_would_improve":    [],
    "latest_what_to_avoid":         [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Auto-reset ticker when asset type changes (only on actual type switch)
if st.session_state.asset_type != st.session_state.previous_asset_type:
    if st.session_state.asset_type == "Forex":
        st.session_state.ticker = "EURUSD"
    else:
        st.session_state.ticker = "VOO"
    st.session_state.previous_asset_type = st.session_state.asset_type

# ==============================================================
# SIDEBAR NAVIGATION
# ==============================================================
st.sidebar.title("MahkiVision")
st.sidebar.caption("AI Market Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Asset Analyzer", "Opportunities", "AI Alerts", "Trade Journal", "Settings / About"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Educational use only. Not financial advice.")

# AI Market Intelligence Summary
def generate_ai_summary(market_context):

    prompt = f"""You are MahkiVision AI.
You provide EDUCATIONAL market analysis only. You are NOT a financial advisor.
You NEVER say "buy", "sell", "hold", "enter", or "exit".
Use cautious, probability-based language. Acknowledge uncertainty honestly.

Market Context:
{market_context}

Respond using EXACTLY this structure:

### 1. Plain-English Read
Explain what is currently happening in 2-3 clear sentences.
No jargon. Speak like you are helping a thoughtful friend understand the market.

### 2. Decision Status
Explain why the current conditions lead to WATCH, WAIT, or AVOID FOR NOW.
Use language like: "Conditions may be developing...", "Setup appears to need...", "Confirmation may be required..."

### 3. Risk Check
Be honest. What could go wrong? What are the main elevated risks right now?

### 4. What Would Improve This Setup
List 2-4 specific improvements that would make the setup cleaner and more readable.

### 5. Human Review Checklist
List 3-5 things the user should manually verify before making any decision.
(Examples: higher timeframe context, news catalysts, key support/resistance, broker conditions)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "You are MahkiVision AI. Educational market intelligence only. No financial advice. Never say buy, sell, or hold."},
                {"role": "user",   "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI analysis could not be generated right now. Please try again. (Error: {e})"

# AI Trade Setup Intelligence
def generate_trade_setup_analysis(market_context):

    prompt = f"""You are MahkiVision AI.
You evaluate market setup conditions for EDUCATIONAL purposes only.
You NEVER say "buy", "sell", "hold", "enter", or "exit".
Describe probabilities and conditions, not certainties.

Market Context:
{market_context}

Respond using EXACTLY this structure:

### 1. Setup Condition
Is this setup currently: Favorable / Mixed / Risky / Unclear?
Explain in 2-3 sentences.

### 2. Timing Quality
Is this setup: Early Stage / Developing / Overextended / Requires Confirmation?
Explain the timing context briefly.

### 3. Supportive Evidence
List 3-5 technical conditions currently supporting the setup direction.

### 4. Warning Signs
List 3-5 conditions that could weaken or invalidate this setup.

### 5. Final Review
Write a clear 1-2 sentence closing using language like:
\"This may be worth watching closely as conditions develop.\"
\"This setup needs more confirmation before the risk is justified.\"
\"This appears too risky right now based on current conditions.\"
Never give financial advice. Never guarantee outcomes.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": "You are MahkiVision AI. Probabilistic setup analysis for educational use only. No financial advice."},
                {"role": "user",   "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Setup review could not be generated right now. Please try again. (Error: {e})"

# --- Normalize ticker symbols based on asset type ---
def normalize_ticker(ticker, asset_type):

    ticker = ticker.upper().strip()

    # Forex handling
    if asset_type == "Forex":

        # If user enters EURUSD
        if len(ticker) == 6 and "-" not in ticker:
            return f"{ticker}=X"

        # If user enters EUR/USD
        if "/" in ticker:
            clean_pair = ticker.replace("/", "")
            return f"{clean_pair}=X"

        # If user already entered EURUSD=X
        if "=X" in ticker:
            return ticker

    # Default stock/ETF behavior
    return ticker

# --- Validate ticker input based on asset type ---
def validate_ticker(ticker, asset_type):

    ticker = ticker.upper().strip()

    # Forex validation
    if asset_type == "Forex":

        # Remove formatting
        clean_ticker = ticker.replace("/", "").replace("=X", "")

        # Forex pairs should be 6 letters
        if len(clean_ticker) != 6:
            return False

        # Must contain only letters
        if not clean_ticker.isalpha():
            return False

        return True

    # Stock/ETF validation
    else:

        # Stocks usually shorter
        if len(ticker) > 5:
            return False

        # Stocks should not contain slash
        if "/" in ticker:
            return False

        # Stocks should not contain =X
        if "=X" in ticker:
            return False

        return True


# --- Smart Interval + Period Validation ---
# yfinance has strict limits on how far back you can request intraday data.
# This function catches invalid combinations before any API call is made.
def validate_interval_period(period, interval):
    """
    Returns a warning string if the period/interval combo is unsupported by yfinance.
    Returns None if the combination is valid.
    """
    # Treat 4h as 1h for limit checking since we map 4h -> 1h internally
    effective_interval = "1h" if interval == "4h" else interval

    # Intraday intervals that have strict maximum lookback windows in yfinance
    intraday_strict_limits = {
        "1m":  ["1d", "5d"],                       # yfinance only holds 1m data for ~7 days
        "5m":  ["1d", "5d", "1mo"],               # 5m data limited to ~60 days
        "15m": ["1d", "5d", "1mo", "3mo"],        # 15m data limited to ~60 days
        "30m": ["1d", "5d", "1mo", "3mo"],        # 30m data limited to ~60 days
    }

    if effective_interval in intraday_strict_limits:
        allowed_periods = intraday_strict_limits[effective_interval]
        if period not in allowed_periods:
            return (
                f"The '{interval}' timeframe only supports short historical ranges: "
                f"{', '.join(allowed_periods)}. Please select a shorter period or a longer timeframe."
            )

    return None  # No issues found — combination is valid


# --- Multi-Timeframe Intelligence Engine ---
# This helper fetches and analyzes a single timeframe for trend direction.
# Call it multiple times with different intervals to build a full market picture.
def analyze_timeframe(ticker_symbol, period, interval):
    """
    Fetches OHLCV data for one timeframe and classifies the trend.

    Returns a dictionary with:
        - interval : the timeframe analyzed  (e.g. '15m', '1h', '1d')
        - period   : the historical range    (e.g. '5d', '1mo', '1y')
        - price    : latest closing price    (float or None)
        - ma20     : 20-period moving average (float or None)
        - ma50     : 50-period moving average (float or None)
        - trend    : 'Bullish', 'Bearish', 'Mixed', or 'Not Enough Data'
        - status   : 'success', 'warning', or 'info'  (for Streamlit styling)
    """
    try:
        tf_data = get_stock_data(ticker_symbol, period, interval)

        # Need at least 20 candles to compute MA20
        if tf_data.empty or len(tf_data) < 20:
            return {
                "interval": interval,
                "period":   period,
                "price":    None,
                "ma20":     None,
                "ma50":     None,
                "trend":    "Not Enough Data",
                "status":   "info"
            }

        # Calculate rolling moving averages
        tf_data["MA20"] = tf_data["Close"].rolling(window=20).mean()
        tf_data["MA50"] = tf_data["Close"].rolling(window=50).mean()

        price    = tf_data["Close"].iloc[-1]
        ma20     = tf_data["MA20"].iloc[-1]    # Valid since len >= 20
        ma50_raw = tf_data["MA50"].iloc[-1]   # May be NaN if len < 50

        # Convert NaN to None for cleaner handling downstream
        ma50 = None if pd.isna(ma50_raw) else ma50_raw

        # Classify trend using price vs moving averages
        if ma50 is None:
            # Not enough candles for MA50 — use MA20 alone
            trend  = "Bullish" if price > ma20 else "Bearish"
            status = "success" if trend == "Bullish" else "warning"
        elif price > ma20 and price > ma50 and ma20 > ma50:
            trend  = "Bullish"
            status = "success"
        elif price < ma20 and price < ma50 and ma20 < ma50:
            trend  = "Bearish"
            status = "warning"
        else:
            trend  = "Mixed"
            status = "info"

        return {
            "interval": interval,
            "period":   period,
            "price":    price,
            "ma20":     ma20,
            "ma50":     ma50,
            "trend":    trend,
            "status":   status
        }

    except Exception:
        # If the fetch fails for any reason, return a safe fallback
        return {
            "interval": interval,
            "period":   period,
            "price":    None,
            "ma20":     None,
            "ma50":     None,
            "trend":    "Not Enough Data",
            "status":   "info"
        }


# ==============================================================
# INTELLIGENCE HELPERS — human-readable labels from raw numbers
# ==============================================================

def compute_market_bias(price, ma20, ma50):
    """Bullish = above both MAs. Bearish = below both. Mixed = between."""
    if price > ma20 and price > ma50:
        return "Bullish"
    elif price < ma20 and price < ma50:
        return "Bearish"
    return "Mixed"


def compute_momentum_status(rsi):
    """Convert RSI into plain-English momentum label."""
    if rsi > 70:
        return "Strong but Overextended"
    elif rsi < 30:
        return "Weak / Oversold"
    return "Neutral / Balanced"


def compute_risk_level(rsi, volatility, market_bias):
    """Risk is never fully safe — this provides a relative indication."""
    if rsi > 70 or rsi < 30:
        return "High"
    if market_bias == "Mixed" or volatility > 3.0:
        return "Medium"
    return "Low"


def compute_alignment_summary(mtf_results):
    """3+ timeframes agreeing = Bullish/Bearish Alignment. Otherwise Mixed."""
    bullish = sum(1 for r in mtf_results if r["trend"] == "Bullish")
    bearish = sum(1 for r in mtf_results if r["trend"] == "Bearish")
    if bullish >= 3:
        return "Bullish Alignment"
    elif bearish >= 3:
        return "Bearish Alignment"
    return "Mixed / Conflicting"


# ==============================================================
# TRADE READINESS SCORE
# Scores how well-aligned conditions are for a human to watch.
# This is NOT a trade signal — it is educational decision-support.
# Higher score = more conditions agree. Lower = more confirmation needed.
# ==============================================================
def calculate_trade_readiness(market_bias, setup_quality, risk_level, momentum_status,
                               alignment_summary, setup_type, warning_signs):
    score = setup_quality  # Base from setup quality

    # Risk always matters — higher risk reduces readiness
    if risk_level == "High":
        score -= 20
    elif risk_level == "Medium":
        score -= 10

    # Overextended momentum raises reversal risk
    if momentum_status == "Strong but Overextended":
        score -= 15

    # Timeframe conflict reduces conviction
    if alignment_summary == "Mixed / Conflicting":
        score -= 15

    # Bias mismatch with setup type penalizes further
    if market_bias == "Bearish" and "Bullish" in setup_type:
        score -= 20
    if market_bias == "Bullish" and "Bearish" in setup_type:
        score -= 20

    # Each warning sign reduces readiness slightly
    score -= len(warning_signs) * 5

    # Bonus: timeframe and bias aligned
    if market_bias == "Bullish" and "Bullish" in alignment_summary:
        score += 10
    if market_bias == "Bearish" and "Bearish" in alignment_summary:
        score += 10

    # Low risk adds confidence
    if risk_level == "Low":
        score += 10

    # Neutral momentum is a positive sign
    if momentum_status == "Neutral / Balanced":
        score += 5

    # Clamp to 0-100
    score = max(0, min(100, score))

    # Label and action based on score
    if score >= 80:
        label = "Strong Watch";       action = "WATCH"
    elif score >= 65:
        label = "Watchlist Candidate"; action = "WATCH"
    elif score >= 45:
        label = "Needs Confirmation";  action = "WAIT"
    elif score >= 25:
        label = "High Risk / Wait";    action = "WAIT"
    else:
        label = "No Clear Edge";       action = "AVOID FOR NOW"

    # Build plain-English reason
    reasons = []
    if risk_level == "High":
        reasons.append("risk is elevated")
    if momentum_status == "Strong but Overextended":
        reasons.append("momentum is overextended")
    if alignment_summary == "Mixed / Conflicting":
        reasons.append("timeframes are not aligned")
    if market_bias == "Mixed":
        reasons.append("market bias is unclear")
    reason = ("Conditions suggest " + ", ".join(reasons) + "." if reasons
              else "Conditions appear broadly favorable.")

    # What would make this setup cleaner
    what_would_improve = []
    if risk_level in ("High", "Medium"):
        what_would_improve.append("Risk cooling to a lower level")
    if momentum_status == "Strong but Overextended":
        what_would_improve.append("Momentum cooling from overextended levels")
    if alignment_summary == "Mixed / Conflicting":
        what_would_improve.append("More timeframes aligning in the same direction")
    if market_bias == "Mixed":
        what_would_improve.append("Price establishing a clearer trend above or below key moving averages")
    if score < 65:
        what_would_improve.append("Setup quality improving with more signal agreement")

    # What to avoid while waiting
    what_to_avoid = []
    if momentum_status == "Strong but Overextended":
        what_to_avoid.append("Chasing after a strong extended move")
    if risk_level == "High":
        what_to_avoid.append("Ignoring elevated risk conditions")
    if alignment_summary == "Mixed / Conflicting":
        what_to_avoid.append("Acting while timeframes are in conflict")
    what_to_avoid.append("Treating this score as a guaranteed outcome — human review required")

    return {
        "score":              score,
        "label":              label,
        "action":             action,
        "reason":             reason,
        "what_would_improve": what_would_improve,
        "what_to_avoid":      what_to_avoid,
    }


# ==============================================================
# DECISION CARD RENDERER
# Displays the MahkiVision Decision in a clean, human-readable block.
# WATCH / WAIT / AVOID FOR NOW are decision-support labels, not trade commands.
# ==============================================================
def render_decision_card(readiness, symbol=""):
    action  = readiness["action"]
    label   = readiness["label"]
    score   = readiness["score"]
    reason  = readiness["reason"]
    improve = readiness["what_would_improve"]
    avoid   = readiness["what_to_avoid"]
    title   = f"MahkiVision Decision{f' — {symbol}' if symbol else ''}"

    if action == "WATCH":
        st.success(f"#### {title}")
        st.success(f"**Action: {action}**  \u00b7  Trade Readiness: {score}/100 \u2014 {label}")
    elif action == "WAIT":
        st.warning(f"#### {title}")
        st.warning(f"**Action: {action}**  \u00b7  Trade Readiness: {score}/100 \u2014 {label}")
    else:
        st.error(f"#### {title}")
        st.error(f"**Action: {action}**  \u00b7  Trade Readiness: {score}/100 \u2014 {label}")

    st.caption(f"\U0001f4ac {reason}")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**What Would Improve This Setup:**")
        for item in improve:
            st.info(f"\u2191 {item}")
    with col_r:
        st.markdown("**What To Avoid:**")
        for item in avoid:
            st.warning(f"\u26a0 {item}")
    st.caption("Educational decision-support only. Not a trade signal. Human review required.")


def render_dashboard():
    st.title("Dashboard")
    st.caption("Your latest market intelligence at a glance.")
    if st.session_state.latest_symbol is None:
        st.info("Welcome to MahkiVision. Start by analyzing a stock, ETF, or forex pair in **Asset Analyzer**.")
        return

    sym   = st.session_state.latest_symbol
    atype = st.session_state.latest_asset_type
    bias  = st.session_state.latest_market_bias
    score = st.session_state.latest_setup_score
    risk  = st.session_state.latest_risk_level
    mom   = st.session_state.latest_momentum_status
    align = st.session_state.latest_alignment_summary

    st.subheader(f"Last Analyzed: {sym}")
    st.caption(f"Asset Type: {atype}")

    # Decision Card — show first so the user immediately sees WATCH / WAIT / AVOID
    tr_action = st.session_state.latest_decision_action
    if tr_action:
        readiness = {
            "score":              st.session_state.latest_trade_readiness_score,
            "label":              st.session_state.latest_trade_readiness_label,
            "action":             tr_action,
            "reason":             st.session_state.latest_decision_reason,
            "what_would_improve": st.session_state.latest_what_would_improve,
            "what_to_avoid":      st.session_state.latest_what_to_avoid,
        }
        render_decision_card(readiness, sym)

    st.markdown("---")

    # Market Snapshot
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Bias",        bias  or "—")
    c2.metric("Setup Quality",      f"{score}/100" if score is not None else "—")
    c3.metric("Risk Level",         risk  or "—")
    c4.metric("Momentum",           mom   or "—")
    st.metric("Timeframe Alignment", align or "—")

    if st.session_state.ai_summary:
        with st.expander("Latest AI Read"):
            st.write(st.session_state.ai_summary)

    st.markdown("---")
    st.caption("Go to **Asset Analyzer** to run a new analysis.")


# ==============================================================
# PAGE: ASSET ANALYZER
# ==============================================================
def render_asset_analyzer():
    st.title("Asset Analyzer")
    st.caption("Analyze any stock, ETF, or forex pair with multi-timeframe intelligence.")

    st.sidebar.markdown("### Analyzer Controls")
    asset_type = st.sidebar.selectbox(
        "Asset Type", ["Stock / ETF", "Forex"], key="asset_type"
    )
    asset_label = "forex pair" if asset_type == "Forex" else "stock or ETF ticker"
    ticker = st.sidebar.text_input(f"Enter {asset_label}:", key="ticker")
    if asset_type == "Forex":
        st.sidebar.caption("Examples: EURUSD, EUR/USD, GBPUSD, USDJPY")
    else:
        st.sidebar.caption("Examples: VOO, AAPL, QQQM, SPY, NVDA")

    period = st.sidebar.selectbox(
        "Historical Range",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        key="historical_range"
    )
    time_interval = st.sidebar.selectbox(
        "Candle Timeframe",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"],
        key="candle_timeframe"
    )
    analyze_button = st.sidebar.button("Analyze", key="aa_analyze")

    if analyze_button:
        st.session_state.analysis_ready = True

    if not st.session_state.analysis_ready:
        st.info("Configure your asset in the sidebar and click **Analyze**. Try VOO, AAPL, or EURUSD.")
        return

    # --- Validation ---
    if not ticker.strip():
        st.error("Please enter a ticker symbol or forex pair.")
        st.stop()

    if not validate_ticker(ticker, asset_type):
        if asset_type == "Forex":
            st.error(f"'{ticker}' is not a valid forex pair. Use 6-character pairs like EURUSD, EUR/USD, or GBPUSD.")
        else:
            st.error(f"'{ticker}' is not a valid stock/ETF ticker. Tickers are 1-5 characters (e.g., VOO, AAPL).")
        st.stop()

    normalized_ticker = normalize_ticker(ticker, asset_type)
    interval_warning = validate_interval_period(period, time_interval)
    if interval_warning:
        st.warning(interval_warning)
        st.stop()

    # --- Fetch Data ---
    with st.spinner(f"Fetching data for {normalized_ticker}..."):
        data = get_stock_data(normalized_ticker, period, time_interval)

    if data.empty:
        st.error(f"No data found for '{normalized_ticker}'. Check the symbol and try again.")
        st.stop()
    if len(data) < 2:
        st.error("Not enough data returned. Try a longer historical range.")
        st.stop()

    # --- Multi-Timeframe Analysis ---
    with st.spinner("Running multi-timeframe analysis..."):
        mtf_results = [
            analyze_timeframe(normalized_ticker, "5d",  "15m"),
            analyze_timeframe(normalized_ticker, "1mo", "1h"),
            analyze_timeframe(normalized_ticker, "6mo", "4h"),
            analyze_timeframe(normalized_ticker, "1y",  "1d"),
        ]

    # --- Core Calculations ---
    current_price  = data["Close"].iloc[-1]
    previous_price = data["Close"].iloc[-2]
    daily_change   = ((current_price - previous_price) / previous_price) * 100
    start_price    = data["Close"].iloc[0]
    period_return  = ((current_price - start_price) / start_price) * 100
    period_high    = data["Close"].max()
    period_low     = data["Close"].min()
    volatility     = data["Close"].pct_change().std() * 100

    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()
    current_ma20     = data["MA20"].iloc[-1]
    current_ma50_raw = data["MA50"].iloc[-1]
    current_ma50 = current_ma20 if pd.isna(current_ma50_raw) else current_ma50_raw

    # RSI
    delta    = data["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(window=14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs       = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    current_rsi = data["RSI"].iloc[-1]

    # Signal Score (Setup Quality)
    signal_score     = 0
    positive_signals = []
    negative_signals = []
    if current_price > current_ma20:
        signal_score += 25
        positive_signals.append("Price is above the 20-period moving average.")
    else:
        negative_signals.append("Price is below the 20-period moving average.")
    if current_price > current_ma50:
        signal_score += 25
        positive_signals.append("Price is above the 50-period moving average.")
    else:
        negative_signals.append("Price is below the 50-period moving average.")
    if period_return > 0:
        signal_score += 25
        positive_signals.append("Period return is positive.")
    else:
        negative_signals.append("Period return is negative.")
    if daily_change > 0:
        signal_score += 25
        positive_signals.append("Daily change is positive.")
    else:
        negative_signals.append("Daily change is negative.")

    # Derived Intelligence Labels
    market_bias       = compute_market_bias(current_price, current_ma20, current_ma50)
    momentum_status   = compute_momentum_status(current_rsi)
    risk_level        = compute_risk_level(current_rsi, volatility, market_bias)
    alignment_summary = compute_alignment_summary(mtf_results)

    # Market Regime
    if current_rsi > 70:
        market_regime = "Overbought Rally"; market_regime_type = "warning"
    elif signal_score >= 75 and current_rsi <= 70:
        market_regime = "Strong Bullish Momentum"; market_regime_type = "success"
    elif signal_score <= 25 and current_rsi < 40:
        market_regime = "Bearish Weakness"; market_regime_type = "warning"
    else:
        market_regime = "Neutral Consolidation"; market_regime_type = "info"

    # Trade Setup Intelligence
    confirmation_signals = []
    invalidation_notes   = []
    if signal_score >= 75 and current_rsi < 70:
        setup_type  = "Bullish Continuation Watch"
        setup_score = signal_score
        confirmation_signals.append("Multiple bullish conditions are currently active.")
        confirmation_signals.append("RSI is below overbought territory — momentum not yet stretched.")
        invalidation_notes.append("A close below MA20 may weaken the bullish structure.")
        invalidation_notes.append("A sharp volatility spike could increase downside risk.")
    elif signal_score >= 75 and current_rsi >= 70:
        setup_type  = "Bullish but Overextended"
        setup_score = signal_score - 10
        risk_level  = "High"
        confirmation_signals.append("Trend structure remains positive across multiple conditions.")
        invalidation_notes.append("RSI above 70 — short-term pullback risk is elevated.")
        invalidation_notes.append("Momentum may be overextended. Watch for reversal signals.")
    elif signal_score <= 25:
        setup_type  = "Bearish Weakness Watch"
        setup_score = signal_score
        risk_level  = "High"
        confirmation_signals.append("Low setup quality suggests weak technical conditions.")
        invalidation_notes.append("A recovery above key moving averages may shift the outlook.")
    else:
        setup_type  = "Neutral / No Clear Edge"
        setup_score = signal_score
        confirmation_signals.append("Mixed signals — no strong directional setup detected.")
        invalidation_notes.append("Wait for clearer confirmation from price, trend, and momentum.")

    # Short AI status for decision card
    if market_bias == "Bullish" and risk_level != "High":
        ai_status = "Bullish structure forming. Watch for momentum confirmation."
    elif market_bias == "Bullish" and risk_level == "High":
        ai_status = "Bullish but overextended. Short-term pullback risk is elevated."
    elif market_bias == "Bearish":
        ai_status = "Bearish conditions present. Risk is elevated — proceed with caution."
    else:
        ai_status = "Mixed signals. No clear directional edge at this time."

    # Calculate Trade Readiness — scores how aligned conditions are
    # This is NOT a trade signal. It is educational decision-support.
    readiness = calculate_trade_readiness(
        market_bias, setup_score, risk_level, momentum_status,
        alignment_summary, setup_type, invalidation_notes
    )

    # --- Save Snapshot for Other Pages ---
    st.session_state.latest_symbol            = normalized_ticker
    st.session_state.latest_asset_type        = asset_type
    st.session_state.latest_market_bias       = market_bias
    st.session_state.latest_setup_score       = setup_score
    st.session_state.latest_risk_level        = risk_level
    st.session_state.latest_momentum_status   = momentum_status
    st.session_state.latest_alignment_summary       = alignment_summary
    st.session_state.latest_trade_readiness_score   = readiness["score"]
    st.session_state.latest_trade_readiness_label   = readiness["label"]
    st.session_state.latest_decision_action         = readiness["action"]
    st.session_state.latest_decision_reason         = readiness["reason"]
    st.session_state.latest_what_would_improve      = readiness["what_would_improve"]
    st.session_state.latest_what_to_avoid           = readiness["what_to_avoid"]

    mtf_lines = []
    for r in mtf_results:
        p_s = f"${r['price']:.2f}" if r["price"] is not None else "N/A"
        mtf_lines.append(f"  {r['interval']} ({r['period']}): Bias={r['trend']}, Price={p_s}")

    market_context = f"""Asset Type: {asset_type}
User Input: {ticker}
Symbol Used: {normalized_ticker}
Historical Range: {period}
Candle Timeframe: {time_interval}
Current Price: ${current_price:.2f}
Daily Change: {daily_change:.2f}%
Period Return: {period_return:.2f}%
Setup Quality: {signal_score}/100
RSI: {current_rsi:.2f}
Momentum Status: {momentum_status}
Market Bias: {market_bias}
Market Regime: {market_regime}
Risk Level: {risk_level}
Setup Type: {setup_type}
Setup Score: {setup_score}/100
Positive Signals: {positive_signals}
Negative Signals: {negative_signals}
What to Watch: {confirmation_signals}
Warning Signs: {invalidation_notes}
Multi-Timeframe Analysis:
{chr(10).join(mtf_lines)}
Alignment Summary: {alignment_summary}
Trade Readiness Score: {readiness["score"]}/100
Trade Readiness Label: {readiness["label"]}
Decision Action: {readiness["action"]}
Decision Reason: {readiness["reason"]}
What Would Improve: {readiness["what_would_improve"]}
What To Avoid: {readiness["what_to_avoid"]}"""

    st.session_state.latest_market_context = market_context

    # ============================================================
    # DISPLAY — Decision Card first, details below
    # ============================================================

    # 1. MahkiVision Decision Card — the main answer to "should I watch this?"
    st.subheader(f"{normalized_ticker} — Analysis")
    st.caption(f"Asset Type: {asset_type}  ·  Range: {period}  ·  Timeframe: {time_interval}")
    render_decision_card(readiness, normalized_ticker)
    st.markdown("---")

    # 2. Market Snapshot — 5 clean metrics, no raw numbers
    st.subheader("Market Snapshot")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Market Bias",        market_bias)
    c2.metric("Setup Quality",      f"{signal_score}/100")
    c3.metric("Risk Level",         risk_level)
    c4.metric("Momentum",           momentum_status)
    c5.metric("Timeframe Alignment", alignment_summary)
    st.markdown("---")

    # 3. Price Chart (left) + Strengths / Warning Signs (right)
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.subheader("Price Chart")
        chart_data = data.reset_index()
        x_axis_col = "Date" if "Date" in chart_data.columns else "Datetime"
        fig = px.line(chart_data, x=x_axis_col, y=["Close", "MA20", "MA50"],
                      title=f"{normalized_ticker} — Short-Term Trend (MA20) & Medium-Term Trend (MA50)")
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("Setup Signals")
        if positive_signals:
            st.markdown("**Strengths:**")
            for s in positive_signals:
                st.success(s)
        if negative_signals:
            st.markdown("**Warning Signs:**")
            for s in negative_signals:
                st.warning(s)
        st.markdown("**Momentum:**")
        if current_rsi > 70:
            st.warning(f"Momentum — Overextended ({current_rsi:.1f})")
        elif current_rsi < 30:
            st.success(f"Momentum — Oversold ({current_rsi:.1f})")
        else:
            st.info(f"Momentum — Neutral ({current_rsi:.1f})")
        st.markdown("**Market Condition:**")
        if market_regime_type == "success":
            st.success(market_regime)
        elif market_regime_type == "warning":
            st.warning(market_regime)
        else:
            st.info(market_regime)

    st.markdown("---")

    # 4. Trade Setup Intelligence
    st.subheader("Trade Setup Intelligence")
    ca, cb, cc = st.columns(3)
    ca.metric("Setup Type",    setup_type)
    cb.metric("Setup Quality", f"{setup_score}/100")
    cc.metric("Risk Level",    risk_level)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**What to Watch:**")
        for s in confirmation_signals:
            st.success(s)
    with col_r:
        st.markdown("**Warning Signs:**")
        for n in invalidation_notes:
            st.warning(n)

    st.markdown("---")

    # 5. Timeframe Alignment (renamed from Multi-Timeframe Analysis)
    st.subheader("Timeframe Alignment")
    st.caption("When multiple timeframes agree, it may indicate stronger directional conviction.")
    mtf_cols = st.columns(len(mtf_results))
    for i, result in enumerate(mtf_results):
        with mtf_cols[i]:
            st.markdown(f"**{result['interval']} / {result['period']}**")
            p_str = f"${result['price']:.2f}" if result["price"] is not None else "N/A"
            st.metric("Price",              p_str)
            st.metric("Short-Term Trend",   f"${result['ma20']:.2f}" if result["ma20"] is not None else "N/A")
            st.metric("Medium-Term Trend",  f"${result['ma50']:.2f}" if result["ma50"] is not None else "N/A")
            if result["status"] == "success":
                st.success(f"Bias: {result['trend']}")
            elif result["status"] == "warning":
                st.warning(f"Bias: {result['trend']}")
            else:
                st.info(f"Bias: {result['trend']}")

    if "Bullish" in alignment_summary:
        st.success(f"Alignment: {alignment_summary}")
    elif "Bearish" in alignment_summary:
        st.warning(f"Alignment: {alignment_summary}")
    else:
        st.info(f"Alignment: {alignment_summary}")

    st.markdown("---")

    # 6. AI Intelligence
    st.subheader("AI Intelligence")
    st.caption("AI outputs are for educational purposes only. Not financial advice.")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("Generate AI Market Read"):
            with st.spinner("Generating AI market read..."):
                st.session_state.ai_summary = generate_ai_summary(market_context)
    with col_btn2:
        if st.button("Generate AI Setup Review"):
            with st.spinner("Analyzing setup conditions..."):
                st.session_state.trade_setup = generate_trade_setup_analysis(market_context)
    with col_btn3:
        if st.button("Clear AI Outputs"):
            st.session_state.ai_summary  = ""
            st.session_state.trade_setup = ""
            st.rerun()

    if st.session_state.ai_summary:
        st.subheader("MahkiVision AI Read")
        st.write(st.session_state.ai_summary)
    if st.session_state.trade_setup:
        st.subheader("AI Setup Review")
        st.write(st.session_state.trade_setup)

    st.markdown("---")

    # 7. Technical Details — collapsed, for power users
    with st.expander("View Technical Indicators"):
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Price",         f"${current_price:.2f}")
        m2.metric("Daily Change",  f"{daily_change:.2f}%")
        m3.metric("Period Return", f"{period_return:.2f}%")
        m4.metric("Period High",   f"${period_high:.2f}")
        m5.metric("Volatility",    f"{volatility:.2f}%")
        m6.metric("RSI",           f"{current_rsi:.2f}")
    with st.expander("View Raw Market Data"):
        st.dataframe(data.tail(10))
    with st.expander("View AI Context (sent to AI)"):
        st.code(market_context)


# ==============================================================
# PAGE: OPPORTUNITIES
# ==============================================================
def render_opportunities():
    st.title("Opportunities")
    st.caption("Potential setups worth monitoring based on your latest analysis.")
    if st.session_state.latest_symbol is None:
        st.info("Run an asset analysis in **Asset Analyzer** first to generate opportunity candidates.")
        st.caption("Future version: automatically scan multiple stocks, ETFs, and forex pairs.")
        return

    sym       = st.session_state.latest_symbol
    atype     = st.session_state.latest_asset_type
    bias      = st.session_state.latest_market_bias
    score     = st.session_state.latest_setup_score
    risk      = st.session_state.latest_risk_level
    align     = st.session_state.latest_alignment_summary
    tr_score  = st.session_state.latest_trade_readiness_score
    tr_label  = st.session_state.latest_trade_readiness_label
    tr_action = st.session_state.latest_decision_action
    tr_reason = st.session_state.latest_decision_reason
    tr_improve = st.session_state.latest_what_would_improve
    tr_avoid   = st.session_state.latest_what_to_avoid

    st.subheader("Current Opportunity Review")
    c1, c2, c3 = st.columns(3)
    c1.metric("Symbol",            sym)
    c2.metric("Asset Type",        atype)
    c3.metric("Market Bias",       bias or "—")
    c4, c5, c6 = st.columns(3)
    c4.metric("Setup Quality",     f"{score}/100" if score is not None else "—")
    c5.metric("Risk Level",        risk  or "—")
    c6.metric("Timeframe Alignment", align or "—")

    if tr_action:
        readiness = {
            "score": tr_score, "label": tr_label, "action": tr_action,
            "reason": tr_reason, "what_would_improve": tr_improve, "what_to_avoid": tr_avoid,
        }
        render_decision_card(readiness, sym)
    elif bias == "Bullish":
        st.success("Bullish setup conditions are present. Monitor for confirmation before acting.")
    elif bias == "Bearish":
        st.warning("Bearish conditions present. Risk is elevated.")
    else:
        st.info("Mixed signals. Wait for clearer directional confirmation.")

    if tr_score is not None and tr_score < 45:
        st.warning("This asset is not a strong opportunity yet. It may need more confirmation before it is worth watching closely.")

    st.markdown("---")
    st.info("**Future version:** MahkiVision will automatically scan multiple stocks, ETFs, forex pairs, and crypto assets to surface high-probability setups.")


# ==============================================================
# PAGE: AI ALERTS
# ==============================================================
def render_ai_alerts():
    st.title("AI Alerts")
    st.caption("Rule-based alert cards generated from your latest analysis. Not trade signals.")
    if st.session_state.latest_symbol is None:
        st.info("Run an asset analysis in **Asset Analyzer** first to generate alert candidates.")
        return

    sym       = st.session_state.latest_symbol
    bias      = st.session_state.latest_market_bias
    score     = st.session_state.latest_setup_score
    risk      = st.session_state.latest_risk_level
    mom       = st.session_state.latest_momentum_status
    align     = st.session_state.latest_alignment_summary
    tr_action = st.session_state.latest_decision_action
    tr_score  = st.session_state.latest_trade_readiness_score
    tr_label  = st.session_state.latest_trade_readiness_label

    st.subheader(f"Alert Candidates for {sym}")
    alerts = []  # Each: (type, title, summary, detail)

    if tr_action == "WATCH" and tr_score is not None and tr_score >= 65:
        alerts.append(("success", "Strong Watch Candidate",
                        f"{sym} has a Trade Readiness score of {tr_score}/100 ({tr_label}).",
                        "Conditions may be developing. Wait for confirmation before acting."))

    if tr_action == "WAIT":
        alerts.append(("warning", "Wait for Confirmation",
                        f"{sym} is rated '{tr_label}' ({tr_score}/100).",
                        "More conditions need to align before this setup may be worth watching closely."))

    if bias == "Bullish" and risk == "High":
        alerts.append(("warning", "Pullback Risk Elevated",
                        f"{sym} shows bullish conditions but risk is high.",
                        "A short-term pullback or reversal may be developing. Monitor before acting."))

    if mom == "Strong but Overextended":
        alerts.append(("warning", "Momentum Overextended",
                        f"{sym} momentum is strong but stretched.",
                        "Waiting for momentum to cool or for support confirmation may reduce risk of chasing."))

    if align == "Mixed / Conflicting":
        alerts.append(("info", "Timeframe Conflict",
                        f"Timeframes are not aligned for {sym}.",
                        "Conflicting signals across timeframes reduce setup clarity. Wait for alignment."))

    if bias == "Bearish":
        alerts.append(("warning", "Bearish Conditions Present",
                        f"{sym} is showing bearish structure.",
                        "Monitor for further weakness. Risk is elevated."))

    if tr_action == "AVOID FOR NOW" or (score is not None and score <= 25):
        alerts.append(("warning", "No Clear Edge",
                        f"{sym} does not show a strong setup at this time.",
                        "Conditions may need significant improvement before this becomes worth monitoring."))

    if not alerts:
        alerts.append(("info", "No High-Conviction Alerts",
                        "Conditions appear neutral.",
                        "No strong alert conditions detected from the latest analysis."))

    for alert_type, title, summary, detail in alerts:
        if alert_type == "success":
            st.success(f"**{title}** — {summary}")
        elif alert_type == "warning":
            st.warning(f"**{title}** — {summary}")
        else:
            st.info(f"**{title}** — {summary}")
        st.caption(f"   {detail}")

    st.markdown("---")
    st.caption("These are rule-based alert candidates. They are NOT trade signals. Always conduct your own research.")


# ==============================================================
# PAGE: TRADE JOURNAL
# ==============================================================
def render_trade_journal():
    st.title("Trade Journal")
    st.caption("Record your analysis reasoning. Learning from setups is how you improve.")

    # Prefill from latest analysis if user clicks the button
    # Use session state flags so the form can reflect prefill values
    if "_journal_prefill" not in st.session_state:
        st.session_state._journal_prefill = False

    if st.button("Use Latest Analysis", help="Prefill the form with the most recent analysis data."):
        st.session_state._journal_prefill = True

    pf = st.session_state._journal_prefill
    pf_sym    = st.session_state.latest_symbol    or "" if pf else ""
    pf_bias   = st.session_state.latest_market_bias or "Neutral"
    pf_action = st.session_state.latest_decision_action or ""
    pf_score  = str(st.session_state.latest_trade_readiness_score or "")

    st.subheader("Add a Journal Entry")
    with st.form("trade_journal_form"):
        col1, col2 = st.columns(2)
        j_symbol  = col1.text_input("Symbol",     value=pf_sym,  placeholder="e.g. VOO, EURUSD")
        j_atype   = col2.selectbox("Asset Type",  ["Stock / ETF", "Forex", "Other"])
        col3, col4 = st.columns(2)
        j_bias    = col3.selectbox("Bias at Entry", ["Bullish", "Bearish", "Neutral"],
                                   index=["Bullish", "Bearish", "Neutral"].index(pf_bias) if pf_bias in ["Bullish", "Bearish", "Neutral"] else 2)
        j_action  = col4.text_input("Decision Action at Time", value=pf_action, placeholder="WATCH / WAIT / AVOID FOR NOW")
        j_readiness = st.text_input("Trade Readiness Score", value=pf_score, placeholder="e.g. 72")
        j_reason  = st.text_area("Reason for Watching", placeholder="What conditions led you to analyze this asset?")
        j_result  = st.selectbox("Outcome (if known)", ["Pending", "Confirmed Bullish", "Confirmed Bearish", "Invalidated"])
        j_lesson  = st.text_area("Lesson Learned", placeholder="What did this setup teach you about market behavior?")
        submitted = st.form_submit_button("Add Entry")

    if submitted:
        if not j_symbol.strip():
            st.error("Please enter a symbol.")
        else:
            entry = {
                "Symbol":          j_symbol.upper().strip(),
                "Asset Type":      j_atype,
                "Bias":            j_bias,
                "Decision Action": j_action,
                "Readiness Score": j_readiness,
                "Reason":          j_reason,
                "Result":          j_result,
                "Lesson":          j_lesson,
            }
            st.session_state.trade_journal.append(entry)
            st.session_state._journal_prefill = False
            st.success(f"Entry added for {entry['Symbol']}.")

    st.markdown("---")

    # Journal Entries display
    st.subheader("Journal Entries")
    if not st.session_state.trade_journal:
        st.info("No journal entries yet. Add your first entry above.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.trade_journal), use_container_width=True)

    # Journal Insights — prototype AI memory foundation
    # This is the beginning of pattern recognition from past setups.
    st.markdown("---")
    st.subheader("Journal Insights")
    entries = st.session_state.trade_journal
    if not entries:
        st.info("Add journal entries to start seeing insights.")
    else:
        total     = len(entries)
        wins      = sum(1 for e in entries if e.get("Result") == "Confirmed Bullish")
        losses    = sum(1 for e in entries if e.get("Result") == "Confirmed Bearish")
        invalid   = sum(1 for e in entries if e.get("Result") == "Invalidated")
        pending   = sum(1 for e in entries if e.get("Result") == "Pending")

        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Total Entries",    total)
        i2.metric("Confirmed Bullish", wins)
        i3.metric("Confirmed Bearish", losses)
        i4.metric("Invalidated",       invalid)

        lessons = [e.get("Lesson", "").strip() for e in entries if e.get("Lesson", "").strip()]
        if lessons:
            st.markdown("**Recent Lessons:**")
            for lesson in lessons[-3:]:
                st.info(f"\u201c{lesson}\u201d")

        if losses > 0 or invalid > 0:
            st.warning(f"You have {losses + invalid} entries marked as bearish or invalidated. Review whether high-risk setups are leading to weaker outcomes.")
        elif wins >= 2:
            st.success(f"You have {wins} bullish confirmations. Review what conditions were common in those setups.")
        else:
            st.info("Keep adding entries. Patterns will emerge over time.")
        st.caption("Journal Insights are a prototype AI memory feature. More pattern analysis coming in future versions.")


# ==============================================================
# PAGE: SETTINGS / ABOUT
# ==============================================================
def render_settings_about():
    st.title("Settings / About")
    st.subheader("About MahkiVision")
    st.markdown("""
**App Name:** MahkiVision Prototype  
**Purpose:** AI-powered market decision-support and educational research tool  
**Asset Support:** Stocks, ETFs, Forex pairs  
**Stage:** Prototype / MVP  
""")
    st.markdown("---")
    st.subheader("Disclaimer")
    st.warning("""**Important — Please Read:**

MahkiVision is an educational research tool only.
- It does **NOT** provide financial advice.
- It does **NOT** guarantee profits or predict future outcomes.
- All analysis is probability-based and for informational purposes only.
- Always consult a qualified financial professional before making investment decisions.
- Past market conditions shown do not guarantee future results.""")
    st.markdown("---")
    st.subheader("Future Roadmap")
    for item in [
        "Real-time push alerts and notifications",
        "Watchlists with automatic multi-asset scanning",
        "User accounts and saved analysis history",
        "News ingestion and sentiment analysis",
        "Mobile application",
        "Broker and live data integrations",
        "Backtesting engine",
        "Portfolio-level risk view",
    ]:
        st.markdown(f"- {item}")
    st.markdown("---")
    st.subheader("Technical Stack")
    st.markdown("""
- **Framework:** Streamlit  
- **Market Data:** yfinance  
- **AI Engine:** OpenAI GPT  
- **Charts:** Plotly  
- **API Key:** Loaded from `.env` — never exposed in code  
""")


# ==============================================================
# PAGE ROUTER
# ==============================================================
if page == "Dashboard":
    render_dashboard()
elif page == "Asset Analyzer":
    render_asset_analyzer()
elif page == "Opportunities":
    render_opportunities()
elif page == "AI Alerts":
    render_ai_alerts()
elif page == "Trade Journal":
    render_trade_journal()
elif page == "Settings / About":
    render_settings_about()
