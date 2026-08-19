import os
import datetime
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
import firebase_admin
from firebase_admin import credentials, firestore
from chart_analyser import analyze_chart
import sys

# Initialize Flask and enable CORS
app = Flask(__name__)
CORS(app)


@app.route('/analysechart', methods=['POST', 'OPTIONS'])
def analyze_chart_route():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight OK'})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200

    return analyze_chart()



def get_client_ip():
    if 'X-Forwarded-For' in request.headers:
        return request.headers['X-Forwarded-For'].split(',')[0].strip()
    return request.remote_addr


# ✅ Log IP address for debugging
@app.before_request
def log_ip():
    print("Client IP:", get_client_ip())


# Initialize Limiter (right here 👇)
limiter = Limiter(
    key_func=get_client_ip,
    app=app,
    default_limits=["10 per day"]
)

WHITELISTED_IPS = {
    "110.226.101.251",                    # Your local IPv4
    "2401:4900:1ca9:ab4c:ad0a:3d4f:d81c:625d"  # Your IPv6 from logs
}

import logging
logging.basicConfig(level=logging.INFO)
# ✅ Whitelist IPs and allow OPTIONS (CORS preflight) to bypass rate limit
@limiter.request_filter
def whitelist_ip():
    client_ip = get_client_ip()
    sys.stdout.write(f"🔍 Limiter sees IP: {client_ip}\n")
    logging.info(f"🔍 Limiter sees IP: {client_ip}")
    return request.method == "OPTIONS" or client_ip in WHITELISTED_IPS

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()

# ===== Firebase Config Fetch =====
def get_dhan_config():
    config_doc = db.collection("config").document("dhan_api").get()
    if not config_doc.exists:
        raise ValueError("Missing Dhan API config in Firebase")
    config = config_doc.to_dict()
    return config["apiToken"], config["apiUrl"] + config["route_historical"], config["apiUrl"] + config["route_intraday"], int(config.get("intraday_cutoff_hour", 10))

# ===== Technical Computations =====
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_ema(series, period=25):
    return series.ewm(span=period, adjust=False).mean()

def detect_breakout(df, rsi, volume):
    recent_high = df['High'][:-1].max()
    recent_low = df['Low'][:-1].min()
    last_close = df['Close'].iloc[-1]
    last_rsi = rsi.iloc[-1] if not rsi.isna().iloc[-1] else 50
    last_vol = volume.iloc[-1]
    avg_vol = volume.rolling(window=5).mean().iloc[-1]
    if last_close > recent_high and last_rsi > 55 and last_vol > avg_vol:
        return 'breakout', recent_high
    elif last_close < recent_low and last_rsi < 45 and last_vol > avg_vol:
        return 'breakdown', recent_low
    return None, None

# ===== API Logic =====
def fetch_dhan_data(security_id, is_intraday, headers, api_url):
    today = datetime.datetime.now().date().isoformat()
    payload = {
        "securityId": str(security_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "fromDate": today if is_intraday else (datetime.datetime.now().date() - datetime.timedelta(days=50)).isoformat(),
        "toDate": today
    }
    r = requests.post(api_url, headers=headers, json=payload)
    if r.status_code != 200:
        return None
    data = r.json()
    df = pd.DataFrame({
        "Open": data["open"],
        "High": data["high"],
        "Low": data["low"],
        "Close": data["close"],
        "Volume": data["volume"]
    })
    df.index = pd.to_datetime(data["timestamp"], unit='s')
    return df

# ===== Reason and Rich Formatting =====
def format_reason_and_rich(confidence_score, ema_trend, rsi_val, vol_spike, action, breakout_type=None, breakout_level=None):
    reason = f"Confidence Score: {confidence_score}. EMA trend is {ema_trend}, RSI at {rsi_val}, Volume spike: {vol_spike}."
    if action == "BUY":
        reason += " Strong bullish bias based on technicals."
        if breakout_type == "breakout":
            reason += f" Possible breakout above ₹{round(breakout_level, 2)}."  
    elif action == "SELL":
        reason += " Technical indicators suggest weakness."
        if breakout_type == "breakdown":
            reason += f" Possible breakdown below ₹{round(breakout_level, 2)}."
    else:
        reason += " Mixed signals, consider watching further."

    reason_rich = [
        {"label": "Confidence Score", "value": str(confidence_score), "type": "score"},
        {"label": "EMA trend", "value": ema_trend, "type": "trend"},
        {"label": "RSI", "value": str(rsi_val), "type": "rsi"},
        {"label": "Volume spike", "value": vol_spike, "type": "volume", "highlight": vol_spike.lower() == "no"},
    ]
    if action == "BUY":
        reason_rich.append({"label": "Summary", "value": "Strong bullish bias based on technicals", "type": "summary"})
        if breakout_type == "breakout":
            reason_rich.append({"label": "Breakout", "value": f"Above ₹{round(breakout_level, 2)}", "type": "breakout"})
    elif action == "SELL":
        reason_rich.append({"label": "Summary", "value": "Weakness in technicals", "type": "summary"})
        if breakout_type == "breakdown":
            reason_rich.append({"label": "Breakdown", "value": f"Below ₹{round(breakout_level, 2)}", "type": "breakdown"})
    else:
        reason_rich.append({"label": "Summary", "value": "Mixed signals", "type": "summary"})

    return reason_rich


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze_stock():
    log_msg = "🔍 analyze_stock1 triggered\n"
    sys.stdout.write(log_msg)         # ✅ Will show in basic logs (stdout)
    logging.info(log_msg.strip())     # ✅ Will show in Google Cloud Logs (structured)

    if request.method == 'OPTIONS':
        method_log = f"⚙️ OPTIONS request received from: {request.headers.get('Origin')}\n"
        sys.stdout.write(method_log)
        logging.info(method_log.strip())
        # Handle CORS preflight request
        response = jsonify({'message': 'CORS preflight OK'})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        return response, 200
    try:
        req = request.get_json()
        symbol = req.get("symbol")
        security_id = req.get("security_id")
        analysis_type = req.get("analysis_type", "INTRADAY").upper()

        log_msg = (
        f"📩 Request received:\n"
        f"  🔹 Symbol: {symbol}\n"
        f"  🔹 Security ID: {security_id}\n"
        f"  🔹 Type: {analysis_type}\n"
        f"  🔹 Full JSON: {req}\n"
        )
        sys.stdout.write(log_msg)
        logging.info(log_msg.strip())

        if not symbol or not security_id:
            response = jsonify({"error": "Missing symbol or security_id"})
            response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
            return response, 400

        token, hist_url, intra_url, raw_cutoff = get_dhan_config()
        cutoff_hour = int(raw_cutoff) if isinstance(raw_cutoff, str) else raw_cutoff

        now = datetime.datetime.now()

        if analysis_type == "INTRADAY" and now.hour < cutoff_hour:
            response= jsonify({"error": f"INTRADAY_BEFORE_CUTOFF"})        
            response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
            return response, 403


        headers = {"access-token": token, "accept": "application/json"}
        api_url = intra_url if analysis_type == "INTRADAY" else hist_url
        today = datetime.datetime.now().strftime('%Y_%m_%d')
        doc_key = f"{analysis_type.lower()}_{today}"
        doc_ref = db.collection("stock_analysis").document(doc_key)

        existing_doc = doc_ref.get()
        if existing_doc.exists and symbol.upper() in existing_doc.to_dict():
            response = jsonify(existing_doc.to_dict()[symbol.upper()])            
            response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
            return response, 200

        df = fetch_dhan_data(security_id, analysis_type == "INTRADAY", headers, api_url)
        if df is None or len(df) < 25:
            response = jsonify({"error": "Insufficient data"})            
            response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
            return response, 404

        rsi = compute_rsi(df["Close"], 14).fillna(50)
        ema_9 = compute_ema(df["Close"], 9)
        ema_20 = compute_ema(df["Close"], 20)
        ema_50 = compute_ema(df["Close"], 50)

        cmp = round(df["Close"].iloc[-1], 2)
        rsi_val = round(rsi.iloc[-1], 2)
        crossover = df["Close"].iloc[-1] > ema_9.iloc[-1] and df["Close"].iloc[-1] > ema_20.iloc[-1]
        momentum = (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
        volume_score = df["Volume"].mean()
        rsi_score = rsi_val / 100

        raw_score = momentum * volume_score * rsi_score * (1.1 if crossover else 1)
        confidence_score = min(int(raw_score * 1000), 100)

        ema_trend = "Up" if ema_20.iloc[-1] > ema_50.iloc[-1] else "Down"
        ema_cross = "Bullish" if crossover else "Bearish"
        vol_spike = "Yes" if df["Volume"].iloc[-1] > df["Volume"].rolling(5).mean().iloc[-1] else "No"
        breakout_type, breakout_level = detect_breakout(df, rsi, df["Volume"])

        if confidence_score > 70:
            action = "BUY"
        elif confidence_score < 30:
            action = "SELL"
        else:
            action = "HOLD"

        trend_strength = "Strong" if confidence_score > 70 else ("Weak" if confidence_score < 30 else "Moderate")

        if action == "BUY":
            sl = round(cmp * 0.985, 2)
            target = round(cmp * 1.02, 2)
        elif action == "SELL":
            sl = round(cmp * 1.02, 2)
            target = round(cmp * 0.985, 2)
        else:
            sl = round(cmp * 0.98, 2)
            target = round(cmp * 1.02, 2)

        # Validate target direction vs action
        if action == "BUY" and target < cmp:
            action = "HOLD"
        elif action == "SELL" and target > cmp:
            action = "HOLD"

        reason_rich = format_reason_and_rich(confidence_score, ema_trend, rsi_val, vol_spike, action, breakout_type,breakout_level)

        result = {
            "symbol": symbol,
            "security_id": security_id,
            "analysis_type": analysis_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "cmp": cmp,
            "sl": sl,
            "target": target,
            "sector": "Auto",
            "reason": reason_rich,
            "indicator_summary": {
                "rsi": rsi_val,
                "ema_trend": ema_trend,
                "ema_crossover": ema_cross,
                "volume_spike": vol_spike
            },
            "confidence_score": confidence_score,
            "trend_strength": trend_strength,
            "chart_tags": ["breakout", "volume", "rising RSI"] if breakout_type else [],
            "chart_url": f"https://firebase.storage/{symbol.lower()}_chart.png",
            "valid_until": (datetime.datetime.now() + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
            "is_manual": False
        }

        if existing_doc.exists:
            doc_ref.update({symbol.upper(): result})
        else:
            doc_ref.set({symbol.upper(): result})

        response = jsonify(result)        
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        return response, 200

    except Exception as e:
        error_message = f"❌ Error: {str(e)}\n"
        sys.stdout.write(error_message)                # Plain stdout logging
        logging.info(error_message)                    # Cloud logging
        
        response = jsonify({"error": str(e)})
        response.headers.add("Access-Control-Allow-Origin", request.headers.get('Origin'))
        return response, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # use dynamic port
    app.run(host="0.0.0.0", port=port,debug=True)

