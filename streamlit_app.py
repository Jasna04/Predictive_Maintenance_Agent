"""
Predictive Maintenance Streamlit Agent
- Dashboard, Trends, Logs, Insights
- Automatic A2A/RAG agent triggered for high-risk predictions (< threshold_days)
- SendGrid email alerts for high-risk predictions
- Stores RAG logs in session state
- Fully A2A compatible
"""

import os
import re
import json
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# Optional plotting & explainability
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except:
    MATPLOTLIB_AVAILABLE = False

# LLMs
try:
    from google import genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# SendGrid
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except:
    SENDGRID_AVAILABLE = False

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Predictive Maintenance (A2A Demo)", layout="wide")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
GEMINI_AVAILABLE = GEMINI_AVAILABLE and bool(GOOGLE_API_KEY)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_model(path="prediction_model.joblib"):
    return joblib.load(path)

def one_hot_encode_input(base_inputs, device_id, device_type, model_features):
    inp = base_inputs.copy()
    for feat in model_features:
        if feat.startswith("device_id_"):
            val = feat.replace("device_id_", "")
            inp[feat] = 1 if device_id == val else 0
        if feat.startswith("device_type_"):
            val = feat.replace("device_type_", "")
            inp[feat] = 1 if device_type == val else 0
    df = pd.DataFrame([inp])
    df = df.reindex(columns=model_features, fill_value=0)
    return df

def format_prediction_msg(pred, threshold_days=30):
    if pred < 25:
        return "🔴", "red", f"⚠️ HIGH FAILURE RISK (Predicted Days Left: {pred:.1f})"
    elif pred < 50:
        return "🟠", "orange", f"⚠️ Moderate Risk — Failure Soon (Predicted Days Left: {pred:.1f})"
    else:
        return "🟢", "green", f"✅ Equipment Healthy (Predicted Days Left: {pred:.1f})"

def plotly_gauge(value, color, max_range=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Predicted Days to Failure"},
        gauge={
            'axis': {'range': [0, max_range]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, max_range*0.25], 'color':'rgba(255,0,0,0.3)'},
                {'range': [max_range*0.25, max_range*0.5], 'color':'rgba(255,165,0,0.3)'},
                {'range': [max_range*0.5, max_range], 'color':'rgba(0,255,0,0.3)'}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
    return fig

def serpapi_search(query, num_results=5):
    if not SERPAPI_KEY:
        return []
    url = "https://serpapi.com/search.json"
    params = {"q": query, "engine": "google", "num": num_results, "api_key": SERPAPI_KEY}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({"title": item.get("title"), "snippet": item.get("snippet") or "", "link": item.get("link")})
    return results

def run_web_rag_search(query, provider_preference="serpapi", top_k=5):
    try:
        if provider_preference=="serpapi" and SERPAPI_KEY:
            return serpapi_search(query, num_results=top_k)
        return []
    except Exception as e:
        st.error(f"Web search error: {e}")
        return []

def build_context_from_hits(hits, max_chars_per_hit=800):
    parts=[]
    for i,h in enumerate(hits):
        parts.append(f"Source {i+1} - {h.get('title','result')}\n{(h.get('snippet','')[:max_chars_per_hit])}\nURL: {h.get('link','')}")
    return "\n\n---\n\n".join(parts)

def rag_answer_with_llm(question, context_text, provider="gemini", max_tokens=512, temperature=0.2):
    system_intro="You are an expert predictive maintenance assistant. Use context sources. If answer not present, say so."
    prompt=f"{system_intro}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"
    if provider=="gemini" and GEMINI_AVAILABLE:
        try:
            client = genai.Client(api_key=GOOGLE_API_KEY)
            response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
            return getattr(response, "text", str(response)).strip()
        except Exception as e:
            return f"[Gemini error: {e}]"
    elif OPENAI_AVAILABLE:
        try:
            completion=openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"system","content":system_intro},{"role":"user","content":prompt}],
                max_tokens=max_tokens, temperature=temperature
            )
            return completion["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[OpenAI error: {e}]"
    else:
        return "[No LLM configured]"

def sendgrid_alert(pred_data, llm_analysis):
    # Get sender and recipients from environment variables
    from_email = os.getenv("Email_ID")  # must be verified sender
    to_emails_raw = os.getenv("Email_ID", "")
    to_emails = [e.strip() for e in to_emails_raw.split(",") if e.strip()]

    # Safety check: must have sender and at least one recipient
    if not from_email or not to_emails:
        st.error("SendGrid sender or recipient not configured correctly")
        return  # exit function early

    # Construct the email message
    message = Mail(
        from_email=from_email,
        to_emails=to_emails,
        subject=f"🚨 HIGH RISK ALERT: {pred_data.get('device_id')}",
        html_content=f"""
        <h2>High Failure Risk Detected</h2>
        <p><b>Device:</b> {pred_data.get('device_id')}</p>
        <p><b>Predicted Days Left:</b> {pred_data.get('predicted_days_to_failure')}</p>
        <h3>Summary</h3><p>{llm_analysis.get('summary')}</p>
        <h3>Root Cause</h3><p>{llm_analysis.get('root_cause')}</p>
        <h3>Recommended Actions</h3>
        <ul>{''.join([f"<li>{a}</li>" for a in llm_analysis.get('recommended_actions', [])])}</ul>
        """
    )

    # Send the email
    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)
        st.success(f"Email sent! Status code: {response.status_code}")
    except Exception as e:
        st.error(f"SendGrid failed: {e}")

    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        sg.send(message)
        st.success("📧 SendGrid email sent!")
    except Exception as e:
        st.error("❌ SendGrid failed")
        st.exception(e)

# -----------------------------
# Load model
# -----------------------------
try:
    model = load_model("prediction_model.joblib")
    model_features = getattr(model, "feature_names_in_", None)
except Exception as e:
    st.error("Failed to load model")
    st.exception(e)
    st.stop()

# -----------------------------
# Sidebar & Config
# -----------------------------
st.sidebar.header("Configuration")
threshold_days = st.sidebar.slider("Failure threshold (days)", 1, 100, 30)
DEVICE_IDS = ["DEV001","DEV002","DEV003"]
DEVICE_TYPES = ["Pump","Motor","Compressor"]

# Tabs
tab_dashboard,tab_insights,tab_logs, tab_trends = st.tabs(["📊 Dashboard","🔎 Insights","📋 Logs","📈 Trends"])
if "pred_log" not in st.session_state:
    st.session_state.pred_log=[]
if "rag_log" not in st.session_state:
    st.session_state.rag_log=[]

# -----------------------------
# Dashboard Tab
# -----------------------------
with tab_dashboard:
    st.header("Predictive Maintenance Demo (A2A)")
    cols=st.columns([1,1])
    with cols[0]:
        device_id=st.selectbox("🆔 Device ID",DEVICE_IDS)
        device_type=st.selectbox("⚙️ Device Type",DEVICE_TYPES)
        temperature=st.slider("🌡 Temperature",100.0,200.0,160.0)
        vibration=st.slider("🌀 Vibration",0.0,10.0,2.5)
        pressure=st.slider("💨 Pressure",80.0,120.0,95.0)
        humidity=st.slider("💧Humidity",10,100,40)
        power=st.slider("⚡ Power (kW)",0,200,50)

        if st.button("Predict Failure Risk"):
            base_inputs={
                "temperature":float(temperature),
                "vibration":float(vibration),
                "pressure":float(pressure),
                "humidity":float(humidity),
                "power_consumption":float(power)
            }
            input_df = one_hot_encode_input(base_inputs, device_id, device_type, model_features)
            pred_value = float(model.predict(input_df)[0])
            emoji,color,text = format_prediction_msg(pred_value, threshold_days)
            st.markdown(f"<h3 style='color:{color}'>{emoji} {text}</h3>",unsafe_allow_html=True)
            if PLOTLY_AVAILABLE:
                fig=plotly_gauge(pred_value,color)
                st.plotly_chart(fig)
            else:
                st.progress(min(max(int(pred_value),0),100))

            # Log prediction
            log_entry={
                "timestamp":datetime.utcnow().isoformat(),
                "device_id":device_id,
                "device_type":device_type,
                "temperature":temperature,
                "vibration":vibration,
                "pressure":pressure,
                "humidity":humidity,
                "power_consumption":power,
                "predicted_days_to_failure":pred_value
            }
            st.session_state.pred_log.append(log_entry)

            # A2A Trigger
            HIGH_RISK_THRESHOLD=25
            if pred_value < HIGH_RISK_THRESHOLD:
                rag_question=f"Predictive maintenance advice for device {device_id} of type {device_type}."
                rag_hits=run_web_rag_search(rag_question, top_k=5)
                rag_context=build_context_from_hits(rag_hits)
                llm_provider="gemini" if GEMINI_AVAILABLE else "openai"
                rag_answer=rag_answer_with_llm(rag_question, rag_context, provider=llm_provider)
                rag_summary={
                    "summary":rag_answer,
                    "top_sources":[{"title":h["title"],"snippet":h["snippet"],"link":h["link"]} for h in rag_hits[:3]]
                }
                st.session_state.rag_log.append({
                    "prediction":pred_value,
                    "device_id":device_id,
                    "rag_summary":rag_summary,
                    "timestamp":datetime.utcnow().isoformat()
                })
                # Send email
                if SENDGRID_AVAILABLE:
                    sendgrid_alert(
                        {"device_id":device_id,"predicted_days_to_failure":pred_value},
                        {"summary":rag_summary["summary"],"root_cause":rag_summary["summary"],
                         "recommended_actions":[s["snippet"] for s in rag_summary["top_sources"]]}
                    )
            else:
                st.session_state.rag_log=[]

with cols[1]:
    st.subheader("Quick Summary")
    if st.session_state.pred_log:
        last=st.session_state.pred_log[-1]
        emoji,color,_=format_prediction_msg(last['predicted_days_to_failure'],threshold_days)
        st.markdown(f"<h2 style='color:{color}'>{emoji} {last['predicted_days_to_failure']:.1f} days</h2>",unsafe_allow_html=True)
        st.metric("Predicted Days Left",f"{last['predicted_days_to_failure']:.1f} days")
    else:
        st.info("Make a prediction to see summary.")

# -----------------------------
# Trends Tab
# -----------------------------
with tab_trends:
    st.header("Sensor Trends")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        hist=pd.read_csv(uploaded,parse_dates=["timestamp"])
    else:
        rng=pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        hist=pd.DataFrame({
            "timestamp":rng,
            "temperature":np.random.normal(150,8,len(rng)),
            "vibration":np.random.normal(2.5,0.8,len(rng)),
            "pressure":np.random.normal(95,5,len(rng)),
            "humidity":np.random.normal(40,10,len(rng)),
            "power_consumption":np.random.normal(60,20,len(rng))
        })
    if not hist.empty:
        st.line_chart(hist.set_index("timestamp")[["temperature","vibration","pressure","humidity","power_consumption"]])

# -----------------------------
# Logs Tab
# -----------------------------
with tab_logs:
    st.header("Prediction Logs")
    if st.session_state.pred_log:
        log_df=pd.DataFrame(st.session_state.pred_log)
        st.dataframe(log_df.sort_values("timestamp",ascending=False))
    else:
        st.info("No predictions yet.")

# -----------------------------
# Insights Tab
# -----------------------------
with tab_insights:
    st.header("RAG / Maintenance Insights")

    # ---- RAG AREA ----
    if st.session_state.rag_log:
        for entry in st.session_state.rag_log[::-1]:
            st.markdown(f"**Device:** {entry['device_id']} — Pred: {entry['prediction']:.1f} days")
            st.markdown(f"**RAG Summary:** {entry['rag_summary']['summary']}")
            st.markdown("**Top Sources:**")
            for src in entry['rag_summary']['top_sources']:
                st.markdown(f"- [{src['title']}]({src['link']}): {src['snippet']}")
    else:
        st.info("No RAG/A2A logs yet. High-risk predictions will trigger them automatically.")

    # ---- SHAP ALWAYS VISIBLE ----
    st.subheader("SHAP Feature Contributions")

    if SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE:
        try:
            if not st.session_state.pred_log:
                st.info("Make a prediction to view SHAP.")
            else:
                last = st.session_state.pred_log[-1]
                base_inputs = {
                    k: last[k]
                    for k in ["temperature", "vibration", "pressure", "humidity", "power_consumption"]
                }

                input_df = one_hot_encode_input(
                    base_inputs, last["device_id"], last["device_type"], model_features
                )

                explainer = shap.Explainer(model)
                shap_values = explainer(input_df)

                fig, ax = plt.subplots(figsize=(8, 4))
                plt.rcParams["xtick.labelsize"] = 10
                plt.rcParams["ytick.labelsize"] = 10
                shap.plots.bar(shap_values[0], show=False, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error("Error computing SHAP values.")
            st.exception(e)

    else:
        st.info("SHAP not available. Showing global feature importances if present.")


