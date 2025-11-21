"""
Predictive Maintenance Streamlit App (A2A + Web-RAG + Multi-Agent)
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
# Try multiple Google GenAI client import patterns for compatibility
GENAI_IMPL = None
genai_module = None
try:
    from google import genai
    GENAI_IMPL = "genai"
    genai_module = genai

except Exception:
        GENAI_IMPL = None

GEMINI_AVAILABLE = GENAI_IMPL is not None

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
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
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or st.secrets.get("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
print(f"OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")
if GOOGLE_API_KEY:
    try:
        GEMINI_AVAILABLE = GEMINI_AVAILABLE and bool(GOOGLE_API_KEY)
        
    except Exception:
        pass
openai.api_key = OPENAI_API_KEY

# Debug: show which genai implementation and keys are present (after keys loaded)
print(f"GENAI_IMPL={GENAI_IMPL}, GEMINI_AVAILABLE={GEMINI_AVAILABLE}")
print(f"GOOGLE_API_KEY present (env/st.secrets): {bool(GOOGLE_API_KEY)}")
print(f"OPENAI_AVAILABLE={OPENAI_AVAILABLE}, OPENAI_API_KEY present: {bool(OPENAI_API_KEY)}")

# Show provider in sidebar for quick visibility
try:
    prov_text = "Gemini available" if GEMINI_AVAILABLE and GOOGLE_API_KEY else "Gemini unavailable"
    prov_text += " | OpenAI available" if OPENAI_AVAILABLE and OPENAI_API_KEY else " | OpenAI unavailable"
    st.sidebar.markdown(f"**LLM Status:** {prov_text}")
except Exception:
    pass

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
    """Return a Plotly gauge figure for predicted days to failure."""
    try:
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
    except Exception:
        return None

def serpapi_search(query, num_results=5):
    if not SERPAPI_KEY:
        return []
    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "engine": "google", "num": num_results, "api_key": SERPAPI_KEY}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", [])[:num_results]:
            results.append({"title": item.get("title"), "snippet": item.get("snippet") or "", "link": item.get("link")})
        return results
    except Exception:
        return []

def run_web_rag_search(query, provider_preference="serpapi", top_k=5):
    try:
        if provider_preference == "serpapi" and SERPAPI_KEY:
            return serpapi_search(query, num_results=top_k)
        return []
    except Exception as e:
        try:
            st.error(f"Web search error: {e}")
        except Exception:
            pass
        return []

def build_context_from_hits(hits, max_chars_per_hit=800):
    parts = []
    for i, h in enumerate(hits):
        parts.append(f"Source {i+1} - {h.get('title','result')}\n{(h.get('snippet','')[:max_chars_per_hit])}\nURL: {h.get('link','')}")
    return "\n\n---\n\n".join(parts)


def safe_run_web_rag_search(query, provider_preference="serpapi", top_k=5):
    """Call `run_web_rag_search` if defined, otherwise return an empty list.
    This prevents NameError when the function is missing in some load order scenarios.
    """
    func = globals().get("run_web_rag_search")
    if callable(func):
        try:
            return func(query, provider_preference=provider_preference, top_k=top_k)
        except Exception:
            return []
    return []

def rag_answer_with_llm(question, context_text, provider="gemini", max_tokens=512, temperature=0.2):
    system_intro = "You are an expert predictive maintenance assistant. Use context sources. If answer not present, say so."
    prompt = f"{system_intro}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{question}"

    # Try Gemini/Google GenAI first when available
    if GEMINI_AVAILABLE and genai_module is not None:
        try:
            if GENAI_IMPL == "genai":
                client = genai_module.Client(api_key=GOOGLE_API_KEY)
                response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
            else:
                try:
                    genai_module.configure(api_key=GOOGLE_API_KEY)
                except Exception:
                    pass
                response = genai_module.generate_text(model="text-bison-001", input=prompt)

            # handle common response shapes
            if hasattr(response, "text") and response.text:
                try:
                    st.session_state['llm_provider_used'] = 'gemini'
                except Exception:
                    pass
                return response.text.strip()

            if hasattr(response, "candidates") and response.candidates:
                cand = response.candidates[0]
                if hasattr(cand, "content"):
                    try:
                        st.session_state['llm_provider_used'] = 'gemini'
                    except Exception:
                        pass
                    return cand.content.strip()
                if hasattr(cand, "text"):
                    try:
                        st.session_state['llm_provider_used'] = 'gemini'
                    except Exception:
                        pass
                    return cand.text.strip()

            if isinstance(response, dict):
                cands = response.get("candidates") or response.get("choices")
                if cands and len(cands) > 0:
                    first = cands[0]
                    try:
                        st.session_state['llm_provider_used'] = 'gemini'
                    except Exception:
                        pass
                    return (first.get("content") or first.get("text") or "").strip()

            try:
                st.session_state['llm_provider_used'] = 'gemini'
            except Exception:
                pass
            return str(response).strip()
        except Exception as e:
            import traceback
            print(f"Gemini attempt failed: {e}\n" + traceback.format_exc())
            try:
                st.session_state['llm_provider_used'] = 'gemini_error'
            except Exception:
                pass

    # Fallback to OpenAI
    if OPENAI_AVAILABLE:
        try:
            messages = [{"role": "system", "content": system_intro}, {"role": "user", "content": prompt}]
            ver = getattr(openai, "__version__", None)
            version_ge_1 = False
            if ver:
                try:
                    version_ge_1 = int(str(ver).split(".")[0]) >= 1
                except Exception:
                    version_ge_1 = False

            if hasattr(openai, "OpenAI"):
                try:
                    if OPENAI_API_KEY:
                        client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    else:
                        client = openai.OpenAI()
                    resp = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages,
                                                          max_tokens=max_tokens, temperature=temperature)
                    try:
                        st.session_state['llm_provider_used'] = 'openai'
                    except Exception:
                        pass
                    try:
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        return resp["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    print(f"OpenAI new-client attempt failed: {e}")
                    if not version_ge_1 and hasattr(openai, "ChatCompletion"):
                        try:
                            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,
                                                                      max_tokens=max_tokens, temperature=temperature)
                            try:
                                st.session_state['llm_provider_used'] = 'openai'
                            except Exception:
                                pass
                            return completion["choices"][0]["message"]["content"].strip()
                        except Exception as e2:
                            return f"[OpenAI error (legacy fallback failed): {e2}]"
                    return f"[OpenAI error (new client failed): {e}]"

            if not version_ge_1 and hasattr(openai, "ChatCompletion"):
                try:
                    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,
                                                              max_tokens=max_tokens, temperature=temperature)
                    try:
                        st.session_state['llm_provider_used'] = 'openai'
                    except Exception:
                        pass
                    return completion["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    return f"[OpenAI error (legacy): {e}]"

            try:
                st.session_state['llm_provider_used'] = 'openai_error'
            except Exception:
                pass
            return "[OpenAI error: No compatible OpenAI chat API found for installed openai package]"
        except Exception as e:
            try:
                st.session_state['llm_provider_used'] = 'openai_error'
            except Exception:
                pass
            return f"[OpenAI error: {e}]"

    try:
        st.session_state['llm_provider_used'] = 'none'
    except Exception:
        pass
    return "[No LLM configured]"


def parse_llm_analysis(text):
    """Try to extract structured fields from a raw LLM text reply.
    Returns dict with keys: summary, root_cause, recommended_actions (list), evidence (list), confidence (0-1).
    Uses heuristics to prefer causal phrases for root cause and selects evidence sentences that mention sensors or failure modes.
    """
    if not text:
        return {"summary": "", "root_cause": "", "recommended_actions": [], "evidence": [], "confidence": 0.0}

    s = text.strip()

    # Normalize and strip empty lines
    lines = [l.strip() for l in re.split(r"\r?\n", s) if l.strip()]
    full = "\n".join(lines)

    # Initialize
    root = None
    actions = []

    # 1) Extract explicit 'Root' and 'Recommended' sections if present
    m_root = re.search(r"(?i)(root\s*cause[:\-]?\s*)(.+?)(?=(recommended|recommended actions|actions|$))", full, re.S)
    if m_root:
        root = m_root.group(2).strip()

    m_actions = re.search(r"(?i)(recommended actions|recommendations|actions|next steps)[:\-]?\s*(.+)$", full, re.S)
    if m_actions:
        tail = m_actions.group(2).strip()
        items = re.split(r"\n|\r|\*|-|\u2022|\d+\.\s+", tail)
        actions = [it.strip() for it in items if it and len(it.strip()) > 3]

    # 2) If no explicit root, look for causal connectors or causal sentences
    if not root:
        m_cause = re.search(r"(?i)(because|due to|caused by|result of|as a result of)\s+([^\.\n]+)", full)
        if m_cause:
            root = m_cause.group(2).strip()
        else:
            # select sentence with failure-related keywords
            sents = re.split(r"(?<=[.!?])\s+", s)
            chosen = None
            for sent in sents:
                if re.search(r"(?i)overheat|overheating|wear|corrod|leak|short circuit|vibration|imbalance|misalign|bearing|lubricat|temperature|pressure|humidity|power|current|voltage|fault|fail", sent):
                    chosen = sent.strip()
                    break
            if not chosen and len(sents) > 1:
                chosen = sents[1].strip()
            if not chosen:
                chosen = sents[0].strip()
            root = chosen

    # 3) Evidence extraction: sentences that mention sensors, readings, or failure modes
    evidence = []
    evid_candidates = []
    all_sents = re.split(r"(?<=[.!?])\s+", s)
    for sent in all_sents:
        if re.search(r"(?i)temperature|vibration|pressure|humidity|power|current|voltage|bearing|lubricat|sensor|reading|rpm|Hz|°C|deg|leak|smoke|sparks|anomal", sent):
            evid_candidates.append(sent.strip())
    # Prefer sentences that contain numbers or units as stronger evidence
    def score_evidence(sent):
        score = 0
        if re.search(r"\d+", sent):
            score += 2
        if re.search(r"(?i)°C|deg|rpm|Hz|kW|V|A|%", sent):
            score += 2
        if re.search(r"(?i)temperature|vibration|pressure|humidity|bearing|lubricat", sent):
            score += 1
        return score

    evid_candidates_sorted = sorted(evid_candidates, key=score_evidence, reverse=True)
    for e in evid_candidates_sorted[:3]:
        evidence.append(e)
    # If none found, take up to two sentences around the root cause
    if not evidence:
        if root and root in s:
            # find sentences near root
            idx = None
            for i, sent in enumerate(all_sents):
                if root.strip() == sent.strip():
                    idx = i
                    break
            if idx is not None:
                if idx > 0:
                    evidence.append(all_sents[idx-1].strip())
                evidence.append(all_sents[idx].strip())
        else:
            for sent in all_sents[:2]:
                if sent.strip():
                    evidence.append(sent.strip())

    # 4) If no actions found, heuristically pick imperative lines or last sentences
    if not actions:
        cand_actions = []
        for ln in lines:
            if re.match(r"^(fix|check|replace|inspect|monitor|schedule|update|apply|restart|test|tighten|lubricate|calibrate)\b", ln, re.I):
                cand_actions.append(ln)
        if cand_actions:
            actions = cand_actions
        else:
            sents_tail = [sent.strip() for sent in all_sents if sent.strip()]
            actions = sents_tail[-2:] if len(sents_tail) >= 2 else sents_tail

    # 5) Build concise summary (one sentence) preferring explicit Summary section
    m_sum = re.search(r"(?i)(summary[:\-]?\s*)(.+?)(?=(root|root cause|recommended|$))", full, re.S)
    if m_sum:
        summary_raw = m_sum.group(2).strip()
    else:
        summary_raw = s
    sents_for_summary = re.split(r"(?<=[.!?])\s+", summary_raw)
    summary = sents_for_summary[0].strip() if sents_for_summary and sents_for_summary[0].strip() else (summary_raw[:500] + "...")

    # 6) Simple confidence heuristic (0-1)
    conf = 0.3
    # more signals increase confidence
    if m_root:
        conf += 0.35
    # presence of numeric evidence increases confidence
    numeric_evidence_count = sum(1 for e in evidence if re.search(r"\d+", e))
    conf += min(0.3, 0.1 * numeric_evidence_count)
    # higher if explicit recommended actions present
    if m_actions:
        conf += 0.05
    conf = max(0.0, min(0.95, conf))

    # Ensure recommended actions is a list of strings
    recommended = actions if isinstance(actions, list) else [actions]

    return {
        "summary": summary,
        "root_cause": root,
        "recommended_actions": recommended,
        "evidence": evidence,
        "confidence": round(conf, 2)
    }

def sendgrid_alert(pred_data, llm_analysis):
    # Get sender and recipients from environment variables
    from_email = st.secrets.get("Email_ID")  # must be verified sender
    to_emails_raw = st.secrets.get("Email_ID", "")
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
        sg = SendGridAPIClient(st.secrets.get("SENDGRID_API_KEY"))
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
                st.info(f"LLM provider used: {llm_provider}")
                # Parse the raw LLM text into structured fields
                llm_analysis = parse_llm_analysis(rag_answer)
                rag_summary={
                    "summary": llm_analysis.get("summary"),
                    "root_cause": llm_analysis.get("root_cause"),
                    "recommended_actions": llm_analysis.get("recommended_actions", []),
                    "top_sources":[{"title":h["title"],"snippet":h["snippet"],"link":h["link"]} for h in rag_hits[:3]]
                }
                # Replace previous insights with the new high-risk alert so the UI shows only the latest
                st.session_state.rag_log = [{
                    "prediction": pred_value,
                    "device_id": device_id,
                    "rag_summary": rag_summary,
                    "timestamp": datetime.utcnow().isoformat()
                }]
                # Send email
                if SENDGRID_AVAILABLE:
                    sendgrid_alert(
                        {"device_id":device_id,"predicted_days_to_failure":pred_value},
                        {"summary": rag_summary.get("summary"),
                         "root_cause": rag_summary.get("root_cause"),
                         "recommended_actions": rag_summary.get("recommended_actions", [])}
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
            # Show parsed Root Cause and Recommended Actions when available
            root = entry['rag_summary'].get('root_cause') or entry['rag_summary'].get('rag_summary', {}).get('root_cause')
            if root:
                st.markdown(f"**Root Cause:** {root}")
            recs = entry['rag_summary'].get('recommended_actions') or entry['rag_summary'].get('rag_summary', {}).get('recommended_actions')
            if recs:
                st.markdown("**Recommended Actions:**")
                for a in recs:
                    st.markdown(f"- {a}")
            # Raw LLM summary removed for cleaner UI
            st.markdown("**Top Sources:**")
            for src in entry['rag_summary']['top_sources']:
                # show only a simple bullet point: snippet (link)
                snippet = (src.get('snippet') or src.get('title') or "").strip()
                link = src.get('link', '').strip()
                if snippet and link:
                    st.markdown(f"- {snippet} — [{link}]({link})")
                elif snippet:
                    st.markdown(f"- {snippet}")
                elif link:
                    st.markdown(f"- [{link}]({link})")
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


