"""
Predictive Maintenance Streamlit App - AGENTIC LOOP ARCHITECTURE
- True Think → Act → Observe → Think loop
- Self-correcting agent that can iterate on decisions
- Dynamic tool selection based on observations
- Memory of previous actions and outcomes
"""

import os
import re
import json
import asyncio
import logging
from io import BytesIO
from datetime import datetime
from typing import TypedDict, Union, List, Dict, Any, Optional
from enum import Enum

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# Optional dependencies
PLOTLY_AVAILABLE = False
SHAP_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import importlib.util
    PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None
    SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None
    MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
except Exception:
    pass

# LLM imports
GENAI_IMPL = None
genai_module = None
try:
    import google.generativeai as genai
    GENAI_IMPL = "google.generativeai"
    genai_module = genai
except Exception:
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

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
    SENDGRID_AVAILABLE = True
except:
    SENDGRID_AVAILABLE = False

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('foresight_agent.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("ForeSight Agentic Loop starting up...")

# Debug: Log configuration status
logger.info(f"GEMINI_AVAILABLE: {GEMINI_AVAILABLE}")
logger.info(f"OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
logger.info(f"GENAI_IMPL: {GENAI_IMPL}")

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="ForeSight Agent - Agentic Loop", 
    layout="wide",
    page_icon="👷",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Custom background image for the app */
    body, .stApp {
        background-image: url('https://img.freepik.com/free-photo/different-wrenches-table_23-2147772262.jpg?t=st=1764246769~exp=1764250369~hmac=cc59f7a74caf9a0d72519a165245a09b69578f168e77609eaed6bc730fb35024&w=2000&h=2000');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center center;
        background-attachment: fixed;
    }
    /* Remove all custom overlays and revert to default Streamlit style */
    section[data-testid="stSidebar"], .main .block-container, [data-testid="stVerticalBlock"], .stExpander, .stAlert, .stInfo {
        background: rgba(255,255,255,0.85) !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        content: none !important;
        display: initial !important;
    }
    /* Fix alignment and overlapping issues for agent trace and summary */
    .agent-thought, .agent-action, .agent-observation {
        width: 100%;
        padding: 10px 12px;
        margin-bottom: 8px;
        background: #f8f9fa;
        border-radius: 6px;
        box-sizing: border-box;
        word-break: break-word;
        font-size: 1rem;
        line-height: 1.5;
        border: 1px solid #e0e0e0;
    }
    .agent-thought strong, .agent-action strong, .agent-observation strong {
        display: block;
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Get secrets
def get_secret(key):
    # Do not cache secrets, always fetch fresh
    try:
        secret_val = st.secrets.get(key)
        logger.info(f"[DEBUG] Fetching {key} from st.secrets: {'SET' if secret_val else 'None'}")
        return secret_val
    except Exception as e:
        logger.warning(f"Could not read {key} from secrets: {e}")
        return None

SERPAPI_KEY = None
GOOGLE_API_KEY = None
OPENAI_API_KEY = None


def refresh_api_keys():
    global SERPAPI_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
    SERPAPI_KEY = get_secret("SERPAPI_API_KEY")
    GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
    OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
    # Always log the actual value for debugging (not in UI)
    logger.info(f"[DEBUG] Refreshed GOOGLE_API_KEY: {GOOGLE_API_KEY}")
    logger.info(f"[DEBUG] Refreshed OPENAI_API_KEY: {OPENAI_API_KEY}")
    logger.info(f"[DEBUG] Refreshed SERPAPI_KEY: {SERPAPI_KEY}")

refresh_api_keys()

if GOOGLE_API_KEY and GEMINI_AVAILABLE:
    try:
        if GENAI_IMPL == "google.generativeai":
            genai_module.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        logger.error(f"Error configuring Gemini: {e}")

if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai.api_key = OPENAI_API_KEY
    except Exception as e:
        logger.warning(f"Could not set OpenAI API key: {e}")

# Constants
TEMP_HIGH_THRESHOLD = 180
VIBRATION_HIGH_THRESHOLD = 5
PRESSURE_LOW_THRESHOLD = 85
PRESSURE_HIGH_THRESHOLD = 110
HUMIDITY_HIGH_THRESHOLD = 70
POWER_HIGH_THRESHOLD = 150

DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.7

# -----------------------------
# AGENTIC LOOP ENUMS & TYPES
# -----------------------------
class AgentState(Enum):
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    ERROR = "error"

class ToolType(Enum):
    PREDICTION = "prediction"
    RAG_ANALYSIS = "rag_analysis"
    EMAIL_ALERT = "email_alert"
    WEB_SEARCH = "web_search"
    SENSOR_CHECK = "sensor_check"

class ThoughtProcess(TypedDict):
    """Agent's internal reasoning"""
    state: str
    reasoning: str
    next_action: Optional[str]
    confidence: float
    timestamp: str

class ActionResult(TypedDict):
    """Result from executing a tool"""
    tool: str
    status: str
    data: Dict[str, Any]
    timestamp: str

class AgentMemory(TypedDict):
    """Agent's working memory"""
    goal: str
    current_state: str
    thoughts: List[ThoughtProcess]
    actions: List[ActionResult]
    observations: List[str]
    iteration: int
    max_iterations: int

# Tool Input/Output Types (from your original code)
class PredictionInput(TypedDict):
    device_id: str
    device_type: str
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    power_consumption: float

class PredictionOutput(TypedDict):
    predicted_days: float
    status: str
    message: str
    timestamp: str

class RAGInput(TypedDict):
    device_type: str
    predicted_days: float
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    power_consumption: float
    search_provider: str

class RAGOutput(TypedDict):
    status: str
    insights: str
    sources: List[str]
    timestamp: str

# -----------------------------
# Core Agent Functions (from your code)
# -----------------------------
@st.cache_resource
def load_model(path="prediction_model.joblib"):
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        st.error(f"❌ Failed to load prediction model: {e}")
        raise

def one_hot_encode_input(base_inputs, device_id, device_type, model_features):
    inp = base_inputs.copy()
    for feat in model_features:
        if feat.startswith("device_id_"):
            val = feat.replace("device_id_", "")
            inp[feat] = 1 if device_id == val else 0
        elif feat.startswith("device_type_"):
            val = feat.replace("device_type_", "")
            inp[feat] = 1 if device_type == val else 0
    df = pd.DataFrame([inp])
    df = df.reindex(columns=model_features, fill_value=0)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def web_search(query, num_results=5):
    # Try Google Custom Search first
    GOOGLE_CSE_ID = get_secret("GOOGLE_CSE_ID")
    GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
    # Debug info removed for page load privacy
    logger.info(f"[DEBUG] GOOGLE_API_KEY: {GOOGLE_API_KEY}")
    logger.info(f"[DEBUG] GOOGLE_CSE_ID: {GOOGLE_CSE_ID}")
    results = google_cse_search(query, num_results, GOOGLE_CSE_ID, GOOGLE_API_KEY)
    return {
        "results": results,
        "provider": "Google CSE"
    }

GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
GOOGLE_CSE_ID = get_secret("GOOGLE_CSE_ID")
GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"

logger = logging.getLogger(__name__)


# --------------------------------------------------------
# 1) GOOGLE CUSTOM SEARCH (REAL WEB DATA)
# --------------------------------------------------------
def google_cse_search(query, num_results=5, GOOGLE_CSE_ID=None, GOOGLE_API_KEY=None):
    # fetch from secrets if not provided
    if GOOGLE_CSE_ID is None:
        GOOGLE_CSE_ID = get_secret("GOOGLE_CSE_ID")
    if GOOGLE_API_KEY is None:
        GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": min(num_results, 10)  # max 10 per request
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logger.info(f"[CSE] Raw response data: {data}")
    except Exception as e:
        logger.error(f"[CSE] Search error: {e}")
        return []

    items = data.get("items", [])
    results = [
        {"title": item.get("title", ""), "snippet": item.get("snippet", ""), "link": item.get("link", "")}
        for item in items
    ]
    
    if not results:
        logger.warning(f"[CSE] No search results found for query: {query}")

    return results


# --------------------------------------------------------
# 2) GEMINI RAG SUMMARIZER + ANALYZER
# --------------------------------------------------------
def gemini_rag_analyze(query, search_results):
    """
    Feed real CSE results to Gemini to reason, summarize, extract insights.
    """

    text_block = " ".join(
        [f"Title: {r['title']}\nURL: {r['link']}\nSnippet: {r['snippet']}\n\n"
         for r in search_results]
    )

    prompt = f"""
You are an intelligent research agent.
Here is the user query: {query}

Here are the REAL web search results from Google Search:

{text_block}

Task:
1. Summarize the most important facts.
2. Extract insights relevant to the query.
3. Remove noise and duplicates.
4. Provide final conclusions.
5. Return actionable recommendations if applicable.

Return clear paragraph form.
    """

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    try:
        r = requests.post(GEMINI_URL, headers=headers, data=json.dumps(payload), timeout=15)
        r.raise_for_status()
    except Exception as e:
        logger.error(f"[Gemini] Error: {e}")
        return {"summary": "", "raw": ""}

    data = r.json()

    try:
        summary = data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        summary = data

    return {
        "summary": summary,
        "raw": data
    }


# --------------------------------------------------------
# 3) HYBRID SEARCH (FULL PIPELINE)
# --------------------------------------------------------
def hybrid_search(query, num_results=5):
    """
    Pipeline:
    - Step 1: Google CSE → Get real web URLs
    - Step 2: Gemini → RAG analyze those URLs
    """

    # Step A: Real search
    cse_results = google_cse_search(query, num_results=num_results)

    # Step B: Gemini RAG
    gemini_output = gemini_rag_analyze(query, cse_results)

    return {
        "provider": "Hybrid (CSE + Gemini)",
        "query": query,
        "results": cse_results,
        "analysis": gemini_output["summary"]
    }
    # Fallback to SerpAPI
    if SERPAPI_KEY:
        try:
            url = "https://serpapi.com/search.json"
            params = {"q": query, "engine": "google", "num": num_results, "api_key": SERPAPI_KEY}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("organic_results", [])[:num_results]:
                results.append({
                    "title": item.get("title"),
                    "snippet": item.get("snippet") or "",
                    "link": item.get("link")
                })
            st.success("[DEBUG] Using SerpAPI for search.")
            logger.info("[DEBUG] Using SerpAPI for search.")
            # Attach provider info
            return {"provider": "SerpAPI", "results": results}
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            st.warning(f"[DEBUG] SerpAPI search failed: {e}")
    logger.warning("No search API configured or all failed")
    return {"provider": "None", "results": []}

def get_llm_status():
    """Check which LLMs are available and configured"""
    status = {
        "gemini": {
            "available": GEMINI_AVAILABLE and GOOGLE_API_KEY is not None,
            "reason": ""
        },
        "openai": {
            "available": OPENAI_AVAILABLE and OPENAI_API_KEY is not None,
            "reason": ""
        }
    }
    
    if not GEMINI_AVAILABLE:
        status["gemini"]["reason"] = "google.generativeai not installed"
    elif not GOOGLE_API_KEY:
        status["gemini"]["reason"] = "GOOGLE_API_KEY not configured"
    
    if not OPENAI_AVAILABLE:
        status["openai"]["reason"] = "openai package not installed"
    elif not OPENAI_API_KEY:
        status["openai"]["reason"] = "OPENAI_API_KEY not configured"
    
    return status

def call_llm(prompt: str, system_instruction: str = None, provider="gemini"):
    """Call LLM for agent reasoning"""
    # Try Gemini first if requested
    if GEMINI_AVAILABLE and GOOGLE_API_KEY and provider == "gemini":
        try:
            if GENAI_IMPL == "google.generativeai":
                model = genai_module.GenerativeModel(
                    GEMINI_MODEL_NAME,
                    system_instruction=system_instruction
                )
                response = model.generate_content(prompt)
                if hasattr(response, "text"):
                    logger.info("✅ LLM call successful (Gemini)")
                    return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
    
    # Try OpenAI as fallback or if requested
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            if hasattr(openai, "OpenAI"):
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                resp = client.chat.completions.create(
                    model=DEFAULT_LLM_MODEL,
                    messages=messages,
                    max_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE
                )
                logger.info("✅ LLM call successful (OpenAI)")
                return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
    
    # No LLM available - provide helpful error
    logger.error("❌ No LLM available - check API keys")
    return "[ERROR: No LLM configured. Please set GOOGLE_API_KEY or OPENAI_API_KEY]"

# -----------------------------
# AGENTIC LOOP TOOLS
# -----------------------------
async def prediction_tool(params: PredictionInput) -> PredictionOutput:
    """ML Prediction Tool"""
    logger.info(f"🔮 TOOL: Prediction - Device: {params['device_id']}")
    try:
        base_inputs = {
            "temperature": params["temperature"],
            "vibration": params["vibration"],
            "pressure": params["pressure"],
            "humidity": params["humidity"],
            "power_consumption": params["power_consumption"]
        }
        
        input_df = one_hot_encode_input(
            base_inputs,
            params["device_id"],
            params["device_type"],
            model_features
        )
        
        pred_value = float(model.predict(input_df)[0])
        logger.info(f"✅ Prediction: {pred_value:.2f} days")
        
        return {
            "predicted_days": pred_value,
            "status": "success",
            "message": f"Predicted {pred_value:.2f} days to failure",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "predicted_days": -1.0,
            "status": "error",
            "message": f"Prediction failed: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }

async def sensor_check_tool(params: Dict[str, float]) -> Dict[str, Any]:
    """Check if sensors are within normal ranges"""
    logger.info("🔍 TOOL: Sensor Check")
    
    anomalies = []
    if params.get("temperature", 0) > TEMP_HIGH_THRESHOLD:
        anomalies.append(f"Temperature critically high: {params['temperature']}°C")
    if params.get("vibration", 0) > VIBRATION_HIGH_THRESHOLD:
        anomalies.append(f"Vibration critically high: {params['vibration']} mm/s")
    if params.get("pressure", 0) < PRESSURE_LOW_THRESHOLD:
        anomalies.append(f"Pressure critically low: {params['pressure']} PSI")
    if params.get("humidity", 0) > HUMIDITY_HIGH_THRESHOLD:
        anomalies.append(f"Humidity critically high: {params['humidity']}%")
    
    return {
        "status": "success",
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "timestamp": datetime.utcnow().isoformat()
    }

async def web_search_tool(params: Union[str, Dict[str, Any]], top_k: int = 3) -> Dict[str, Any]:
    """Web search tool for agent"""
    # Handle both string query and dict params
    if isinstance(params, str):
        query = params
    elif isinstance(params, dict):
        query = params.get("query", "")
        if not query:
            # Build query from context
            device_type = params.get("device_type", "equipment")
            query = f"{device_type} maintenance failure prediction"
    else:
        query = str(params)
    
    logger.info(f"🌐 TOOL: Web Search - Query: {query[:50]}...")
    
    search_data = web_search(query, num_results=top_k)
    results = search_data.get("results", [])
    provider = search_data.get("provider", "Unknown")
    
    if results:
        logger.info(f"✅ Found {len(results)} search results from {provider}")
    else:
        logger.warning(f"⚠️ No search results found from {provider}")
    
    return {
        "status": "success" if results else "no_results",
        "results": results,
        "provider": provider,
        "result_count": len(results),
        "timestamp": datetime.utcnow().isoformat()
    }

async def email_alert_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send email alert for maintenance"""
    logger.info("📧 TOOL: Email Alert")
    
    if not SENDGRID_AVAILABLE:
        logger.error("❌ SendGrid library not installed")
        return {
            "status": "error",
            "message": "SendGrid library not installed - run: pip install sendgrid",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    sendgrid_key = get_secret("SENDGRID_API_KEY")
    email_to = get_secret("Email_ID") or "admin@example.com"
    email_from = get_secret("Email_ID") or "noreply@example.com"
    
    logger.info(f"📧 SendGrid key configured: {bool(sendgrid_key)}")
    if sendgrid_key:
        logger.info(f"📧 SendGrid key format check: starts with SG. = {sendgrid_key.startswith('SG.')}")
        logger.info(f"📧 SendGrid key length: {len(sendgrid_key.strip())}")
    logger.info(f"📧 Email recipient: {email_to}")
    logger.info(f"📧 Email sender: {email_from}")
    
    if not sendgrid_key:
        logger.error("❌ SendGrid API key not configured")
        return {
            "status": "error",
            "message": "SendGrid API key not configured in secrets.toml",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Clean the API key (remove any whitespace/newlines)
    sendgrid_key = sendgrid_key.strip().replace('\n', '').replace('\r', '').replace(' ', '')
    
    # Validate SendGrid key format
    if not sendgrid_key.startswith("SG."):
        logger.error(f"❌ SendGrid API key invalid format. First 10 chars: {sendgrid_key[:10]}")
        return {
            "status": "error",
            "message": f"SendGrid API key has invalid format. Should start with 'SG.' but starts with '{sendgrid_key[:10]}'",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Extract parameters with defaults
    device_info = f"{params.get('device_type', 'Unknown')} {params.get('device_id', 'Unknown')}"
    predicted_days = params.get('predicted_days', 0)
    
    # If predicted_days not in params, check last_action for prediction result
    if predicted_days == 0 and 'last_action' in params:
        last_action = params.get('last_action', {})
        if isinstance(last_action, dict):
            action_data = last_action.get('data', {})
            predicted_days = action_data.get('predicted_days', 0)
    
    subject = f"🚨 Maintenance Alert: {device_info}"
    # Extract RAG details if present
    rag_summary = params.get('rag_summary', '')
    rag_root_cause = params.get('rag_root_cause', '')
    rag_recommendations = params.get('rag_recommendations', '')
    # If full RAG insights string is present, try to parse it
    rag_insights = params.get('rag_insights', '')
    if rag_insights:
        # Try to extract sections from the formatted string
        import re
        summary_match = re.search(r'Summary:\s*(.*)', rag_insights)
        root_cause_match = re.search(r'Root Cause:\s*(.*)', rag_insights)
        actions_match = re.search(r'Actions:\s*(.*)', rag_insights, re.DOTALL)
        if summary_match:
            rag_summary = summary_match.group(1).strip()
        if root_cause_match:
            rag_root_cause = root_cause_match.group(1).strip()
        if actions_match:
            rag_recommendations = actions_match.group(1).strip()

    body_text = f"""
📧 MAINTENANCE ALERT

Device: {device_info}
Predicted Days to Failure: {predicted_days:.1f} days
Temperature: {params.get('temperature', 'N/A')}°C
Vibration: {params.get('vibration', 'N/A')} mm/s
Pressure: {params.get('pressure', 'N/A')} PSI
Humidity: {params.get('humidity', 'N/A')}%
Power: {params.get('power_consumption', 'N/A')} kW

Summary: {rag_summary}
Root Cause: {rag_root_cause}
Recommendations:
{rag_recommendations}

Recommendation: {params.get('recommendation', 'Schedule maintenance inspection')}

Generated by ForeSight Predictive Maintenance Agent
    """
    body_html = f"""
    <h2>Predictive Maintenance Alert</h2>
    <p><strong>Device:</strong> {device_info}</p>
    <p><strong>Predicted Days to Failure:</strong> {predicted_days:.1f} days</p>
    <p><strong>Temperature:</strong> {params.get('temperature', 'N/A')}°C</p>
    <p><strong>Vibration:</strong> {params.get('vibration', 'N/A')} mm/s</p>
    <p><strong>Pressure:</strong> {params.get('pressure', 'N/A')} PSI</p>
    <p><strong>Humidity:</strong> {params.get('humidity', 'N/A')}%</p>
    <p><strong>Power:</strong> {params.get('power_consumption', 'N/A')} kW</p>
    <h3>Summary</h3>
    <p>{rag_summary}</p>
    <h3>Root Cause</h3>
    <p>{rag_root_cause}</p>
    <h3>Recommendations</h3>
    <pre style="font-size:13px;">{rag_recommendations}</pre>
    <h3>General Recommendation</h3>
    <p>{params.get('recommendation', 'Schedule maintenance inspection')}</p>
    <p><em>Generated by ForeSight Predictive Maintenance Agent</em></p>
    """
    
    # Try to send via SendGrid
    if sendgrid_key:
        try:
            message = Mail(
                from_email=email_from,
                to_emails=email_to,
                subject=subject,
                html_content=body_html
            )
            
            sg = SendGridAPIClient(sendgrid_key)
            response = sg.send(message)
            
            logger.info(f"✅ Email sent successfully to {email_to}")
            logger.info(f"✅ SendGrid response code: {response.status_code}")
            return {
                "status": "success",
                "message": f"✅ Alert successfully sent to {email_to}",
                "status_code": response.status_code,
                "email_subject": subject,
                "email_body": body_text,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ SendGrid failed: {error_msg}")
            
            # Provide specific error guidance
            if "403" in error_msg or "Forbidden" in error_msg:
                guidance = "403 Forbidden - Check: 1) API key is valid, 2) Sender email is verified in SendGrid dashboard"
            elif "401" in error_msg or "Unauthorized" in error_msg:
                guidance = "401 Unauthorized - Your SendGrid API key is invalid or expired"
            elif "from_email" in error_msg.lower():
                guidance = f"Sender email '{email_from}' must be verified in SendGrid dashboard (Settings > Sender Authentication)"
            else:
                guidance = f"Error: {error_msg[:200]}"
            
            # Fallback: Return email content for display
            return {
                "status": "simulated",
                "message": f"⚠️ SendGrid Error: {guidance}",
                "email_subject": subject,
                "email_body": body_text,
                "recipient": email_to,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            }
    else:
        # No SendGrid key - simulate email
        logger.warning("No SendGrid key - simulating email")
        return {
            "status": "simulated",
            "message": f"Email prepared for {email_to} (SendGrid not configured)",
            "email_subject": subject,
            "email_body": body_text,
            "recipient": email_to,
            "timestamp": datetime.utcnow().isoformat()
        }

async def rag_analysis_tool(params: RAGInput) -> RAGOutput:
    """RAG Analysis Tool (simplified for agent loop)"""
    logger.info("📚 TOOL: RAG Analysis")
    
    # Extract parameters with defaults
    device_type = params.get('device_type', 'equipment')
    predicted_days = params.get('predicted_days', 0)
    temperature = params.get('temperature', 0)
    vibration = params.get('vibration', 0)
    pressure = params.get('pressure', 0)
    humidity = params.get('humidity', 0)
    
    # If predicted_days not in params, check last_action for prediction result
    if predicted_days == 0 and 'last_action' in params:
        last_action = params.get('last_action', {})
        if isinstance(last_action, dict):
            action_data = last_action.get('data', {})
            predicted_days = action_data.get('predicted_days', 0)
    
    logger.info(f"📚 RAG Analysis params: device={device_type}, days={predicted_days}")
    
    # Build query
    if predicted_days > 0:
        query = f"{device_type} maintenance {predicted_days:.0f} days failure"
    else:
        query = f"{device_type} predictive maintenance best practices"
    
    # Search
    search_result = await web_search_tool(query, top_k=3)

    provider = search_result.get("provider", "Unknown")

    if search_result["status"] == "no_results" or search_result["status"] == "error":
        return {
            "status": "no_sources",
            "insights": "No web sources found. Using internal knowledge for maintenance recommendations.",
            "sources": [],
            "provider": provider,
            "timestamp": datetime.utcnow().isoformat()
        }

    # Build context
    context = "\n\n".join([
        f"Source: {r['title']}\n{r['snippet']}"
        for r in search_result.get("results", [])
    ])

    # LLM analysis
    prompt = f"""Analyze this maintenance situation:

Device Type: {device_type}
Predicted Days to Failure: {predicted_days:.1f}
Temperature: {temperature}°C
Vibration: {vibration} mm/s
Pressure: {pressure} PSI
Humidity: {humidity}%

Context from web:
{context}

Provide:
1. Summary (2-3 sentences)
2. Root cause analysis
3. Top 3 recommended actions

Format as:
Summary: ...
Root Cause: ...
Actions:
1. ...
2. ...
3. ..."""

    analysis = call_llm(
        prompt,
        system_instruction="You are a predictive maintenance expert.",
        provider="gemini" if GEMINI_AVAILABLE else "openai"
    )
    
    sources = [r["link"] for r in search_result.get("results", [])]

    return {
        "status": "success",
        "insights": analysis,
        "sources": sources,
        "provider": provider,
        "timestamp": datetime.utcnow().isoformat()
    }

# -----------------------------
# AGENTIC LOOP CORE
# -----------------------------
class MaintenanceAgent:
    """Autonomous agent with Think → Act → Observe loop"""
    
    def __init__(self, max_iterations: int = 5):
        self.memory: AgentMemory = {
            "goal": "",
            "current_state": AgentState.THINKING.value,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "iteration": 0,
            "max_iterations": max_iterations
        }
        self.tools = {
            ToolType.PREDICTION: prediction_tool,
            ToolType.SENSOR_CHECK: sensor_check_tool,
            ToolType.WEB_SEARCH: web_search_tool,
            ToolType.RAG_ANALYSIS: rag_analysis_tool,
            ToolType.EMAIL_ALERT: email_alert_tool,
        }
    
    async def think(self, context: Dict[str, Any]) -> ThoughtProcess:
        """Agent reasoning step"""
        logger.info(f"💭 THINKING (Iteration {self.memory['iteration']})")
        
        # Build reasoning prompt
        history = self._format_history()
        
        prompt = f"""You are an autonomous predictive maintenance agent. Analyze the situation and decide what to do next.

GOAL: {self.memory['goal']}

CURRENT CONTEXT:
{json.dumps(context, indent=2)}

HISTORY:
{history}

AVAILABLE TOOLS:
- prediction: Get ML prediction for equipment failure
- sensor_check: Check if sensor readings are abnormal
- web_search: Search web for maintenance information
- rag_analysis: Get detailed maintenance analysis with web sources
- email_alert: Send email alert for maintenance (use after getting prediction and analysis)

IMPORTANT DECISION RULES:
1. ALWAYS start with 'prediction' if not done yet
2. If prediction shows HIGH RISK (<25 days) or MODERATE RISK (<50 days):
   - MUST use 'rag_analysis' to get detailed insights
   - MUST use 'email_alert' to notify maintenance team
3. Use 'sensor_check' to identify specific sensor anomalies
4. Only mark 'complete' after all necessary actions are taken

Based on the context and history, what should you do next?

Respond in JSON format:
{{
    "reasoning": "explain your thinking",
    "next_action": "tool_name or 'complete' if done",
    "tool_params": {{}},
    "confidence": 0.0-1.0
}}"""

        response = call_llm(
            prompt,
            system_instruction="You are an autonomous agent. Always respond with valid JSON.",
            provider="gemini" if GEMINI_AVAILABLE else "openai"
        )
        
        # Parse response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
            else:
                decision = {
                    "reasoning": response,
                    "next_action": "complete",
                    "confidence": 0.5
                }
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            decision = {
                "reasoning": response,
                "next_action": "complete",
                "confidence": 0.5
            }
        
        thought = ThoughtProcess(
            state=AgentState.THINKING.value,
            reasoning=decision.get("reasoning", ""),
            next_action=decision.get("next_action"),
            confidence=decision.get("confidence", 0.5),
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.memory["thoughts"].append(thought)
        return thought
    
    async def act(self, action: str, params: Dict[str, Any]) -> ActionResult:
        """Execute tool"""
        logger.info(f"🎬 ACTING: {action}")
        
        self.memory["current_state"] = AgentState.ACTING.value
        
        # Map action to tool
        tool_map = {
            "prediction": ToolType.PREDICTION,
            "sensor_check": ToolType.SENSOR_CHECK,
            "web_search": ToolType.WEB_SEARCH,
            "rag_analysis": ToolType.RAG_ANALYSIS,
            "email_alert": ToolType.EMAIL_ALERT,
        }
        
        if action not in tool_map:
            return ActionResult(
                tool=action,
                status="error",
                data={"error": f"Unknown tool: {action}"},
                timestamp=datetime.utcnow().isoformat()
            )
        
        tool_type = tool_map[action]
        tool_func = self.tools[tool_type]
        
        try:
            result = await tool_func(params)
            action_result = ActionResult(
                tool=action,
                status=result.get("status", "success"),
                data=result,
                timestamp=datetime.utcnow().isoformat()
            )
            self.memory["actions"].append(action_result)
            return action_result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            action_result = ActionResult(
                tool=action,
                status="error",
                data={"error": str(e)},
                timestamp=datetime.utcnow().isoformat()
            )
            self.memory["actions"].append(action_result)
            return action_result
    
    def observe(self, action_result: ActionResult) -> str:
        """Observe and interpret result"""
        logger.info(f"👁️ OBSERVING: {action_result['tool']} → {action_result['status']}")
        
        self.memory["current_state"] = AgentState.OBSERVING.value
        
        # Generate observation
        if action_result["status"] == "success":
            observation = f"Tool '{action_result['tool']}' succeeded. "
            
            if action_result["tool"] == "prediction":
                days = action_result["data"].get("predicted_days", 0)
                observation += f"Predicted {days:.1f} days to failure. "
                if days < 25:
                    observation += "HIGH RISK - immediate action needed."
                elif days < 50:
                    observation += "MODERATE RISK - plan maintenance soon."
                else:
                    observation += "LOW RISK - equipment healthy."
            
            elif action_result["tool"] == "sensor_check":
                count = action_result["data"].get("anomaly_count", 0)
                observation += f"Found {count} sensor anomalies. "
                if count > 0:
                    observation += "Sensor readings require attention."
            
            elif action_result["tool"] == "web_search":
                count = action_result["data"].get("result_count", 0)
                observation += f"Found {count} web results."
            
            elif action_result["tool"] == "rag_analysis":
                observation += "Generated maintenance analysis with recommendations."
            
            elif action_result["tool"] == "email_alert":
                status = action_result['data'].get('status', 'unknown')
                if status == "success":
                    observation += f"Email successfully sent: {action_result['data'].get('message', 'sent')}"
                elif status == "simulated":
                    observation += f"Email prepared (SendGrid unavailable): {action_result['data'].get('message', 'prepared')}"
                else:
                    observation += f"Email alert: {action_result['data'].get('message', 'attempted')}"
        
        else:
            observation = f"Tool '{action_result['tool']}' failed: {action_result['data'].get('error', 'unknown error')}"
        
        self.memory["observations"].append(observation)
        return observation
    
    async def run(self, goal: str, initial_context: Dict[str, Any]):
        """Main agentic loop"""
        logger.info(f"🚀 AGENT STARTING - Goal: {goal}")
        
        self.memory["goal"] = goal
        self.memory["iteration"] = 0
        
        context = initial_context.copy()
        
        while self.memory["iteration"] < self.memory["max_iterations"]:
            self.memory["iteration"] += 1
            
            # THINK
            thought = await self.think(context)
            
            # Check if done
            if thought["next_action"] in ["complete", None]:
                self.memory["current_state"] = AgentState.COMPLETE.value
                logger.info("✅ AGENT COMPLETE")
                break
            
            # ACT
            action_result = await self.act(
                thought["next_action"],
                context
            )
            
            # OBSERVE
            observation = self.observe(action_result)
            
            # Update context with new information
            context["last_action"] = action_result
            context["last_observation"] = observation
            
            # Small delay for readability
            await asyncio.sleep(0.5)
        
        if self.memory["iteration"] >= self.memory["max_iterations"]:
            logger.warning("⚠️ AGENT: Max iterations reached")
            self.memory["current_state"] = AgentState.COMPLETE.value
        
        return self.memory
    
    def _format_history(self) -> str:
        """Format agent history for prompt"""
        lines = []
        
        for i, (thought, action, obs) in enumerate(zip(
            self.memory["thoughts"],
            self.memory["actions"],
            self.memory["observations"]
        ), 1):
            lines.append(f"Iteration {i}:")
            lines.append(f"  Thought: {thought['reasoning'][:100]}...")
            lines.append(f"  Action: {action['tool']} → {action['status']}")
            lines.append(f"  Observation: {obs}")
        
        return "\n".join(lines) if lines else "No history yet"

# -----------------------------
# Load Model
# -----------------------------
try:
    model = load_model("prediction_model.joblib")
    model_features = getattr(model, "feature_names_in_", None)
except Exception as e:
    st.error("Failed to load model")
    st.exception(e)
    st.stop()

# -----------------------------
# Session State
# -----------------------------
if "agent_runs" not in st.session_state:
    st.session_state.agent_runs = []

# -----------------------------
# UI
# -----------------------------
st.title("👷 ForeSight Agent")
st.markdown("**AI-Powered Predictive Maintenance System**")

# Friendly note from Bob the agent
st.info("Hey.. I'm Bob the agent! Enter the equipment inputs to get technical insights, failure patterns, and industry best practices instantly.")

st.markdown("---")

# Sidebar
st.sidebar.header("Agent Configuration")
max_iterations = st.sidebar.slider("Max Iterations", 1, 10, 5)
show_reasoning = st.sidebar.checkbox("Show Detailed Reasoning", value=True)


# LLM Status
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 LLM Status")
llm_status = get_llm_status()


if llm_status["gemini"]["available"]:
    st.sidebar.success("✅ Gemini API configured")
else:
    st.sidebar.error(f"❌ Gemini: {llm_status['gemini']['reason']}")

if llm_status["openai"]["available"]:
    st.sidebar.success("✅ OpenAI API configured")
else:
    st.sidebar.error(f"❌ OpenAI: {llm_status['openai']['reason']}")

if not (llm_status["gemini"]["available"] or llm_status["openai"]["available"]):
    st.sidebar.warning("⚠️ No LLM available! Agent will not work properly.")
    with st.sidebar.expander("📝 Setup Instructions"):
        st.markdown("""
        **To enable LLM functionality:**
        
        1. **For Gemini (Recommended):**
           ```bash
           pip install google-generativeai
           export GOOGLE_API_KEY="your-key-here"
           ```
        
        2. **For OpenAI:**
           ```bash
           pip install openai
           export OPENAI_API_KEY="your-key-here"
           ```
        
        3. **Or add to Streamlit secrets:**
           Create `.streamlit/secrets.toml`:
           ```toml
           GOOGLE_API_KEY = "your-key"
           OPENAI_API_KEY = "your-key"
           ```
        
        Then restart the app.
        """)

# API Configuration Status
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 API Configuration")

# Google CSE configuration indicator (moved to top)
google_cse_status = bool(get_secret("GOOGLE_API_KEY")) and bool(get_secret("GOOGLE_CSE_ID"))
if google_cse_status:
    st.sidebar.success("✅ Google CSE configured")
else:
    st.sidebar.error("❌ Google CSE not configured")

if SERPAPI_KEY:
    st.sidebar.success("✅ SerpAPI configured")
else:
    st.sidebar.error("❌ SerpAPI not configured")

if SENDGRID_AVAILABLE and get_secret("SENDGRID_API_KEY"):
    st.sidebar.success("✅ SendGrid configured")
else:
    st.sidebar.error("❌ SendGrid not configured")

if get_secret("Email_ID"):
    st.sidebar.success("✅ Email configured")
else:
    st.sidebar.warning("⚠️ Email not configured")

# Main interface
st.header("Equipment Input")
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

# Use wider columns and more padding
col1, col2 = st.columns([1.2,1.2], gap="medium")

with col1:
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    device_id = st.selectbox("🆔 Device ID", ["Device001", "Device002", "Device003"])
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    device_type = st.selectbox("⚙️ Device Type", ["Pump", "Compressor", "Motor"])
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
    temperature = st.slider("🌡️ Temperature (°C)", 100.0, 200.0, 160.0)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    vibration = st.slider("🌀 Vibration (mm/s)", 0.0, 10.0, 2.5)

with col2:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    pressure = st.slider("💨 Pressure (PSI)", 80.0, 120.0, 95.0)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    humidity = st.slider("💧 Humidity (%)", 10, 100, 40)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    power = st.slider("⚡ Power (kW)", 0, 200, 50)
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)

if st.button("🚀 Start Agent", type="primary"):
    # Check if LLM is available
    llm_status = get_llm_status()
    if not (llm_status["gemini"]["available"] or llm_status["openai"]["available"]):
        st.error("""
        ❌ **Cannot start agent: No LLM configured!**
        
        The agent requires either Google Gemini or OpenAI API to function.
        Please configure at least one LLM provider in the sidebar.
        """)
        st.stop()

    st.markdown("---")
    st.subheader("🤖 Parallel Agents Execution")

    # Show diagnostic info (always collapsed)
    with st.expander("🔍 System Status", expanded=False):
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("**LLM:**")
            st.write(f"Gemini: {llm_status['gemini']['available']}")
            st.write(f"OpenAI: {llm_status['openai']['available']}")
        with diag_col2:
            st.markdown("**APIs:**")
            # Google CSE status (moved to top)
            google_cse_status = bool(get_secret("GOOGLE_API_KEY")) and bool(get_secret("GOOGLE_CSE_ID"))
            st.write(f"Google CSE: {google_cse_status}")
            st.write(f"SerpAPI: {bool(SERPAPI_KEY)}")
            st.write(f"SendGrid: {bool(SENDGRID_AVAILABLE and get_secret('SENDGRID_API_KEY'))}")
            st.write("Email: Configured")

    # Prepare context
    context = {
        "device_id": device_id,
        "device_type": device_type,
        "temperature": float(temperature),
        "vibration": float(vibration),
        "pressure": float(pressure),
        "humidity": float(humidity),
        "power_consumption": float(power)
    }


    # Define goals for each agent
    prediction_goal = f"Predict failure for {device_type} {device_id}"
    rag_goal = f"Generate RAG analysis for {device_type} {device_id}"
    email_goal = f"Send maintenance alert for {device_type} {device_id}"

    # Create agents
    prediction_agent = MaintenanceAgent(max_iterations=2)
    rag_agent = MaintenanceAgent(max_iterations=2)
    email_agent = MaintenanceAgent(max_iterations=2)

    async def run_parallel_agents():
        # Run prediction agent
        pred_mem = await prediction_agent.run(prediction_goal, context)
        # Use prediction result for RAG agent
        pred_result = None
        for action in pred_mem["actions"]:
            if action["tool"] == "prediction":
                pred_result = action["data"]
        rag_context = context.copy()
        if pred_result:
            rag_context["predicted_days"] = pred_result.get("predicted_days", 0)
        rag_mem = await rag_agent.run(rag_goal, rag_context)
        # Use RAG result for email agent
        rag_result = None
        for action in rag_mem["actions"]:
            if action["tool"] == "rag_analysis":
                rag_result = action["data"]
        email_context = context.copy()
        if pred_result:
            email_context["predicted_days"] = pred_result.get("predicted_days", 0)
        if rag_result:
            email_context["rag_insights"] = rag_result.get("insights", "")
        email_mem = await email_agent.run(email_goal, email_context)
        return pred_mem, rag_mem, email_mem

    with st.spinner("🤖 Agents thinking in parallel..."):
        pred_mem, rag_mem, email_mem = asyncio.run(run_parallel_agents())

    # Extract key results
    prediction_result = None
    rag_result = None
    email_result = None
    for action in pred_mem["actions"]:
        if action["tool"] == "prediction":
            prediction_result = action["data"]
    for action in rag_mem["actions"]:
        if action["tool"] == "rag_analysis":
            rag_result = action["data"]
    for action in email_mem["actions"]:
        if action["tool"] == "email_alert":
            email_result = action["data"]

    st.markdown("### 📋 Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        if prediction_result:
            days = prediction_result.get("predicted_days", 0)
            if days <= 0.0:
                st.warning("Prediction unavailable or invalid. Please check input values and try again.")
            elif days < 25:
                st.error(f"⚠️ **HIGH RISK**\n\n{days:.1f} days to failure")
            elif days < 50:
                st.warning(f"⚡ **MODERATE RISK**\n\n{days:.1f} days to failure")
            else:
                st.success(f"✅ **LOW RISK**\n\n{days:.1f} days to failure")
        else:
            st.info("No prediction available")

    with col2:
        if rag_result and rag_result.get("status") == "success":
            st.success("📚 **RAG Analysis Generated**\n\nInsights available")
        else:
            st.info("📚 **RAG Analysis**\n\nNo analysis generated.")

    with col3:
        if email_result:
            status = email_result.get("status")
            if status == "success":
                st.success("📧 **Email Sent**\n\nAlert delivered successfully")
            elif status == "simulated":
                st.info("📧 **Email Prepared**\n\nSendGrid unavailable")
            else:
                st.warning("📧 **Email Failed**\n\nNot sent")
        else:
            st.info("No email sent")
    # ...existing code...
    
    # RAG Details
    if rag_result and rag_result.get("status") == "success":
        st.markdown("---")
        st.markdown("### 📚 RAG Analysis Details")
        insights = rag_result.get("insights", "")
        sources = rag_result.get("sources", [])
        provider = rag_result.get("provider", "Unknown")
        st.markdown(f"**AI-Generated Insights:**  ")
        st.info(insights)
        st.markdown(f"**Search Provider Used:** `{provider}`")
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"- [{source}]({source})")
        # SHAP feature chart below RAG analysis
        if prediction_result and SHAP_AVAILABLE and MATPLOTLIB_AVAILABLE:
            st.markdown("---")
            st.subheader("🔍 SHAP Feature Contributions")
            base_inputs = {
                "temperature": context["temperature"],
                "vibration": context["vibration"],
                "pressure": context["pressure"],
                "humidity": context["humidity"],
                "power_consumption": context["power_consumption"]
            }
            input_df = one_hot_encode_input(base_inputs, context["device_id"], context["device_type"], model_features)
            import shap
            import matplotlib.pyplot as plt
            explainer = shap.Explainer(model)
            shap_values = explainer(input_df)
            fig, ax = plt.subplots(figsize=(14, 6))  # Maximum visibility
            shap.plots.bar(shap_values[0], show=False, ax=ax)
            plt.tight_layout(pad=0.2)
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.margins(0)
            ax.tick_params(axis='both', labelsize=12)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(12)
            for text in ax.texts:
                text.set_fontsize(10)
            st.pyplot(fig, clear_figure=True)
    # Show execution trace
    if show_reasoning:
        st.markdown("### 📊 Execution Trace")
        st.markdown("#### Prediction Agent")
        for i in range(len(pred_mem["thoughts"])):
            with st.expander(f"Prediction Iteration {i+1}", expanded=(i == len(pred_mem["thoughts"])-1)):
                if i < len(pred_mem["thoughts"]):
                    thought = pred_mem["thoughts"][i]
                    st.markdown(f"""
                    <div class="agent-thought">
                        <strong>💭 THOUGHT:</strong>
                        {thought['reasoning']}<br>
                        <span style='font-size:0.95em; color:#555;'>Next: {thought['next_action']} | Confidence: {thought['confidence']:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(pred_mem["actions"]):
                    action = pred_mem["actions"][i]
                    st.markdown(f"""
                    <div class="agent-action">
                        <strong>🎬 ACTION:</strong> {action['tool']}<br>
                        <strong>Status:</strong> {action['status']}<br>
                        <span style='font-size:0.95em; color:#555;'>{json.dumps(action['data'], indent=2)[:200]}...</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(pred_mem["observations"]):
                    obs = pred_mem["observations"][i]
                    st.markdown(f"""
                    <div class="agent-observation">
                        <strong>👁️ OBSERVATION:</strong>
                        {obs}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("#### RAG Agent")
        for i in range(len(rag_mem["thoughts"])):
            with st.expander(f"RAG Iteration {i+1}", expanded=(i == len(rag_mem["thoughts"])-1)):
                if i < len(rag_mem["thoughts"]):
                    thought = rag_mem["thoughts"][i]
                    st.markdown(f"""
                    <div class="agent-thought">
                        <strong>💭 THOUGHT:</strong>
                        {thought['reasoning']}<br>
                        <span style='font-size:0.95em; color:#555;'>Next: {thought['next_action']} | Confidence: {thought['confidence']:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(rag_mem["actions"]):
                    action = rag_mem["actions"][i]
                    st.markdown(f"""
                    <div class="agent-action">
                        <strong>🎬 ACTION:</strong> {action['tool']}<br>
                        <strong>Status:</strong> {action['status']}<br>
                        <span style='font-size:0.95em; color:#555;'>{json.dumps(action['data'], indent=2)[:200]}...</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(rag_mem["observations"]):
                    obs = rag_mem["observations"][i]
                    st.markdown(f"""
                    <div class="agent-observation">
                        <strong>👁️ OBSERVATION:</strong>
                        {obs}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("#### Email Agent")
        for i in range(len(email_mem["thoughts"])):
            with st.expander(f"Email Iteration {i+1}", expanded=(i == len(email_mem["thoughts"])-1)):
                if i < len(email_mem["thoughts"]):
                    thought = email_mem["thoughts"][i]
                    st.markdown(f"""
                    <div class="agent-thought">
                        <strong>💭 THOUGHT:</strong>
                        {thought['reasoning']}<br>
                        <span style='font-size:0.95em; color:#555;'>Next: {thought['next_action']} | Confidence: {thought['confidence']:.0%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(email_mem["actions"]):
                    action = email_mem["actions"][i]
                    st.markdown(f"""
                    <div class="agent-action">
                        <strong>🎬 ACTION:</strong> {action['tool']}<br>
                        <strong>Status:</strong> {action['status']}<br>
                        <span style='font-size:0.95em; color:#555;'>{json.dumps(action['data'], indent=2)[:200]}...</span>
                    </div>
                    """, unsafe_allow_html=True)
                if i < len(email_mem["observations"]):
                    obs = email_mem["observations"][i]
                    st.markdown(f"""
                    <div class="agent-observation">
                        <strong>👁️ OBSERVATION:</strong>
                        {obs}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Save to history
        st.session_state.agent_runs.append({
            "timestamp": datetime.now().isoformat(),
            "device": f"{device_type} {device_id}",
            "prediction_iterations": pred_mem["iteration"],
            "rag_iterations": rag_mem["iteration"],
            "email_iterations": email_mem["iteration"],
            "prediction_memory": pred_mem,
            "rag_memory": rag_mem,
            "email_memory": email_mem
        })


# History
if st.session_state.agent_runs:
    st.markdown("---")
    st.header("📜 Agent Run History")
    
    for run in reversed(st.session_state.agent_runs[-5:]):
        expander_label = (
            f"{run['device']} - {run['timestamp'][:19]} "
            f"(Prediction: {run.get('prediction_iterations', 'N/A')}, "
            f"RAG: {run.get('rag_iterations', 'N/A')}, "
            f"Email: {run.get('email_iterations', 'N/A')})"
        )
        with st.expander(expander_label):
            st.markdown("**Prediction Agent Memory:**")
            st.json(run.get("prediction_memory", {}))
            st.markdown("**RAG Agent Memory:**")
            st.json(run.get("rag_memory", {}))
            st.markdown("**Email Agent Memory:**")
            import re
            def redact_emails(obj):
                if isinstance(obj, dict):
                    return {k: redact_emails(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [redact_emails(v) for v in obj]
                elif isinstance(obj, str):
                    # Replace email addresses with [REDACTED]
                    return re.sub(r"[\w\.-]+@[\w\.-]+", "[REDACTED]", obj)
                else:
                    return obj
            st.json(redact_emails(run.get("email_memory", {})))

