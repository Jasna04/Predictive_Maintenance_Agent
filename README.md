**ForeSight Predictive Maintenance Agent**
ForeSight is an intelligent, multi-agent Streamlit application designed to help organizations proactively manage equipment health and prevent costly downtime. Leveraging advanced machine learning, real-time analytics, and agentic orchestration, ForeSight predicts potential failures, analyzes root causes, and recommends actionable maintenance strategies.

**Key Features**

Multi-Agent Architecture: Parallel agents collaborate to analyze device data, predict failures, and generate maintenance insights.
Predictive Analytics: Uses historical and real-time sensor data to forecast equipment breakdowns and estimate time-to-failure.
Root Cause Analysis (RAG): Integrates retrieval-augmented generation (RAG) to explain failures and suggest targeted interventions.
Automated Alerts: Sends email notifications to maintenance teams with detailed recommendations and risk assessments.
Interactive UI: Modern Streamlit interface with customizable backgrounds, clear system status indicators, and device input controls.
Privacy & Security: Sensitive information, such as email addresses and API keys, is protected and redacted from UI logs.

**How It Works**

Data Input: Users provide device sensor readings and operational details.
Prediction: The app’s ML model estimates days to failure and highlights risk factors.
Analysis: Agents perform RAG analysis to explain predictions and recommend actions.
Notification: Automated email alerts are sent to relevant stakeholders.
Visualization: SHAP charts and summary blocks help users understand model decisions.

**Technologies Used**

Python 3.11, Streamlit, scikit-learn, pandas, numpy, joblib, matplotlib, SHAP, plotly
SendGrid for email alerts
Google CSE and Gemini/OpenAI for web search and LLM integration

**Use Cases**

Manufacturing equipment monitoring
Industrial IoT predictive maintenance
Facility management and asset health tracking

