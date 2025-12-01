✏️ **PROBLEM STATMENT**

The Challenge:
Industrial equipment failures cost manufacturers billions annually through unplanned downtime, emergency repairs, and lost productivity.

Traditional maintenance approaches fall into two problematic extremes:

Reactive Maintenance: Waiting for equipment to fail leads to costly emergency repairs, production shutdowns, and safety risks.

Preventive Maintenance: Fixed schedules result in unnecessary maintenance, wasted resources, and still miss unexpected failures

**Real-World Impact:**

Average manufacturing downtime costs: **$260,000 per hour**
**70% of companies have experienced unexpected downtime in the past 3 years**
Maintenance teams struggle to prioritise resources across hundreds of assets
Critical failure patterns go undetected until catastrophic breakdowns occur

🚀 **SOLUTION**

An AI-powered predictive maintenance platform that analyzes equipment sensor data in real-time to predict failures before they happen, transforming maintenance from reactive to proactive.

🎁 **Key Features:**

AI Failure Prediction: Machine learning models detect anomalies and predict failure probability giving advance warning
Smart Prioritisation: Risk-based alerts help teams focus on the most critical issues first
Actionable Insights: Specific recommendations on what needs attention and why
Integration Ready: Connects with existing IoT sensors and maintenance management systems


**ForeSight Predictive Maintenance Agent**
ForeSight is an intelligent, multi-agent Streamlit application designed to help organizations proactively manage equipment health and prevent costly downtime. Leveraging advanced machine learning, real-time analytics, and agentic orchestration, ForeSight predicts potential failures, analyzes root causes, and recommends actionable maintenance strategies.

**Key Features**

Multi-Agent Architecture: Parallel agents collaborate to analyze device data, predict failures, and generate maintenance insights.
Predictive Analytics: Uses historical and real-time sensor data to forecast equipment breakdowns and estimate time-to-failure.
Root Cause Analysis (RAG): Integrates retrieval-augmented generation (RAG) to explain failures and suggest targeted interventions.
Automated Alerts: Sends email notifications to maintenance teams with detailed recommendations and risk assessments.
Interactive UI: Modern Streamlit interface with customizable backgrounds, clear system status indicators, and device input controls.

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

**Problem statement**

**The Challenge:** Industrial equipment failures cost manufacturers billions annually through unplanned downtime, emergency repairs, and lost productivity. Traditional maintenance approaches fall into two problematic extremes:
* Reactive Maintenance: Waiting for equipment to fail leads to costly emergency repairs, production shutdowns, and safety risks
* Preventive Maintenance: Fixed schedules result in unnecessary maintenance, wasted resources, and still miss unexpected failures
  
**Real-World Impact:**
* Average manufacturing downtime costs: **$260,000 per hour**
* 70% of companies have experienced unexpected downtime in the past **3 years**
* Maintenance teams struggle to prioritize resources across hundreds of assets
* Critical failure patterns go undetected until catastrophic breakdowns occur

**Solution:**

An AI-powered predictive maintenance platform that analyzes equipment sensor data in real-time to predict failures before they happen, transforming maintenance from reactive to proactive.

**Key Features:**
* AI Failure Prediction: Machine learning models detect anomalies and predict failure probability giving advance warning
* Smart Prioritization: Risk-based alerts help teams focus on the most critical issues first
* Actionable Insights: Specific recommendations on what needs attention and why
* Integration Ready: Connects with existing IoT sensors and maintenance management systems
  
**How It Works:**
1. Sensors continuously monitor equipment health metrics
2. AI models analyze patterns and detect early warning signs
3. System generates prioritized alerts with failure predictions
4. Maintenance teams receive specific action recommendations

**Quantifiable Benefits:**

**For Operations Teams:**
* **30-50%** reduction in unplanned downtime
* **20-25%** decrease in maintenance costs
* **15-30%** improvement in equipment lifespan
* **70-90%** advance warning before critical failures
  
**For Maintenance Teams:**
* Replace gut-feel decisions with data-driven prioritization
* Shift from firefighting to planned, strategic maintenance
* Optimize spare parts inventory based on predicted needs
* Document and prove maintenance ROI
  
**For the Business:**
* Increased Revenue: More uptime means more production capacity
* Cost Savings: Prevent expensive emergency repairs and rush orders
* Safety Improvement: Reduce workplace accidents from equipment failures
* Competitive Advantage: Reliability becomes a differentiator
  
**ROI Example:** A mid-sized manufacturer with 100 critical assets:
* Current annual downtime cost: **~$2.6M**
* Predictive maintenance investment: **~$150K**
* Expected 40% downtime reduction: **$1M+ annual savings**
* Payback period: **< 2 months**
* Platform learns from outcomes to improve accuracy over time
  
**Strategic Value:** This isn't just about fixing things faster—it's about fundamentally changing how organizations think about asset management. By shifting from reactive to predictive, companies can plan better, optimize resources, and turn maintenance from a cost center into a strategic advantage.

**Why This Matters Now:** With IoT sensors becoming ubiquitous and AI models more accessible, predictive maintenance is no longer a luxury for Fortune 500 companies—it's becoming table stakes for any manufacturer who wants to remain competitive. Our platform makes this technology accessible, affordable, and actionable for organizations of all sizes.

------------------------------------------------------------------------------------------------------------
**Architectural Diagram**


  
<img width="738" height="687" alt="image" src="https://github.com/user-attachments/assets/6fe57b81-41ad-4486-abe8-2648334dfdd2" />


**Features Implemented:**

1. **Multi-agent system, Sequential,Loop agents** [Core Loop]
   
The snippet from MaintenanceAgent shows the Think → Act → Observe loop, which is the core driver of the MCP model.

2. **Custom Tool implementation**
   
Custom Tools - google_cse_search (Google Custom Search Engine - CSE) 
web_search_tool, email_alert_tool, rag_analysis_tool, prediction_tool

3. **Long-running Operations (Pause/Resume Agents)**

The AgentMemory TypedDict defines the minimal required data (thoughts, actions, iteration count) needed to restart and resume the agent at any time.

4. **Sessions & state management**
   
st.session_state and the AgentMemory TypedDict to save the history of runs, thoughts, actions, and observations, fulfilling the role of an InMemorySessionService.

5. **Context engineering**
    
format_history method takes the memory lists (thoughts, actions, observations) and formats them into a compact text block for the LLM's prompt

6. **Observability: Logging, Tracing, Metrics**
    
Python logging, timestamp module to track the agent's internal decisions and actions

7. **Agent deployment**
    
Streamlit cloud communtiy - https://predictivemaintenanceagent-k93b9czgzxvsldqmbgyfnc.streamlit.app


**Steps to run the code**

**Clone the repository:**

git clone https://github.com/Jasna04/Predictive_Maintenance_Agent.git

Open codespace **[ The application should start automatically in the terminal if does not start then follow the steps below ]**

Install Python and pip (if not already installed):

install python3.11 python3-pip

Install required Python packages:

pip3 install -r requirements.txt

**Set up your secrets:**
Create **.streamlit/secrets.toml** in your project root.

![A772EF43-1F29-41E3-8516-0CFE42867B59](https://github.com/user-attachments/assets/52e78178-b620-48e4-b98d-2f20378a028e)

Add your API keys and credentials as shown in the example provided.

SENDGRID_API_KEY = "SENDGRID_API_KEY" --- Email configuration

SERPAPI_API_KEY = "SERPAPI_API_KEY" --- Search API

Email_ID = "Email_ID" --- Email ID for mailing the prediction

OPENAI_API_KEY = "OPENAI_API_KEY" --- Open API Key

GOOGLE_API_KEY = "GOOGLE_API_KEY" --- Google API key

GOOGLE_CSE_ID = "GOOGLE_CSE_ID" --- Google custom search engine key

Run - **streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false**


Open the local URL provided by Streamlit in your browser to use the app.
Note: Never commit your .secrets.toml file or any sensitive keys to GitHub. Use environment variables or secret management for production.

------------------------------------


