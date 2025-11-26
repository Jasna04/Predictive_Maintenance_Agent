#!/bin/bash

echo "🔧 ForeSight Agent Startup"
echo "=========================="

# Navigate to parent directory
cd /workspaces/Predictive_Maintenance_Agent

echo "📦 Installing required packages..."
pip install -q google-generativeai openai 2>&1 | grep -v "Requirement already satisfied" || true

echo "🔑 Setting API keys from secrets.toml..."
export GOOGLE_API_KEY="AIzaSyDFUBzyJBjyfotr_tj6y64XAJPPw5uumlY"
export OPENAI_API_KEY="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"
export SERPAPI_API_KEY="9bb735b0c6a29e05efeb8051b860bd273dd2b14ec909edcb6b7e13def5c4ade8"

echo ""
echo "✅ Verifying setup..."
python3 << 'EOF'
import os
import sys

print("Python executable:", sys.executable)
print("Working directory:", os.getcwd())
print()

# Check packages
try:
    import google.generativeai as genai
    print("✅ google-generativeai installed")
except Exception as e:
    print(f"❌ google-generativeai: {e}")

try:
    import openai
    print("✅ openai installed")
except Exception as e:
    print(f"❌ openai: {e}")

# Check API keys
print()
print("🔑 API Keys:")
print(f"  GOOGLE_API_KEY: {'✅ SET' if os.getenv('GOOGLE_API_KEY') else '❌ NOT SET'}")
print(f"  OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
EOF

echo ""
echo "🚀 Starting Streamlit from /workspaces/Predictive_Maintenance_Agent..."
echo ""

streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
