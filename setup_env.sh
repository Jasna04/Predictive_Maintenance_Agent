#!/bin/bash
# Setup script for environment variables
# Run this script to configure your API keys

echo "Setting up environment variables for Predictive Maintenance Agent..."
echo ""
echo "Please enter your API keys (press Enter to skip):"
echo ""

read -p "Google API Key: " GOOGLE_API_KEY
read -p "OpenAI API Key: " OPENAI_API_KEY
read -p "SendGrid API Key (starts with SG.): " SENDGRID_API_KEY
read -p "SerpAPI Key: " SERPAPI_API_KEY
read -p "Email Address: " Email_ID

# Create .env file
cat > .env << EOF
# API Keys for Predictive Maintenance Agent
export GOOGLE_API_KEY="${GOOGLE_API_KEY}"
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export SENDGRID_API_KEY="${SENDGRID_API_KEY}"
export SERPAPI_API_KEY="${SERPAPI_API_KEY}"
export Email_ID="${Email_ID}"
EOF

echo ""
echo "✅ Environment variables saved to .env file"
echo ""
echo "To use these variables, run:"
echo "  source .env"
echo ""
echo "Then start Streamlit:"
echo "  streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false"
