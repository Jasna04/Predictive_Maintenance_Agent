# Quick Setup Guide for API Keys

## Problem
The `.streamlit/secrets.toml` file is not accessible from the workspace. You need to use environment variables instead.

## Solution

### Option 1: Manual Setup (Recommended)
Create a `.env` file directly:
```bash
cat > .env << 'EOF'
export GOOGLE_API_KEY="your-actual-google-key"
export OPENAI_API_KEY="your-actual-openai-key"
export SENDGRID_API_KEY="SG.your-actual-sendgrid-key"
export SERPAPI_API_KEY="your-actual-serpapi-key"
export Email_ID="your-email@example.com"
EOF
```

Then edit it with your actual keys:
```bash
nano .env
```

### Option 2: Copy from Template
1. Copy the example file:
```bash
cp .env.example .env
```

2. Edit `.env` with your actual API keys:
```bash
nano .env
```

3. Fill in your keys:
```bash
export GOOGLE_API_KEY="your-actual-google-key"
export OPENAI_API_KEY="your-actual-openai-key"
export SENDGRID_API_KEY="SG.your-actual-sendgrid-key"
export SERPAPI_API_KEY="your-actual-serpapi-key"
export Email_ID="your-email@example.com"
```

### Option 3: Interactive Script
Run the setup script:
```bash
bash setup_env.sh
```

### Option 4: Direct Export (Temporary)
Export variables directly in your terminal:
```bash
export GOOGLE_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export SENDGRID_API_KEY="SG.your-key-here"
export SERPAPI_API_KEY="your-key-here"
export Email_ID="your-email@example.com"
```

## Running the App

After setting up environment variables:

```bash
# Load environment variables (if using .env file)
source .env

# Start Streamlit
streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false
```

## Troubleshooting

### Check if keys are loaded:
```bash
echo $GOOGLE_API_KEY
echo $OPENAI_API_KEY
```

### If keys show as "False" in the app:
- Make sure you ran `source .env` before starting Streamlit
- Verify your terminal session has the variables set with `env | grep API`
- Restart Streamlit after loading environment variables

## Getting API Keys

- **Google Gemini**: https://makersuite.google.com/app/apikey
- **OpenAI**: https://platform.openai.com/api-keys
- **SendGrid**: https://app.sendgrid.com/settings/api_keys
- **SerpAPI**: https://serpapi.com/manage-api-key
