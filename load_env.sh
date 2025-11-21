#!/bin/bash
# Load environment variables from .env file

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✅ Environment variables loaded from .env"
    echo "GOOGLE_API_KEY set: $([ -n "$GOOGLE_API_KEY" ] && echo "Yes" || echo "No")"
    echo "OPENAI_API_KEY set: $([ -n "$OPENAI_API_KEY" ] && echo "Yes" || echo "No")"
    echo "SENDGRID_API_KEY set: $([ -n "$SENDGRID_API_KEY" ] && echo "Yes" || echo "No")"
else
    echo "❌ .env file not found"
    exit 1
fi
