#!/usr/bin/env python3
"""
Quick diagnostic script to check LLM availability
"""
import os
import sys

print("=" * 60)
print("LLM DIAGNOSTIC CHECK")
print("=" * 60)

# Check Python environment
print(f"\n1. Python: {sys.executable}")
print(f"   Version: {sys.version.split()[0]}")
print(f"   Working Dir: {os.getcwd()}")

# Check environment variables
print("\n2. Environment Variables:")
google_key = os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

if google_key:
    print(f"   ✅ GOOGLE_API_KEY: {google_key[:20]}...")
else:
    print("   ❌ GOOGLE_API_KEY: NOT SET")

if openai_key:
    print(f"   ✅ OPENAI_API_KEY: {openai_key[:20]}...")
else:
    print("   ❌ OPENAI_API_KEY: NOT SET")

# Check Streamlit secrets
print("\n3. Streamlit Secrets:")
try:
    import streamlit as st
    try:
        google_secret = st.secrets.get("GOOGLE_API_KEY")
        if google_secret:
            print(f"   ✅ GOOGLE_API_KEY in secrets: {google_secret[:20]}...")
        else:
            print("   ❌ GOOGLE_API_KEY: not in secrets.toml")
    except:
        print("   ❌ Cannot read secrets (not in Streamlit context)")
except Exception as e:
    print(f"   ⚠️  Streamlit not available: {e}")

# Check packages
print("\n4. Package Availability:")

# Google Generative AI
try:
    import google.generativeai as genai
    print("   ✅ google-generativeai: INSTALLED")
    genai_available = True
except ImportError as e:
    print(f"   ❌ google-generativeai: NOT INSTALLED")
    print(f"      Run: pip install google-generativeai")
    genai_available = False

# OpenAI
try:
    import openai
    print("   ✅ openai: INSTALLED")
    openai_available = True
except ImportError:
    print(f"   ❌ openai: NOT INSTALLED")
    print(f"      Run: pip install openai")
    openai_available = False

# Test actual API call
print("\n5. API Test:")

if genai_available and google_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=google_key)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        response = model.generate_content("Say 'OK' if you can hear me")
        print(f"   ✅ Gemini API: WORKING")
        print(f"      Response: {response.text[:50]}...")
    except Exception as e:
        print(f"   ❌ Gemini API: FAILED - {str(e)[:100]}")
elif not genai_available:
    print("   ⏭️  Gemini API: SKIPPED (package not installed)")
elif not google_key:
    print("   ⏭️  Gemini API: SKIPPED (API key not set)")

if openai_available and openai_key:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10
        )
        print(f"   ✅ OpenAI API: WORKING")
        print(f"      Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"   ❌ OpenAI API: FAILED - {str(e)[:100]}")
elif not openai_available:
    print("   ⏭️  OpenAI API: SKIPPED (package not installed)")
elif not openai_key:
    print("   ⏭️  OpenAI API: SKIPPED (API key not set)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if (genai_available and google_key) or (openai_available and openai_key):
    print("✅ At least one LLM is configured and ready!")
else:
    print("❌ NO LLM AVAILABLE!")
    print("\nQuick Fix:")
    print("1. Install packages:")
    print("   pip install google-generativeai openai")
    print("\n2. Set API keys:")
    print("   export GOOGLE_API_KEY='your-key'")
    print("   export OPENAI_API_KEY='your-key'")
    print("\n3. Or use secrets.toml:")
    print("   Create .streamlit/secrets.toml with your keys")

print("=" * 60)
