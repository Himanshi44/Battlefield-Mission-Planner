#!/bin/bash
# ─── Battlefield Mission Planner — Quick Start ─────────────────────────────
# Run this script to install dependencies and launch the Streamlit app
# Usage: bash run.sh

set -e

echo ""
echo "  ⬡ BATTLENET AI — MISSION COMMAND  "
echo "  ═════════════════════════════════  "
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: Python 3 is required. Install from https://python.org"
    exit 1
fi

echo "  [1/4] Python found: $(python3 --version)"

# Create virtual environment if not present
if [ ! -d "venv" ]; then
    echo "  [2/4] Creating virtual environment..."
    python3 -m venv venv
else
    echo "  [2/4] Virtual environment found."
fi

# Activate
source venv/bin/activate || source venv/Scripts/activate 2>/dev/null || true

# Install dependencies
echo "  [3/4] Installing dependencies..."
pip install -q -r requirements.txt

# Check for .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo ""
        echo "  ┌─────────────────────────────────────────────┐"
        echo "  │  ACTION REQUIRED: Set your Groq API key     │"
        echo "  │  Edit .env and replace:                     │"
        echo "  │    GROQ_API_KEY=your_groq_api_key_here      │"
        echo "  │  Get a free key at: console.groq.com        │"
        echo "  └─────────────────────────────────────────────┘"
        echo ""
        echo "  You can also enter the key in the Streamlit sidebar."
    fi
fi

# Launch
echo "  [4/4] Launching Streamlit..."
echo ""
echo "  Open your browser at: http://localhost:8501"
echo ""
streamlit run app.py --server.port 8501 --server.headless false
