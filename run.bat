@echo off
:: Battlefield Mission Planner — Windows Quick Start
:: Run: run.bat

echo.
echo   BATTLENET AI -- MISSION COMMAND
echo   ================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python 3 is required. Install from https://python.org
    pause
    exit /b 1
)
echo   [1/4] Python found.

:: Create venv
if not exist venv (
    echo   [2/4] Creating virtual environment...
    python -m venv venv
) else (
    echo   [2/4] Virtual environment found.
)

:: Activate
call venv\Scripts\activate.bat

:: Install
echo   [3/4] Installing dependencies...
pip install -q -r requirements.txt

:: Check .env
if not exist .env (
    if exist .env.example (
        copy .env.example .env >nul
        echo.
        echo   ACTION REQUIRED: Set your Groq API key
        echo   Edit .env and replace: GROQ_API_KEY=your_groq_api_key_here
        echo   Get a free key at: https://console.groq.com
        echo.
    )
)

:: Launch
echo   [4/4] Launching Streamlit...
echo.
echo   Open your browser at: http://localhost:8501
echo.
streamlit run app.py --server.port 8501
