# Battlefield-Mission-Planner
AI-powered multi-agent tactical mission planning system using LangGraph, Groq LLM (Llama 3.3-70B), HTN Planning, and MCTS. Built during internship at DRDO-ISSA.

## What it does
Give it a mission brief in plain English — it activates 7 specialized 
AI agents that produce a complete tactical mission plan, including 
task breakdown, specialist reports, an optimized action sequence, 
and a GO/NO-GO verdict.

## Tech Stack
- **LangGraph** — multi-agent pipeline (StateGraph + Send() API)
- **Groq + Llama 3.3-70B** — ultra-fast LLM inference
- **HTN Planning** — hierarchical task decomposition
- **MCTS + UCB1** — optimal action sequence selection
- **Streamlit** — military-themed web dashboard
- **Pydantic** — structured agent outputs
- **Python 3.12**

## How to run
```bash
# 1. Clone the repo
git clone https://github.com/Himanshi44/Battlefield-Mission-Planner

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# 4. Launch
streamlit run app.py
```
