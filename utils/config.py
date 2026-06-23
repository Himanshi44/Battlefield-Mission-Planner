"""
Configuration loader for Battlefield Mission Planner.
Reads GROQ_API_KEY from .env file or environment variable.
"""
import os
from pathlib import Path

# Try loading from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on environment variables

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# Groq-supported models
SUPPORTED_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# Default MCTS parameters
DEFAULT_MCTS_ITERATIONS = 40
DEFAULT_MCTS_C = 1.414  # UCB1 exploration constant

# Groq API settings
GROQ_API_BASE = "https://api.groq.com/openai/v1"
MAX_TOKENS = 2048
TEMPERATURE = 0.3
