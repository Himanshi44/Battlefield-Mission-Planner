"""
Groq LLM client wrapper.
Provides a simple synchronous interface used by all agents.
"""
import json
import time
from typing import Optional
import requests

from utils.config import GROQ_API_BASE, MAX_TOKENS, TEMPERATURE


class GroqClient:
    """
    Thin wrapper around the Groq OpenAI-compatible REST API.
    Used by all agents for LLM inference.
    """

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is required. Set it in .env or the sidebar.")
        self.api_key = api_key
        self.model = model
        self.base_url = GROQ_API_BASE

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = MAX_TOKENS,
        temperature: float = TEMPERATURE,
        retries: int = 3,
    ) -> str:
        """Send a chat completion request to Groq and return the response text."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                elif resp.status_code == 429:
                    # Rate limit — back off
                    wait = 2 ** (attempt + 1)
                    time.sleep(wait)
                    last_error = f"Rate limited (429). Retry {attempt+1}/{retries}"
                else:
                    last_error = f"API error {resp.status_code}: {resp.text[:300]}"
                    break
            except requests.exceptions.Timeout:
                last_error = f"Request timeout (attempt {attempt+1})"
                time.sleep(2)
            except Exception as e:
                last_error = str(e)
                break

        raise RuntimeError(f"Groq API call failed after {retries} attempts: {last_error}")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Request JSON-mode response from Groq.
        Strips markdown fences and parses JSON.
        """
        json_system = system_prompt + "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no preamble."
        raw = self.chat(json_system, user_prompt, max_tokens=max_tokens, temperature=0.1)
        # Strip markdown fences
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract first JSON object
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from model response:\n{raw[:500]}")
