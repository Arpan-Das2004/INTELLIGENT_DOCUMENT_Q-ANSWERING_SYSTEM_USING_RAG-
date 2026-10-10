"""
One place that talks to the LLM (Groq - free plan, no credit card).

Get a free key at https://console.groq.com  ->  API Keys -> Create API Key

PowerShell setup (run in the terminal you will run the code from):

    $env:GROQ_API_KEY="your-key-here"

Optional: change the model if Groq renames/retires it
(current list: https://console.groq.com/docs/models)

    $env:GROQ_MODEL="llama-3.1-8b-instant"
"""

import os
import time

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# Free plans limit requests per minute, so evaluate.py pauses between calls.
SECONDS_BETWEEN_CALLS = 3


class DailyQuotaError(Exception):
    """Raised when the free daily limit is used up (retrying won't help)."""


def has_api_key() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def missing_key_message() -> str:
    return ('GROQ_API_KEY is not set. In PowerShell run:\n'
            '    $env:GROQ_API_KEY="your-key-here"')


def _ask_groq(prompt: str, system: str | None, max_tokens: int) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=max(max_tokens, 1500),
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def ask_llm(prompt: str, system: str | None = None, max_tokens: int = 500) -> str:
    """Send a prompt to Groq and return the text answer.
    Retries automatically on temporary errors (429 too many requests, 5xx)."""
    last_error = None
    for attempt in range(1, 6):
        try:
            return _ask_groq(prompt, system, max_tokens)
        except Exception as e:  # noqa: BLE001
            last_error = e
            text = str(e)
            if "per day" in text.lower() or "PerDay" in text:
                raise DailyQuotaError(
                    "Daily free quota for this Groq model is used up. Wait until it resets, "
                    'or switch model: $env:GROQ_MODEL="another-model-name" and run again.'
                ) from e
            temporary = any(code in text for code in ("429", "500", "502", "503", "overloaded"))
            if temporary and attempt < 5:
                wait = 15 * attempt
                print(f"   (temporary error from the API, waiting {wait}s then retrying...)")
                time.sleep(wait)
                continue
            raise
    raise last_error
