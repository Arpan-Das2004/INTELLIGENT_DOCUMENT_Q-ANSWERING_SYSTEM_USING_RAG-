"""
Prints the Groq models that YOUR API key can actually use right now.
Pick one name from the list and set it:

    $env:GROQ_MODEL="name-from-the-list"

Run:  python list_models.py
"""

import os
from llm import has_api_key, missing_key_message

SKIP = ("whisper", "tts", "guard", "orpheus", "playai", "embed", "prompt-guard")

if not has_api_key():
    print(missing_key_message())
else:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    ids = sorted(m.id for m in client.models.list().data)
    chat_models = [i for i in ids if not any(s in i.lower() for s in SKIP)]
    print("Chat models available to your key:\n")
    for i in chat_models:
        print("  ", i)
    print(f"\n({len(chat_models)} models) Set one with:")
    print('   $env:GROQ_MODEL="<name from the list>"')
