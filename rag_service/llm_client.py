import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def stream_llm(prompt: str):
    """
    Stream tokens from Groq LLM
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a UCSM troubleshooting assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        stream=True,
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta:
            content = chunk.choices[0].delta.content
            if content:
                yield content
