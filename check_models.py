"""Check each Gemini model on its own, to see which ones can answer right now."""
from llm import MODELS, create_model

for name in MODELS:
    try:
        response = create_model(name).invoke("Reply with the word OK")
        print(f"{name}: OK -> {response.content}")
    except Exception as error:
        print(f"{name}: FAILED -> {str(error)[:250]}")
