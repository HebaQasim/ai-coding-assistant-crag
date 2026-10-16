"""Part 4 (step A): the code explanation route. The LLM explains the user's code directly, without RAG."""
from langchain_core.prompts import PromptTemplate

from llm import llm

EXPLAIN_PROMPT = PromptTemplate.from_template(
    """You are a programming teacher inside an AI coding assistant.
The user wants to understand some code. Explain it clearly and simply:
- Start with one or two sentences that say what the code does.
- Then explain the important parts step by step.
- If the user asks about a problem (for example "why is this loop incorrect?"), explain the cause and show the corrected code.
- Put any code in Markdown code blocks.
Keep the explanation short and focused on what the user asked.
If the message refers to earlier code ("the second line", "your function"), find that code in the conversation memory.

Conversation memory:
{memory}

User message:
{message}"""
)


def explain_code(message, memory):
    """Send the user's message and the memory straight to the LLM. The answer is streamed piece by piece."""
    prompt = EXPLAIN_PROMPT.format(memory=memory.as_text(), message=message)
    return (chunk.content for chunk in llm.stream(prompt))
