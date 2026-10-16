"""Part 3 (step B): the intent classifier. The LLM reads the user's message and decides: explain or generate.
Part 8 adds the memory: it helps with follow-up messages and picks up the user's preferences.
It also answers "other" for messages that are not a coding request, so they never reach the two routes."""
from langchain_core.prompts import PromptTemplate

from llm import fast_llm

CLASSIFIER_PROMPT = PromptTemplate.from_template(
    """You are the intent classifier of an AI coding assistant.
Read the user's new message and decide which class it belongs to:

explain: the user wants to understand code that already exists.
Examples: "Explain this code", "What does this function do?", "Why is this loop incorrect?", "Explain line by line".

generate: the user wants new code to be written, or changes to code.
Examples: "Write Python code", "Generate a CNN model", "Build a Flask API", "Create a sorting algorithm".

other: the message is not a request to explain or write code, or it is not clear what the user wants.
Examples: "hello", "thanks", "asdfgh", "what is the weather today?".
A message that only gives a preference, like "I prefer short comments", is also other.

Use the conversation memory to understand follow-up messages. For example, after the assistant wrote some code,
"now make it faster" is generate, and "what does the second line do?" is explain.

If the new message clearly states a new or changed preference for future answers (for example "I prefer...",
"always use...", "from now on..."), such as a programming language, a framework or a coding style,
write after the class, with a | sign, the user's COMPLETE list of preferences for this chat, separated by ";".
Keep the old preferences from the memory that still apply, and replace the ones the user changed.
Example: the memory has "short comments" and the user says "now I prefer detailed comments and type hints":
generate | detailed comments; type hints

Reply with the class only (explain, generate or other), or with the class | preferences.

Conversation memory:
{memory}

New message:
{message}"""
)


def classify(message, memory):
    """Return the class ("explain", "generate" or "other") and the updated preferences ("" if they did not change)."""
    prompt = CLASSIFIER_PROMPT.format(memory=memory.as_text(), message=message)
    answer = fast_llm.invoke(prompt).content.strip()
    intent_part, _, preferences = answer.partition("|")
    intent_part = intent_part.lower()
    if "other" in intent_part:
        return "other", preferences.strip()
    if "explain" in intent_part:
        return "explain", preferences.strip()
    return "generate", preferences.strip()   # any other answer is treated as a request for code
