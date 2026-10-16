"""Part 5 (step B): the code generation route: rewrite the request -> retrieve from Chroma -> relevance check -> generate code."""
from langchain_core.prompts import PromptTemplate

from llm import fast_llm, llm
from relevance import format_context, is_relevant
from vector_db import get_vector_db

TOP_K = 3

REWRITE_PROMPT = PromptTemplate.from_template(
    """You prepare the search query of an AI coding assistant.
The assistant will search its knowledge base of Python functions with this query.
If the new message is a follow-up that only makes sense with the conversation (for example "now write it in Java"),
rewrite it as one complete request that names the task, for example:
"Write a Java function that checks if a string is a palindrome".
If the new message is already complete on its own, or starts a new topic, return it unchanged.
Reply with the request only.

Conversation memory:
{memory}

New message:
{message}"""
)

NOT_RELEVANT_MESSAGE = (
    "I couldn't find relevant knowledge for your request. "
    "Could you provide the correct solution or reference so I can learn it for future interactions?"
)

GENERATE_PROMPT = PromptTemplate.from_template(
    """You are an expert developer inside an AI coding assistant.
Write code for the user's request, in Python unless the request or the user's preferences ask for another language.
Use the reference code below as a guide: it comes from the assistant's knowledge base and solves a similar problem.
Follow the user's preferences from the conversation memory, and use the recent conversation for follow-up requests
(for example "now make it faster" means: change the code written before).

Your answer must contain:
- The complete source code in one Markdown code block (```python)
- Clear comments
- Best practices: clear names, type hints and a docstring
- Error handling for invalid input
- At the end of the code, a short example that calls the code and prints the result (do not use input())
- After the code block, a short explanation (2-3 sentences) that mentions which reference source helped

Conversation memory:
{memory}

Reference code:
{context}

User request:
{message}"""
)

vectordb = get_vector_db()


def make_search_query(message, memory):
    """Turn a follow-up like "now write it in Java" into a complete request, so the search finds the right topic."""
    if not memory.last_question():   # the first request of a chat has nothing to depend on: no LLM call needed
        return message
    prompt = REWRITE_PROMPT.format(memory=memory.as_text(), message=message)
    return fast_llm.invoke(prompt).content.strip() or message


def generate_code(message, memory):
    """Return the sources found in Chroma, and the answer as a stream of text pieces."""
    query = make_search_query(message, memory)
    documents = vectordb.similarity_search(query, k=TOP_K)
    sources = [document.metadata["source"] for document in documents]

    if not is_relevant(query, documents, memory):
        return sources, [NOT_RELEVANT_MESSAGE]   # a list with one piece, so the caller can treat it like a stream

    prompt = GENERATE_PROMPT.format(memory=memory.as_text(), context=format_context(documents), message=message)
    return sources, (chunk.content for chunk in llm.stream(prompt))
