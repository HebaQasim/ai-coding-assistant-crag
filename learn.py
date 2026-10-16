"""Part 6: human feedback learning. When retrieval fails and the user sends a solution, it is checked and added to Chroma."""
import ast
from datetime import date

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from code_runner import extract_code
from llm import fast_llm
from prepare_documents import split_documents
from vector_db import get_vector_db

THANKS_MESSAGE = (
    "Thank you! I checked your solution and saved it in my knowledge base, "
    "so I can use it for similar requests in the future."
)

REVIEW_PROMPT = PromptTemplate.from_template(
    """You review a solution that a user gave to an AI coding assistant, before it is saved in its knowledge base.
Does the code correctly solve the request? Small style issues are fine; wrong results or missing parts are not.

Request:
{request}

Solution:
{code}

Reply with one line only:
correct
or
wrong | the reason in one short sentence"""
)

vectordb = get_vector_db()


def find_code(message):
    """Return the Python code in the user's message, or None if the message is not code (a new question)."""
    code = extract_code(message)   # the code is inside a ``` block
    if code is not None:
        return code
    try:
        tree = ast.parse(message)  # or the whole message is code, pasted without a ``` block
    except SyntaxError:
        return None
    # "hello" is valid Python too (a variable name), so real code must define or import something
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            return message
    return None


def check_solution(request, code):
    """Return "" if the solution can be saved, or the reason why it cannot."""
    try:
        ast.parse(code)
    except SyntaxError as error:
        return f"the code has a syntax error on line {error.lineno}: {error.msg}."
    prompt = REVIEW_PROMPT.format(request=request, code=code)
    answer = fast_llm.invoke(prompt).content.strip()
    if answer.lower().startswith("correct"):
        return ""
    _, _, reason = answer.partition("|")
    return reason.strip() or "it does not seem to solve the request."


def learn_solution(request, code):
    """Store the solution in Chroma: create metadata, split it, embed it and insert it."""
    document = Document(
        page_content=f"# Request: {request}\n{code}",
        metadata={
            "source": "user_feedback",
            "request": request,
            "added_on": str(date.today()),
            "checked": "syntax + LLM review",
        },
    )
    chunks = split_documents([document])
    vectordb.add_documents(chunks)   # Chroma creates the embeddings with our embedding model
    return len(chunks)


def handle_solution(request, code):
    """Check the user's solution and save it only if it passes. Return (saved, message for the user)."""
    problem = check_solution(request, code)
    if problem:
        return False, (f"I did not save this solution: {problem}\n\n"
                       "Please send a corrected solution, or press **Skip**.")
    learn_solution(request, code)
    return True, THANKS_MESSAGE


if __name__ == "__main__":
    request = "Write a function that converts a temperature from Celsius to Fahrenheit"
    messages = [
        "Thanks, I will look for it later",                                                  # not code
        "```python\ndef to_fahrenheit(celsius)\n    return celsius * 9 / 5 + 32\n```",         # syntax error
        "```python\ndef to_fahrenheit(celsius):\n    return celsius * 9 / 5\n```",             # wrong (no + 32)
        "```python\ndef to_fahrenheit(celsius):\n    return celsius * 9 / 5 + 32\n```",        # correct
    ]
    print("Chunks before:", len(vectordb.get()["ids"]))
    for message in messages:
        print("\n" + "=" * 60)
        print("User sends:", message.replace("\n", " "))
        code = find_code(message)
        if code is None:
            print("-> Not code: treated as a new question, nothing is saved.")
            continue
        saved, answer = handle_solution(request, code)
        print("-> Saved:", saved)
        print(answer)
    print("\nChunks after:", len(vectordb.get()["ids"]))
