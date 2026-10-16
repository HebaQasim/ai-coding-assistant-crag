"""Part 7: the code runner tool.
Step A: run_code runs Python code in a separate process and returns what the code printed.
Step B: the LLM invokes the tool. The LLM cannot run code itself: it answers with a "tool call"
        (the name of the tool + the code to run), and our code executes that call.

What happens when the user clicks "Run code" in the app:
    app.py -> run_with_llm(code) -> the LLM answers with a tool call -> run_code runs the code
           -> format_run_result turns the result into a chat message -> the message is saved in the memory
"""
import re
import subprocess
import sys
import tempfile

from langchain_core.tools import tool

from llm import create_model

TIME_LIMIT = 10        # seconds: a longer run is stopped (for example an infinite loop)
MAX_OUTPUT = 10_000    # characters of output shown to the user


def extract_code(answer):
    """Take the code out of the first Python block in a text: ```python, ```py or a plain ``` (None if there is none)."""
    # Examples:
    #   an LLM answer with a ```python block around print(1)  ->  "print(1)\n"
    #   a run result with ```text blocks only                 ->  None (it is output, not code to run)
    #   "No code here"                                        ->  None
    # findall gives one (language, code) pair for each block, for example [("text", "5\n"), ("python", "print(5)\n")].
    # Each match is one whole block (its opening and closing marks), so the end of a block is never read as a start.
    for language, code in re.findall(r"```(\w*)\n(.*?)```", answer, re.DOTALL):
        if language.lower() in ("", "python", "py"):   # we can only run Python
            return code
    return None


def shorten(text):
    """Keep very long outputs readable: show only the first MAX_OUTPUT characters."""
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... (output cut: only the first {MAX_OUTPUT:,} characters are shown)"
    return text


# ---------- Step A: the tool ----------
# @tool turns the function into a LangChain tool. The LLM learns about the tool from three things:
#   name         ->  "run_code"      (the function name)
#   description  ->  the docstring
#   input        ->  code: str       (the type hint)
# A tool is called with a dictionary of its inputs: run_code.invoke({"code": "print(2 + 3)"})
@tool
def run_code(code: str) -> dict:
    """Run Python code in a separate process and return its standard output and standard error."""
    # Examples:
    #   {"code": "print(2 + 3)"}        ->  {"output": "5\n", "error": ""}
    #   {"code": "print(10 / 0)"}       ->  {"output": "", "error": "Traceback ... ZeroDivisionError: division by zero"}
    #   {"code": "while True: pass"}    ->  {"output": "", "error": "Stopped: the code ran for more than 10 seconds."}
    with tempfile.TemporaryDirectory() as folder:   # an empty folder that is deleted after the run
        try:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", code],   # the same Python as the app; -c runs the text as code
                cwd=folder,                     # files made by the code go into the temporary folder
                stdin=subprocess.DEVNULL,       # nobody can type: input() fails at once instead of waiting
                capture_output=True,            # keep what the code prints in result.stdout and result.stderr
                text=True,
                encoding="utf-8",
                timeout=TIME_LIMIT,
            )
        except subprocess.TimeoutExpired:
            return {"output": "", "error": f"Stopped: the code ran for more than {TIME_LIMIT} seconds."}
    # output: what print() wrote. error: the traceback if the code failed. Both can be filled at the same time.
    return {"output": shorten(result.stdout), "error": shorten(result.stderr)}


# ---------- Step B: the LLM invokes the tool ----------
# bind_tools gives the model the description of run_code, so it can answer with a request to call it.
# Each model gets the tool before the fallbacks are linked: if the first model is busy, the next one knows the tool too.
runner_models = [
    create_model(name).bind_tools([run_code])
    for name in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
]
runner_llm = runner_models[0].with_fallbacks(runner_models[1:])


def run_with_llm(code):
    """Ask the LLM to run the code: it answers with a call to the run_code tool, and we execute that call."""
    prompt = (
        "Run the following Python code with the run_code tool. "
        "Pass the code exactly as it is, without the ``` marks and without changing anything.\n\n"
        f"```python\n{code}\n```"
    )
    message = runner_llm.invoke(prompt)
    # The answer is not normal text (message.content is usually empty). It is a list of tool calls, for example:
    #   [{"name": "run_code", "args": {"code": "print(2 + 3)"}, "id": "...", "type": "tool_call"}]
    for tool_call in message.tool_calls:
        if tool_call["name"] == "run_code":
            return run_code.invoke(tool_call["args"])   # here the code really runs
    # The model answered with text instead of a tool call: return the same shape, with a clear message
    return {"output": "", "error": "The assistant did not call the code runner tool. Please try again."}


def format_run_result(result):
    """The run result as a chat message, so it is shown in the chat and saved in the memory."""
    # Example: {"output": "5\n", "error": ""} becomes the message
    #   **Output:**
    #   (a ```text block with 5 inside)
    # The blocks are marked as text, not python, so extract_code does not show a Run button under a result.
    parts = []
    if result["output"]:
        parts.append(f"**Output:**\n```text\n{result['output'].rstrip()}\n```")
    if result["error"]:
        parts.append(f"**Error:**\n```text\n{result['error'].rstrip()}\n```")
    return "\n\n".join(parts) or "The code ran without printing anything."


if __name__ == "__main__":
    tests = {
        "Normal code": "numbers = [3, 1, 2]\nprint('Sorted:', sorted(numbers))",
        "Code with an error": "print('Before the error')\nprint(10 / 0)",
        "Infinite loop": "while True:\n    pass",
        "Code that asks for input": "name = input('Your name: ')\nprint('Hello', name)",
        "Very long output": "for i in range(100_000):\n    print(i)",
    }
    for name, code in tests.items():
        result = run_code.invoke({"code": code})
        print(f"===== {name}")
        print("output:", result["output"][-200:])   # the end of the output is enough here
        print("error:", result["error"])

    answer = "Here is the code:\n```python\nprint('Hello from the extracted code')\n```\nIt prints a message."
    print("===== Code taken from an LLM answer")
    print(run_code.invoke({"code": extract_code(answer)}))

    print("===== The LLM invokes the tool (step B)")
    print(run_with_llm("scores = {'Heba': 98, 'Ali': 91}\nfor name, score in scores.items():\n    print(name, score)"))
