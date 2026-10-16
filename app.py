"""Part 9: the Streamlit app. A chat interface for the AI coding assistant."""
import streamlit as st

from code_runner import extract_code, format_run_result, run_with_llm
from generate import NOT_RELEVANT_MESSAGE
from learn import find_code, handle_solution
from memory import ConversationMemory
from router import route

MAX_MESSAGE_LENGTH = 15_000   # characters

ROUTE_LABELS = {
    "explain": "Route: explain code (direct LLM answer, no RAG)",
    "generate": "Route: generate code (Chroma retrieval + relevance check)",
    "learning": "Route: learning (the solution is checked before it is saved in Chroma)",
    "run": "Route: code runner tool (the LLM called run_code)",
    "other": "Route: not a coding request",
}

st.set_page_config(page_title="AI Coding Assistant", page_icon="💻")
st.title("💻 AI Coding Assistant")
st.caption("Explains code, writes code using a knowledge base of solved problems, runs it, "
           "and learns from your solutions.")

# Session state: these values stay the same when Streamlit reruns the script after each click
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()   # opens memory.db, so old conversations come back
    st.session_state.pending_request = ""            # a request that is waiting for a solution from the user
    st.session_state.uploader_key = 0                # changing it empties the file uploader
memory = st.session_state.memory


def show_run_button(number, text):
    """Under an answer that contains Python code: a button that runs the code with the code runner tool.
    The result is saved as a new message, so later questions like "why did it fail?" can use it."""
    code = extract_code(text)
    if code is None:
        return
    if st.button("▶ Run code", key=f"run_{number}"):
        with st.spinner("Running the code..."):
            try:
                result = run_with_llm(code)
            except Exception:
                st.error("The AI model is busy right now. Please try again in a minute.")
                return
        first_line = code.strip().splitlines()[0]
        memory.add_exchange(f"▶ Run the code that starts with `{first_line}`", format_run_result(result), "run")
        st.rerun()


# ---------- Sidebar: conversations, memory visualization and file upload ----------
with st.sidebar:
    st.header("💬 Conversations")
    if st.button("➕ New chat"):
        memory.new_conversation()   # the old chats stay saved
        st.session_state.pending_request = ""
        st.rerun()
    chats = memory.list_conversations()
    chat_ids = [chat["id"] for chat in chats]
    titles = {chat["id"]: chat["title"] for chat in chats}
    chosen = st.radio("Your chats", chat_ids, index=chat_ids.index(memory.conversation_id),
                      format_func=lambda chat_id: titles[chat_id])
    if chosen != memory.conversation_id:   # the user picked another chat: continue that one
        memory.conversation_id = chosen
        st.session_state.pending_request = ""
        st.rerun()
    if st.button("🗑️ Delete this chat"):
        memory.delete_conversation()
        st.session_state.pending_request = ""
        st.rerun()

    st.header("🧠 Memory")
    st.write("**Preferences for this chat:**")
    for preference in memory.preferences:
        st.write("-", preference)
    if not memory.preferences:
        st.write("None yet")
    elif st.button("Forget my preferences"):
        memory.clear_preferences()
        st.rerun()
    st.write(f"**Messages in this chat:** {len(memory.messages)}")
    with st.expander("What the LLM receives"):
        st.text(memory.as_text())

    st.header("📎 File upload")
    uploaded_file = st.file_uploader("Add a code file to your next message", type=["py", "txt"],
                                     key=f"file_{st.session_state.uploader_key}")

# ---------- Conversation history ----------
for number, message in enumerate(memory.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if message["role"] == "assistant":
            if message.get("route") in ROUTE_LABELS:
                st.caption(ROUTE_LABELS[message["route"]])
            if message.get("sources"):
                decision = "not relevant ❌" if message["text"] == NOT_RELEVANT_MESSAGE else "relevant ✅"
                st.caption(f"Retrieved from Chroma: {', '.join(message['sources'])} | Relevance check: {decision}")
            show_run_button(number, message["text"])

# ---------- Waiting for a solution (human feedback learning) ----------
if st.session_state.pending_request:
    st.info(f"I'm waiting for a solution to: **{st.session_state.pending_request}**  \n"
            "Send it as Python code (you can put it inside a ```python block). I will check it before saving it. "
            "If you send a normal question instead, I will just answer it.")
    if st.button("Skip"):
        st.session_state.pending_request = ""
        st.rerun()

# ---------- New message ----------
question = st.chat_input("Ask me to explain or write code...")
if question and uploaded_file is not None:
    file_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    question += f"\n\nFile `{uploaded_file.name}`:\n```python\n{file_text}\n```"
    st.session_state.uploader_key += 1   # the file is used once, then the uploader is emptied

if question and len(question) > MAX_MESSAGE_LENGTH:
    # A very long message would make a huge prompt: stop before sending anything to the LLM
    st.warning(f"Your message is too long ({len(question):,} characters). Please send at most "
               f"{MAX_MESSAGE_LENGTH:,} characters, for example only the part of the code you need help with.")
elif question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            pending = st.session_state.pending_request
            code = find_code(question) if pending else None
            if code is not None:
                # The user sent code while we wait for a solution: check it, and save it only if it passes
                with st.spinner("Checking your solution..."):
                    saved, answer = handle_solution(pending, code)
                if saved:
                    st.session_state.pending_request = ""
                intent, sources = "learning", []
                st.markdown(answer)
            else:
                st.session_state.pending_request = ""   # not a solution: a new question, so we stop waiting
                with st.spinner("Thinking..."):
                    intent, sources, stream = route(question, memory)
                answer = st.write_stream(stream)   # shows the answer piece by piece and returns the full text
                if answer == NOT_RELEVANT_MESSAGE:
                    st.session_state.pending_request = question

            memory.add_exchange(question, answer, intent, sources)
            st.rerun()   # show the new messages from the memory, with their Run buttons
        except Exception as error:
            st.error("The AI model is busy or unavailable right now. Please try again in a minute.")
            st.caption(f"Details: {str(error)[:300]}")
