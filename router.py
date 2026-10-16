"""Part 4 (step B): the router. The classifier's answer decides which route handles the message.
Part 8 adds the memory, and part 9 makes the answer a stream (it is saved in the memory once it is complete)."""
from classifier import classify
from explain import explain_code
from generate import generate_code

OTHER_MESSAGE = (
    "I'm a coding assistant: I can explain code or write new code. Could you tell me what you need? "
    'For example: "Explain this code: ..." or "Write a Python function that ...".'
)


def route(message, memory):
    """Return the chosen route, the sources found in Chroma and the answer as a stream of text pieces."""
    intent, preferences = classify(message, memory)
    if preferences:   # the message changed the preferences: the new list replaces the old one
        memory.set_preferences(preferences)

    if intent == "explain":
        return intent, [], explain_code(message, memory)   # no sources: this route does not use RAG
    if intent == "other":
        # not a coding request: no search, no LLM answer, and we do not ask for a solution
        if preferences:
            return intent, [], [f"Got it! Your preferences for this chat are now: {preferences}."]
        return intent, [], [OTHER_MESSAGE]
    sources, stream = generate_code(message, memory)
    return intent, sources, stream


if __name__ == "__main__":
    from memory import ConversationMemory

    memory = ConversationMemory()
    memory.new_conversation()   # the test runs in a new chat; the old chats stay saved

    # A small conversation: the 2nd and 3rd messages only make sense with the memory.
    # The 4th one checks that a request with no knowledge in Chroma is still "not relevant",
    # and the 5th one that a message that is not a coding request gets a short reply.
    messages = [
        "Write a Python function that checks if a string is a palindrome. I prefer short comments in my code.",
        "Now make it ignore spaces and capital letters",
        "Which line removes the spaces?",
        "Train a CNN model on the MNIST dataset with PyTorch",
        "asdfgh",
    ]
    for message in messages:
        intent, sources, stream = route(message, memory)
        print("=" * 60)
        print("Message:", message)
        print("Route:", intent)
        print("Sources:", sources)
        print("Answer:")
        answer = ""
        for piece in stream:   # print each piece as soon as it arrives
            print(piece, end="", flush=True)
            answer += piece
        print()
        memory.add_exchange(message, answer, intent, sources)

    print("=" * 60)
    print("Saved preferences:", memory.preferences)
    print("Saved messages:", len(memory.messages))
