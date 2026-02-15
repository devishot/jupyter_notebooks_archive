"""
Ollama Chat with LangChain and Memory
Uses phi3 model running locally on Ollama
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the Ollama model
llm = ChatOllama(
    model="phi3",
    base_url="http://localhost:11434",
    temperature=0.7,
)

# Store for chat histories
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat history for a session"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    # ("system", "You are a helpful AI assistant. Use the conversation history to provide contextual responses."),
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

# Create chain with memory
chain = prompt | llm

# Wrap chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)


def chat(user_input: str, session_id: str = "default") -> str:
    """
    Send a message and get a response with conversation context

    Args:
        user_input: The user's message
        session_id: Session ID for conversation history

    Returns:
        The model's response
    """
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    return response.content


def clear_history(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in store:
        store[session_id].clear()


def main():
    """
    Run automated chat demonstration with memory
    """
    print("=" * 60)
    print("Ollama Chat with Memory Demo (phi3 model)")
    print("=" * 60)
    print()

    # Predefined conversation to demonstrate memory
    conversation = [
        "Hello! My name is Alice and I'm learning about LLMs.",
        "What's my name?",
        "I'm interested in machine learning and AI. Can you recommend some topics to study?",
        "Based on what I told you about my interests, what should I focus on first?",
    ]

    try:
        for i, user_message in enumerate(conversation, 1):
            print(f"\n[Request {i}]")
            print(f"User: {user_message}")
            print("Assistant: ", end="", flush=True)

            response = chat(user_message)
            print(response)
            print("-" * 60)

        print("\n✓ Conversation completed successfully!")
        print("Memory was used across all requests.")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()