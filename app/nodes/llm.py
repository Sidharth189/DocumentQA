from langchain_groq import ChatGroq
import os

def get_llm(model: str = "llama-3.1-2b-instant"):
    """
    Returns a Groq LangChain LLM client.
    Requires environment variable: GROQ_API_KEY
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set.")

    return ChatGroq(api_key=api_key, model_name=model)


def call_llm(prompt: str, model: str = "llama-3.1-2b-instant") -> str:
    """
    Sends a prompt to Groq's LLM and returns the generated text.
    """
    llm = get_llm(model)
    response = llm.invoke(prompt)
    return response.content
