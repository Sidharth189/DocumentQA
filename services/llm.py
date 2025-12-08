import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env into environment variables early
load_dotenv()


def get_llm():

    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in environment variables.")

    # Create and return the Groq LLM
    llm = ChatGroq(
        model=model_name,
        temperature=0
    )

    return llm
