from typing import Optional

def build_prompt(question: str, context: str, instructions: Optional[str] = None) -> str:
    """
    Build a simple instruction + context + question prompt.
    Keep the template minimal and predictable.
    """
    if not instructions:
        instructions = (
            "You are an assistant. Use the provided context to answer the question. "
            "If the answer is not contained in the context, say you don't know. "
            "Cite sources in square brackets like [doc_id:page]. Be concise."
        )

    parts = [
        instructions.strip(),
        "CONTEXT:",
        context.strip() if context else "(no useful context provided)",
        "QUESTION:",
        question.strip(),
        "ANSWER:"
    ]
    return "\n\n".join(parts)
