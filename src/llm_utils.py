"""LLM utility functions.

Handles compatibility with reasoning models (DeepSeek-R1, Tongyi-DeepResearch)
that output to `reasoning_content` instead of `content`.
"""


def get_llm_content(message) -> str:
    """
    Extract content from LLM response message.

    Reasoning models (like DeepSeek-R1, Tongyi-DeepResearch) output their
    chain-of-thought to `reasoning_content` and may leave `content` empty
    for short prompts.

    Args:
        message: OpenAI-compatible message object with content attribute

    Returns:
        Content string, falling back to reasoning_content if content is empty
    """
    # Try standard content field first
    content = getattr(message, 'content', None) or ""

    if not content:
        # Fall back to reasoning_content for reasoning models
        content = getattr(message, 'reasoning_content', None) or ""

    return content
