"""LLM-backed summarizer for MemoryStore.compact().

The default summarizer in store.py just concatenates clustered memories,
which doesn't actually compress anything. claude_summarizer() asks Claude
to merge a cluster of related memories into one shorter memory that keeps
every distinct fact, so compaction genuinely bounds context/storage size.
"""

from __future__ import annotations

from collections.abc import Callable

DEFAULT_MODEL = "claude-opus-5"


def claude_summarizer(model: str = DEFAULT_MODEL, api_key: str | None = None) -> Callable[[list[str]], str]:
    """Build a summarizer callable backed by the Claude API.

    Requires the 'llm' extra. Pass the returned callable as
    `MemoryStore.compact(summarizer=...)`.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("claude_summarizer requires the 'llm' extra: pip install 'effective-memory[llm]'") from e

    client = anthropic.Anthropic(api_key=api_key)

    def summarize(contents: list[str]) -> str:
        bullets = "\n".join(f"- {c}" for c in contents)
        response = client.messages.create(
            model=model,
            max_tokens=256,
            output_config={"effort": "low"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Merge these related memories into a single, shorter memory that "
                        "preserves every distinct fact from all of them. Respond with only "
                        "the merged memory text, no preamble or explanation:\n\n" + bullets
                    ),
                }
            ],
        )
        text = next((block.text for block in response.content if block.type == "text"), "")
        return text.strip() or bullets

    return summarize
