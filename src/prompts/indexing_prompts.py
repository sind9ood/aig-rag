def build_chunk_summary_prompt(
    page_hint=None,
    context_lines=None,
    snippet=None,
    title_lines=None,
    table_intro=None,
    initial_summary=None,
):
    """
    Build a summary prompt for any chunk type using available metadata.
    """
    prompt_parts = []
    if page_hint:
        prompt_parts.append(f"Page context: {page_hint}")
    if title_lines:
        prompt_parts.append(f"Titles: {' | '.join(title_lines)}")
    if table_intro:
        prompt_parts.append(f"Table intro: {table_intro}")
    if context_lines:
        prompt_parts.append(f"Context: {' | '.join(context_lines[-5:])}")
    if snippet:
        prompt_parts.append(f"Snippet: {snippet}")
    if initial_summary:
        prompt_parts.append(f"Initial summary candidate: {initial_summary}")

    prompt = "\n".join(prompt_parts)
    return (
        f"You are summarizing a document chunk from an SEC 10-K filing.\n"
        f"{prompt}\n\n"
        "Write a summary header for this chunk in 20 words or fewer. "
        "If an initial summary candidate is provided, refine and merge it with page context. "
        "If no initial summary candidate exists, infer from the provided lines and page context. "
        "Be specific: include what financial metric, time period, and entity if possible. "
        "Return ONLY the summary string, nothing else."
    )
