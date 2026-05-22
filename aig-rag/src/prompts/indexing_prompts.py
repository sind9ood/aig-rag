def build_table_summary_prompt(
    page_hint,
    context_lines,
    table_lines,
    initial_summary = None,
):
    context_text = "\n".join(context_lines[-5:])
    table_preview = "\n".join(table_lines[:8])
    initial_text = initial_summary if initial_summary else "none"

    return f"""You are summarizing a financial table from an SEC 10-K filing.

Page context: {page_hint if page_hint else "unknown"}

Initial summary candidate: {initial_text}

Lines before the table:
{context_text}

First lines of the table:
{table_preview}

Write a summary header for this table in 20 words or fewer.
If an initial summary candidate is provided, refine and merge it with page context.
If no initial summary candidate exists, infer from the provided lines and page context.
Be specific: include what financial metric, time period, and entity (e.g. "AIG Consolidated Statements of Income 2022-2024").
Return ONLY the summary string, nothing else."""
