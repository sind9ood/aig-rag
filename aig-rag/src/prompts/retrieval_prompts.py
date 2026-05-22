from typing import Any


def build_query_plan_system_prompt():
    return "You return only JSON with keys locate_terms, table_terms, constraint_terms."


def build_extraction_system_prompt():
    return "You answer using only the supplied SEC filing context."


def build_retrieval_query_plan_prompt(
    variable_name,
    display_name,
    description,
    base_query,
    target_year,
):
    _ = target_year
    return (
        "You are creating retrieval location cues for financial-report search. "
        "Return strict JSON only with keys: locate_terms, table_terms, constraint_terms. "
        "Do not return a query string. "
        "Use general accounting knowledge about where values are usually reported. "
        "Do not rely on filing-specific section labels (for example Item 8). "
        "locate_terms: short phrases for likely locations in financial reports (for example notes, balance sheet, income statement, cash flow statement). "
        "table_terms: short table/statement cues likely near the value. "
        "constraint_terms: short business-document context phrases tied to the target value (for example annual, consolidated, attributable to, in millions, parent). "
        "Return 2-3 terms for each category. "
        "Each returned phrase should be concise (ideally 1-3 words). "
        "Choose high-value terms so merged unique terms across categories are about 5 total. "
        "Do not include any year terms or numeric year values. "
        "Do not include explanations.\n\n"
        f"Variable key: {variable_name}\n"
        f"Display name: {display_name}\n"
        f"Description: {description}\n"
        f"Base query: {base_query}\n"
    )

def build_extraction_prompt(
    chunks,
    variable_name,
    target_year,
    aliases,
):
    variable_labels = {
        "total_revenues": "total revenue",
        "total_assets": "total assets",
        "net_income": "net income",
        "sp_rating": "S&P long-term debt rating",
    }
    target_label = variable_labels.get(variable_name, variable_name.replace("_", " "))

    doc_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        doc_blocks.append(
            f"Document {index}\n"
            f"Filing year: {chunk.get('filing_year')}\n"
            f"Section: {chunk.get('section')}\n"
            f"Table row count: {chunk.get('table_row_count')}\n"
            f"Excerpt:\n{chunk.get('excerpt', '')}"
        )
    

    docs_text = "\n\n---\n\n".join(doc_blocks)

    task_instruction = (
            f"Extract the exact reported numeric value or categorical value for '{target_label}' in {target_year-1}.\n"
            "Return valid JSON with exactly two keys: value and supporting_docs.\n"
            f"Possible aliases for this target include: {', '.join(aliases[:8])}.\n"
            f"Think of possible formats (e.g. numeric, percentage, currency) and available representations (e.g. text, symbols) for this target.\n"
            "If the target label or aliases are clearly shown in a table, first look for the values adjacent to that row label.\n"
            "A pipe separator can appear between the row label and value (for example: 'net income (loss) | 9,123'), and should still be treated as adjacent.\n"
            "If the table structure is broken across lines, column headers (such as year labels) may appear "
            "on a separate line above the data rows, and the values for each column will follow in subsequent "
            "rows aligned to those headers — in this case, identify the column position of the target year "
            "from the header line first, then read the value at that same column position in the data row.\n"
            "Similarly, a row label may appear alone on one line with its corresponding values on the next "
            "line; treat them as belonging to the same row if no other label intervenes between them.\n"
            "Be careful when the displayed row name is only similar to the target label: similar names can refer to different financial attributes, so do not assume a partial match is correct.\n"
            "Make a clear distinction between full-year values and other figures such as quarterly, year-to-date, or interim amounts. If both appear, prefer the full-year value for the requested year.\n"
            "Many financial tables present values for the last three years including the target year, usually ordered from most recent year to older years. Use the column ordering carefully.\n"
            "If different values for the same target variable appear across documents, choose by majority vote among valid matches; if there is no clear majority, use the value from the first retrieved document.\n"
            "Canonical unit is millions, but do not print the word 'million'.\n"
            "If the source unit is dollars, divide by 1,000,000 to convert to millions.\n"
            "If the source unit is billions, multiply by 1,000 to convert to millions.\n"
            "If the source unit is thousands, divide by 1,000 to convert to millions.\n"
            "If the source omits units, infer from nearby context (e.g., 'in millions').\n"
            "If a number is wrapped by parentheses, treat it as a negative value (example: '(1,234)' means '-1,234').\n"
            "For losses/negative values, use '-<amount>'.\n"
            "Do not add commentary or analysis."
    )

    return (
        "You are analyzing SEC filing source documents.\n"
        f"{task_instruction}\n"
        "Use only the provided documents. Do not invent values.\n"
        "supporting_docs should be a short list of document references.\n\n"
        f"Documents:\n{docs_text}"
    )
