import json
import os
import re
from dotenv import load_dotenv

from src.llm.openrouter_client import chat_completion
from src.prompts.retrieval_prompts import build_extraction_prompt, build_extraction_system_prompt
from src.config.runtime_aliases import get_variable_aliases


DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
ADVANCED_MODEL = os.getenv("OPENROUTER_ADVANCED_MODEL", "anthropic/claude-sonnet-4.6")


load_dotenv()


def extract_from_metadata(
    chunks,
    variable_name,
    target_year,
    max_docs=3,
):
    """Validate retrieved docs, then summarize them with OpenRouter."""
    retrieved_docs = _collect_docs(chunks, variable_name, target_year, max_docs)

    if not retrieved_docs:
        return {
            "value": None,
            "retrieved_docs": [],
            "source": "not_found",
            "raw_response": None,
        }
    
    aliases = get_variable_aliases(variable_name)
    prompt = build_extraction_prompt(
        chunks=retrieved_docs,
        variable_name=variable_name,
        target_year=target_year,
        aliases=aliases,
    )

    if variable_name in ["total_revenue", "shareholder_equity"]:
        response_text = call_openrouter(prompt, model=DEFAULT_MODEL)
    else:
        response_text = call_openrouter(prompt, model=ADVANCED_MODEL)
    # call LLM to extract the variable value based on retrieved docs and examples
    # response_text = call_openrouter(prompt, model=DEFAULT_MODEL)
    parsed = parse_json_response(response_text)

    if parsed is not None:

        parsed_value = str(parsed.get("value") or "").strip()
        supporting_docs = parsed.get("supporting_docs")
        if not parsed_value and (supporting_docs == [] or supporting_docs is None):
            parsed_value = "N/A"
        return {
            "value": parsed_value or response_text.strip(),
            "retrieved_docs": retrieved_docs,
            "raw_response": response_text,
            "supporting_docs": supporting_docs
        }

    # Handle malformed JSON-like empty outputs such as:
    # { VALUE: , SUPPORTING_DOCS: [] }
    if _is_empty_value_response(response_text):
        return {
            "value": "N/A",
            "retrieved_docs": retrieved_docs,
            "raw_response": response_text,
            "supporting_docs": [],
        }

    return {
        "value": response_text.strip() if response_text else None,
        "retrieved_docs": retrieved_docs,
        "raw_response": response_text,
        "supporting_docs": [],
    }


def _excerpt_around_line(lines, center_index, radius=2, max_chars=500):
    start = max(0, center_index - radius)
    end = min(len(lines), center_index + radius + 1)
    excerpt = "\n".join(lines[start:end]).strip()
    return excerpt[:max_chars]


def _build_value_excerpt(chunk_text, variable_name, max_chars=500):
    lines = [line for line in (chunk_text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    aliases = get_variable_aliases(variable_name)

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if "table_row" in lowered and any(alias in lowered for alias in aliases):
            return _excerpt_around_line(lines, idx, max_chars=max_chars)

    for idx, line in enumerate(lines):
        lowered = line.lower()
        if any(alias in lowered for alias in aliases):
            return _excerpt_around_line(lines, idx, max_chars=max_chars)

    for idx, line in enumerate(lines):
        if "table_row" in line.lower():
            return _excerpt_around_line(lines, idx, max_chars=max_chars)

    return (chunk_text or "")[:max_chars]


def _collect_docs(chunks, variable_name, target_year, max_docs):
    """Collect chunks that cover the target year window, regardless of chunk source type."""
    docs = []
    seen_keys = set()
    for chunk in chunks:
        filing_year = chunk.get("year")
        if filing_year is None:
            continue
        year_index = filing_year - target_year
        if not 0 <= year_index <= 2:
            continue
        table_row_count = chunk.get("table_row_count")
        if table_row_count is None:
            table_row_count = len(chunk.get("table_rows", [])) or 1
        excerpt = _build_value_excerpt(chunk.get("text", ""), variable_name)
        full_text = chunk.get("text") or excerpt or ""
        dedupe_key = (
            filing_year,
            chunk.get("section"),
            excerpt,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        docs.append({
            "filing_year": filing_year,
            "section": chunk.get("section"),
            "excerpt": excerpt,
            "full_text": full_text,
            "table_row_count": table_row_count,
            "year_index": year_index,
            "bm25_score": chunk.get("bm25_score"),
            "embedding_score": chunk.get("embedding_score"),
            "hybrid_embedding_rank": chunk.get("hybrid_embedding_rank"),
        })
        if len(docs) >= max_docs:
            break
    return docs


def call_openrouter(prompt, model=DEFAULT_MODEL):
    return chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": build_extraction_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        timeout=120,
    )


def parse_json_response(text):
    if not text:
        return None

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: try to extract malformed JSON with uppercase keys or unquoted values
        return _extract_malformed_json(text)


def _extract_malformed_json(text):
    """Extract value and supporting_docs from malformed JSON patterns like {VALUE: BBB+, SUPPORTING_DOCS: [1]}"""
    if not text:
        return None
    
    cleaned = text.strip().lower()
    
    # Extract value (handles both quoted and unquoted values)
    value_match = re.search(r'[\{,]\s*"?value"?\s*:\s*(["\']?)([^,\}]+?)\1\s*[,\}]', cleaned, re.IGNORECASE)
    if value_match:
        value = value_match.group(2).strip().strip('"\'')
        
        # Extract supporting_docs
        docs_match = re.search(r'[\{,]\s*"?supporting_docs"?\s*:\s*\[([^\]]*)\]', cleaned, re.IGNORECASE)
        supporting_docs = []
        if docs_match:
            docs_str = docs_match.group(1)
            # Extract all numbers from the list
            supporting_docs = [int(d.strip()) for d in docs_str.split(',') if d.strip().isdigit()]
        
        return {
            "value": value if value and value.lower() != 'null' else None,
            "supporting_docs": supporting_docs
        }
    
    return None


def _is_empty_value_response(text):
    if not text:
        return False

    cleaned = text.strip().lower()
    # Normalize code fences if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    has_empty_value = bool(re.search(r"\bvalue\b\s*:\s*(?:,|$|\}|null|\"\")", cleaned))
    has_empty_supporting_docs = bool(re.search(r"\bsupporting_docs\b\s*:\s*\[\s*\]", cleaned))
    return has_empty_value and has_empty_supporting_docs