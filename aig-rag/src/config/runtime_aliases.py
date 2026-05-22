import json
import os
import re
from functools import lru_cache
from dotenv import load_dotenv

from src.llm.openrouter_client import chat_completion, has_openrouter_api_key
from src.prompts.retrieval_prompts import (
    build_query_plan_system_prompt,
    build_retrieval_query_plan_prompt,
)
from src.config.variable_paths import VARIABLE_PATHS


DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


load_dotenv()


def _normalize_alias(alias):
    alias = (alias or "").strip().lower()
    alias = alias.replace("\xa0", " ")
    alias = re.sub(r"\s+", " ", alias)
    alias = alias.strip(" .,:;|\"'")
    return alias


def clear_query_rewrite_caches():
    get_variable_aliases.cache_clear()
    get_retrieval_query_plan.cache_clear()


def _call_openrouter_retrieval_query_plan(
    variable_name,
    base_query,
    target_year,
):
    if not has_openrouter_api_key():
        return {}

    config = VARIABLE_PATHS[variable_name]
    prompt = build_retrieval_query_plan_prompt(
        variable_name=variable_name,
        display_name=config.display_name,
        description=config.description,
        base_query=base_query,
        target_year=target_year,
    )

    try:
        content = chat_completion(
            model=DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": build_query_plan_system_prompt(),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=60,
        ).strip()
    except RuntimeError:
        return {}

    try:
        if content.startswith("```"):
            content = content.strip("`")
            content = content.removeprefix("json").strip()
        parsed = json.loads(content)
    except Exception:
        return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


@lru_cache(maxsize=None)
def get_variable_aliases(variable_name):
    config = VARIABLE_PATHS[variable_name]
    fixed = config.aliases if config.aliases else []
    return [_normalize_alias(alias) for alias in fixed if _normalize_alias(alias)]


@lru_cache(maxsize=None)
def get_retrieval_query_plan(
    variable_name,
    base_query,
    target_year,
):
    parsed = _call_openrouter_retrieval_query_plan(variable_name, base_query, target_year)

    locate_terms_raw = parsed.get("locate_terms", []) if isinstance(parsed, dict) else []
    table_terms_raw = parsed.get("table_terms", []) if isinstance(parsed, dict) else []
    constraint_terms_raw = parsed.get("constraint_terms", []) if isinstance(parsed, dict) else []

    locate_terms = []
    for term in locate_terms_raw:
        normalized = _normalize_alias(str(term))
        if normalized and normalized not in locate_terms:
            locate_terms.append(normalized)

    table_terms = []
    for term in table_terms_raw:
        normalized = _normalize_alias(str(term))
        if normalized and normalized not in table_terms:
            table_terms.append(normalized)

    constraint_terms = []
    for term in constraint_terms_raw:
        normalized = _normalize_alias(str(term))
        if normalized and normalized not in constraint_terms:
            constraint_terms.append(normalized)

    return {
        "locate_terms": locate_terms[:8],
        "table_terms": table_terms[:8],
        "constraint_terms": constraint_terms[:8],
    }


def _compact_terms(terms, max_terms):
    compacted = []
    for term in terms:
        cleaned = _normalize_alias(term)
        if not cleaned:
            continue
        if re.search(r"\b(?:19|20)\d{2}\b", cleaned):
            continue
        if len(cleaned.split()) > 6:
            continue
        if cleaned in compacted:
            continue
        compacted.append(cleaned)
        if len(compacted) >= max_terms:
            break
    return compacted


def build_retrieval_queries(variable_name, base_query, target_year=None):
    aliases = _compact_terms(get_variable_aliases(variable_name), 6)
    query_plan = get_retrieval_query_plan(variable_name, base_query, target_year)

    locate_terms = _compact_terms(query_plan.get("locate_terms", []), 3)
    table_terms = _compact_terms(query_plan.get("table_terms", []), 3)
    constraint_terms = _compact_terms(query_plan.get("constraint_terms", []), 3)

    bm25_query = " ".join(aliases) or " ".join(base_query.strip().split()[:6])

    embedding_terms = []
    for group in [aliases, locate_terms, table_terms, constraint_terms]:
        for term in group:
            if term not in embedding_terms:
                embedding_terms.append(term)

    embedding_terms = embedding_terms[:7]

    embedding_query = " ".join(embedding_terms)
    if not embedding_query:
        embedding_query = "financial statement table value"

    return {
        "bm25_query": bm25_query,
        "embedding_query": embedding_query,
        "query_plan": query_plan,
    }


def build_search_query(variable_name, base_query, target_year=None):
    queries = build_retrieval_queries(variable_name, base_query, target_year)
    return queries.get("embedding_query", "")

