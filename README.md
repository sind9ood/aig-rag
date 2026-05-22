# AIG 10-K RAG Workflow

This repository builds a retrieval pipeline for AIG 10-K filings, with:
- model-based line classification (`table_row` vs `narrative`, footer handled separately),
- simplified chunking for indexing,
- Chroma indexing,
- hybrid retrieval (BM25 on chunk text + embedding on summary/context),
- LLM-based extraction and evaluation.

## 1) Setup

Run from repo root:

```bash
cd /Users/jshin/Workspace/aig-rag
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Required environment variable:

- `OPENROUTER_API_KEY`
- `EDGAR_IDENTITY_EMAIL` (email used by SEC EDGAR identity)

Optional:

- `OPENROUTER_MODEL` (default: `openai/gpt-4o-mini`)

Create `.env` in repo root:

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
EDGAR_IDENTITY_EMAIL=you@example.com
OPENROUTER_MODEL=openai/gpt-4o-mini
```

ChromaDB setup (short):
- Chroma is installed from `requirements.txt` during setup.
- Start Chroma server (DB path is `./data`):

```bash
source .venv/bin/activate
chroma run --path ./data
```

- Then initialize/update indexed data:

```bash
source .venv/bin/activate
python index_chunks_to_chroma.py --max-filings 2
```

## 2) Current Pipeline

Start Chroma server once per session:

```bash
source .venv/bin/activate
chroma run --path ./data
```

### Step A. Optional: review generated chunks before indexing

Generate chunk preview for a single filing item:

```bash
source .venv/bin/activate
python review_session_data.py \
  --max-filings 3 \
  --filing-index 1 \
  --max-chunk-chars 4000 \
  --chunk-overlap-chars 800 \
  --out data/review_session_data.json
```

### Step B. Build and index chunks to Chroma

```bash
source .venv/bin/activate
python index_chunks_to_chroma.py --max-filings 2
```

### Step C. Run app

```bash
source .venv/bin/activate
streamlit run src/app.py
```

The app loads existing `data/chroma` automatically.
If DB files exist but collection has no chunks, run Step B first.

### Step D. Evaluate retrieval + extraction

```bash
source .venv/bin/activate
python rag_search.py
```

Useful options:

```bash
python rag_search.py --year 2021
python rag_search.py --limit 2
python rag_search.py --retrieval-review-path output/eval_results.json
```

### Step E. Retrieval-only diagnostics (no extraction)

```bash
source .venv/bin/activate
python rag_search.py --year 2021 --limit 2
```

## 3) Current Chunking Logic (Simplified)

Chunking is implemented in `src/indexing/chunk_builder.py` and currently does:

1. classify lines (`table_row` or `narrative`, skipping footer lines),
2. split chunks only when label type changes (`table` <-> `narrative`),
3. merge any chunk smaller than 500 chars into previous chunk,
4. add character overlap from previous chunk (`--chunk-overlap-chars`),
5. compute summary once at the end from final merged content.

Table summary logic:
- if a line starts with "The following table ...", use that as summary,
- otherwise use LLM summary when API key is available,
- fallback to first table line / page header.

## 4) Current Retrieval Logic

Retriever: `src/retrieval/retrieve_docs.py` (`TableAwareRetriever`)

Current ranking path:
- BM25 similarity against chunk `text`,
- embedding similarity against:
  - chunk `summary` if available,
  - else `page_context + page_section`.

Fusion:
- Reciprocal rank fusion (RRF) combines BM25 and embedding ranks.

Query building:
- `src/config/runtime_aliases.py` builds search query using cached LLM outputs (`@lru_cache`):
  - alias terms,
  - time-period include/exclude terms,
  - retrieval query plan (`query`, `locate_terms`, `table_terms`).

There is no separate legacy reranking stage anymore.

## 5) Training / Pseudo Labels

Train table detector model:

```bash
source .venv/bin/activate
python src/indexing/train/train_table_detector.py
```

Model artifact:
- `model/table_row_classifier.joblib`

Runtime behavior:
- The app/retrieval code loads this saved model from `model/table_row_classifier.joblib`.
- Model loading is cached in-process, so it is loaded once and reused until process restart.

Generate pseudo labels for CSV line data: (Need human review for editing)

```bash
source .venv/bin/activate
python -m src.indexing.train.pseudo_label_table_rows --input-dir data
```
