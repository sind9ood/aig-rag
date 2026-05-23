import os
import statistics
import numpy as np

from dotenv import load_dotenv
from edgar.core import set_identity
from edgar.entity import Company
from typing import Any, Dict, List

from src.config.runtime_aliases import get_variable_aliases
from src.config.variable_paths import VARIABLE_PATHS
from .chunk_builder import preprocess_section

TARGET_LABELS = {name: get_variable_aliases(name) for name in VARIABLE_PATHS}


class FilingChunkOrchestrator:
    """
    FilingChunkOrchestrator is responsible for orchestrating the loading of EDGAR filings and building of chunks with metadata.
    """
    def __init__(self, company_cik=None, filing_form=None, identity_email=None):
        self.target_labels = TARGET_LABELS
        self.company_cik = company_cik
        self.filing_form = filing_form

    def get_raw_filings(self):
        # Set identity for SEC EDGAR access
        load_dotenv()

        set_identity(os.getenv("EDGAR_IDENTITY_EMAIL"))

        company = Company(str(self.company_cik))
        filings = company.get_filings(form=self.filing_form)
        return filings

    
    def build(self, max_filings=3, max_chunk_chars=4000, chunk_overlap_chars=800):

        self.filings = self.get_raw_filings()

        chunks = []
        for filing in self.filings[:max_filings]:
            cur_obj = filing.obj()
            import logging
            logging.info(f"Processing filing: {cur_obj.filing_date}")            
            for item in cur_obj.items:
                section = item.lower()
                logging.info(f"  Processing section: {section}")
                try:
                    source_text = cur_obj[item]
                except Exception as exc:
                    logging.warning(f"Skipping item {item}: unable to read source document ({exc})")
                    continue

                if source_text is None:
                    continue

                if not isinstance(source_text, str):
                    source_text = str(source_text)

                source_text = source_text.strip()
                if not source_text:
                    logging.warning(f"Skipping item {item}: source document is empty")
                    continue    

                # preprocess into chunks with metadata
                filing_year = cur_obj.filing_date.year
                chunks += preprocess_section(
                    source_text,
                    section,
                    filing_year,
                    self.target_labels,
                    max_chunk_chars=max_chunk_chars,
                    chunk_overlap_chars=chunk_overlap_chars,
                )
        return chunks

    def summarize_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not chunks:
            return {
                "total_chunks": 0,
                "table_chunks": 0,
                "without_table_chunks": 0,
                "chunk_char_stats": {},
                "chunk_line_stats": {},
                "chunk_size_distribution": {},
            }

        char_sizes = [len(chunk.get("text", "")) for chunk in chunks]
        line_sizes = [len((chunk.get("text", "") or "").splitlines()) for chunk in chunks]
        table_chunks = sum(1 for chunk in chunks if int(chunk.get("table_row_count") or 0) > 0)

        def _quantile(values_sorted, q):
            if not values_sorted:
                return 0.0
            return float(np.quantile(values_sorted, q))

        def _dynamic_histogram(values):
            if not values:
                return {}
            counts, edges = np.histogram(values, bins="auto")
            return {
                f"{int(round(edges[i]))}-{int(round(edges[i + 1]))}": int(counts[i])
                for i in range(len(counts))
            }

        def _basic_stats(values):
            values_sorted = sorted(values)
            return {
                "min": values_sorted[0],
                "max": values_sorted[-1],
                "mean": round(statistics.mean(values), 2),
                "median": statistics.median(values),
                "stdev": round(statistics.pstdev(values), 2),
                "p10": round(_quantile(values_sorted, 0.10), 2),
                "p25": round(_quantile(values_sorted, 0.25), 2),
                "p75": round(_quantile(values_sorted, 0.75), 2),
                "p90": round(_quantile(values_sorted, 0.90), 2),
            }

        return {
            "total_chunks": len(chunks),
            "table_chunks": table_chunks,
            "without_table_chunks": len(chunks) - table_chunks,
            "chunk_char_stats": _basic_stats(char_sizes),
            "chunk_line_stats": _basic_stats(line_sizes),
            "chunk_char_histogram": _dynamic_histogram(char_sizes),
            "chunk_line_histogram": _dynamic_histogram(line_sizes),
        }