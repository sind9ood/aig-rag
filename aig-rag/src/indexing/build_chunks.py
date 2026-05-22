"""

This module owns:
- loading 10-K filings from EDGAR
- building chunks by delegating to pre_process
- reporting chunk statistics
"""

import os
import math
import statistics

from edgar.core import set_identity
from edgar.entity import Company
from typing import Any, Dict, List

from src.config.runtime_aliases import get_variable_aliases
from src.config.variable_paths import VARIABLE_PATHS
from .pre_process import preprocess_section

TARGET_LABELS = {name: get_variable_aliases(name) for name in VARIABLE_PATHS}


class ChunkBuilder:
    def __init__(self):
        self.target_labels = TARGET_LABELS

    def get_raw_filings(self) -> List[Dict[str, Any]]:
        # Set identity for SEC EDGAR access
        # load from .env or system environment

        set_identity("sind9ood@gmail.com")  # replace with your email or load from env variable

        # Initialize Company with AIG's CIK
        aig = Company("0000005272")

        # Get all 10-K filings for AIG
        aig_10k = aig.get_filings(form="10-K")
        return aig_10k
    
    def build(self, filings, max_filings=3, max_chunk_chars=4000, chunk_overlap_chars=800):

        chunks = []
        for filing in filings[:max_filings]:
            cur_obj = filing.obj()
            print(f"Processing filing: {cur_obj.filing_date}")            
            for item in cur_obj.items:
                section = item.lower()
                print(f"  Processing section: {section}")
                try:
                    source_text = cur_obj[item]
                except Exception as exc:
                    print(f"Skipping item {item}: unable to read source document ({exc})")
                    continue

                if source_text is None:
                    print(f"Skipping item {item}: source document is None")
                    continue

                if not isinstance(source_text, str):
                    source_text = str(source_text)

                source_text = source_text.strip()
                if not source_text:
                    print(f"Skipping item {item}: source document is empty")
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
                "has_table_rows": 0,
                "without_table_rows": 0,
                "chunk_char_stats": {},
                "chunk_line_stats": {},
                "chunk_size_distribution": {},
            }

        char_sizes = [len(chunk.get("text", "")) for chunk in chunks]
        line_sizes = [len((chunk.get("text", "") or "").splitlines()) for chunk in chunks]
        table_chunks = sum(1 for chunk in chunks if chunk.get("has_table_rows"))

        def _quantile(values_sorted: List[int], q: float) -> float:
            if not values_sorted:
                return 0.0
            if len(values_sorted) == 1:
                return float(values_sorted[0])
            idx = (len(values_sorted) - 1) * q
            lo = math.floor(idx)
            hi = math.ceil(idx)
            if lo == hi:
                return float(values_sorted[lo])
            frac = idx - lo
            return float(values_sorted[lo] * (1 - frac) + values_sorted[hi] * frac)

        def _dynamic_histogram(values: List[int]) -> Dict[str, int]:
            if not values:
                return {}
            values_sorted = sorted(values)
            if len(values_sorted) == 1:
                only = values_sorted[0]
                return {f"{only}-{only}": 1}

            q1 = _quantile(values_sorted, 0.25)
            q3 = _quantile(values_sorted, 0.75)
            iqr = max(q3 - q1, 1.0)
            bin_width = max((2 * iqr) / (len(values_sorted) ** (1 / 3)), 1.0)
            min_v = values_sorted[0]
            max_v = values_sorted[-1]
            bin_count = int(math.ceil((max_v - min_v) / bin_width))
            bin_count = max(5, min(bin_count, 20))

            edges = [min_v + (max_v - min_v) * i / bin_count for i in range(bin_count + 1)]
            counts = [0] * bin_count

            for value in values:
                if value == max_v:
                    idx = bin_count - 1
                else:
                    ratio = (value - min_v) / max(max_v - min_v, 1)
                    idx = min(int(ratio * bin_count), bin_count - 1)
                counts[idx] += 1

            histogram = {}
            for i in range(bin_count):
                left = int(round(edges[i]))
                right = int(round(edges[i + 1]))
                if i == bin_count - 1:
                    label = f"{left}-{right}"
                else:
                    label = f"{left}-{max(left, right - 1)}"
                histogram[label] = counts[i]
            return histogram

        def _basic_stats(values: List[int]) -> Dict[str, float]:
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
            "has_table_rows": table_chunks,
            "without_table_rows": len(chunks) - table_chunks,
            "chunk_char_stats": _basic_stats(char_sizes),
            "chunk_line_stats": _basic_stats(line_sizes),
            "chunk_char_histogram": _dynamic_histogram(char_sizes),
            "chunk_line_histogram": _dynamic_histogram(line_sizes),
        }
    
    
