import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from src.indexing.chunk_utils import parse_financial_table

@pytest.mark.parametrize("line,expected", [
    ("Revenues$5,500 $4,800", "Revenues | 5500.0 | 4800.0"),
    ("Net Loss (5,000) (3,000)", "Net Loss | -5000.0 | -3000.0"),
    ("Equity $2,000", "Equity $2,000"),
    ("Random text", "Random text"),
])
def test_parse_financial_table(line, expected):
    result = parse_financial_table(line)
    assert result == expected
