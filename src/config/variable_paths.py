from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VariablePathConfig:
    display_name: str
    aliases: list[str]
    description: str
    top_k: int
    max_docs: int
    query_template: str
    fixed_context_query: str | None = None


VARIABLE_PATHS = {
    "total_revenues": VariablePathConfig(
        display_name="Total Revenues",
        aliases=["total revenues"],
        description="The total revenues (or total net revenues) for AIG in the target year, as reported in the income statement. This figure represents the total amount of money earned from AIG's business activities before any expenses are deducted. It may be labeled as 'Total Revenues', 'Total Net Revenues', 'Revenues', or similar variations in the financial statements.",
        top_k=20,
        max_docs=3,
        query_template="total revenues in {year}",
    ),
    # "total_assets": VariablePathConfig(
    #     display_name="Total Assets",
    #     aliases=["total assets"],
    #     description="The total assets for AIG in the target year, as reported in the balance sheet. This figure represents the total value of everything AIG owns, including cash, investments, property, and other assets. It may be labeled as 'Total Assets' or similar variations in the financial statements.",
    #     top_k=20,
    #     max_docs=3,
    #     query_template="total assets in {year}"
    # ),
    "shareholders_equity": VariablePathConfig(
        display_name="Total AIG shareholders' equity",
        aliases=["total aig shareholders' equity"],
        description="The total AIG shareholders' equity for AIG in the target year, as reported in the balance sheet. This figure represents the residual interest in the assets of AIG after deducting liabilities, essentially showing the net worth of the company attributable to its shareholders. It may be labeled as 'Total Shareholders' Equity', 'Total Stockholders' Equity', 'Total Equity', or similar variations in the financial statements.",
        top_k=20,
        max_docs=3,
        query_template="total aig shareholders' equity in {year}",
    ),
    "sp_rating": VariablePathConfig(
        display_name="S&P Long-term Debt Rating",
        aliases=["s&p long-term debt rating", "senior long-term debt", "s&p ratings"],
        description="The S&P long-term debt credit rating for AIG in the target year, such as 'A', 'BBB-', etc.",
        top_k=10,
        max_docs=3,
        query_template="s&p long-term debt rating in {year}",
    ),
}
