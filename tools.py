import json
import requests
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from config import FMP_API_KEY


@tool
def ticker_search(company_name: str) -> str:
    """Search for the ticker for the given company name using live web search.

    Args:
        company_name: name of the company we need the ticker for
    """
    search_query = f"{company_name} stock ticker"
    return DuckDuckGoSearchRun().run(search_query)


@tool
def get_stock_metrics(ticker: str) -> str:
    """
    Get LIVE market metrics for the company with this tool.
    CRITICAL RULE: DO NOT use this tool to find Revenue, Net Income, or historical data.
    You MUST use the 'search_<company>_report' tool to read the PDFs for those numbers.

    Args:
        ticker: stock ticker symbol
    """
    url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data_list = response.json()
        if not data_list:
            return (
                f"Could not find data for ticker '{ticker}'. "
                "Please use the ticker_search tool to verify the symbol and try again."
            )
        data = data_list[0]
        metrics = {
            "symbol": data.get("symbol"),
            "company_name": data.get("name"),
            "live_price": data.get("price"),
            "pe_ratio": data.get("pe"),
            "market_cap": data.get("marketCap"),
            "earnings_per_share": data.get("eps"),
        }
        return json.dumps(metrics)
    except Exception as e:
        return f"API Error fetching data for {ticker}: {str(e)}"


def get_market_searcher() -> DuckDuckGoSearchRun:
    """Live news/sentiment search tool, used to cross-check risk factors from 10-Ks."""
    return DuckDuckGoSearchRun(
        name="market_searcher",
        description=(
            "USE THIS TOOL to search the live internet for current events, breaking news, "
            "and recent market sentiment. Crucial: When checking 'Risk Factors' found in a "
            "10-K report, use this tool to search if those specific risks (e.g., 'supply "
            "chain disruption', 'lawsuits') are currently happening in the news today."
        ),
    )
