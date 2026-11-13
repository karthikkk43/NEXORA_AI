
"""Tavily Web Search integration for Nexora AI."""

import json
import logging
import os

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

from dotenv import load_dotenv

from app.observability import trace


log = logging.getLogger(__name__)

load_dotenv(
    Path(__file__).resolve().parent.parent / ".env"
)


# ============================================
# 1. CUSTOM SEARCH ERROR
# ============================================

class WebSearchError(Exception):

    def __init__(self, kind):
        self.kind = kind
        super().__init__(kind)


# ============================================
# 2. TAVILY WEB SEARCH
# ============================================

@trace("tavily_search")
def search_web(query):
    """
    Runs one Tavily search.

    Retry decisions are handled by LangGraph.
    """

    api_key = os.getenv(
        "TAVILY_API_KEY", ""
    ).strip()

    if not api_key:
        raise WebSearchError(
            "missing_api_key"
        )

    payload = {
        "query": query[:380],
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    request = Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode(
            "utf-8"
        ),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urlopen(
            request,
            timeout=12,
        ) as response:
            data = json.load(response)

    except HTTPError as exc:

        if exc.code in (401, 403):
            raise WebSearchError(
                "authentication_failed"
            ) from exc

        if exc.code == 429:
            raise WebSearchError(
                "rate_limited"
            ) from exc

        if 500 <= exc.code < 600:
            raise WebSearchError(
                "temporary_server_error"
            ) from exc

        raise WebSearchError(
            "request_rejected"
        ) from exc

    except (
        URLError,
        TimeoutError,
        OSError,
    ) as exc:
        raise WebSearchError(
            "network_error"
        ) from exc

    except ValueError as exc:
        raise WebSearchError(
            "invalid_response"
        ) from exc

    # ========================================
    # 3. VALIDATE API RESPONSE
    # ========================================

    if (
        not isinstance(data, dict)
        or not isinstance(
            data.get("results"),
            list,
        )
    ):
        raise WebSearchError(
            "invalid_response"
        )

    cleaned = []
    seen = set()

    for row in data["results"]:

        if not isinstance(row, dict):
            continue

        url = str(
            row.get("url", "")
        ).strip()

        if (
            urlsplit(url).scheme
            not in ("http", "https")
        ):
            continue

        if url in seen:
            continue

        seen.add(url)

        cleaned.append({
            "title": str(
                row.get("title", "")
            )[:250],
            "url": url,
            "content": str(
                row.get("content", "")
            )[:2200],
        })

    return cleaned
