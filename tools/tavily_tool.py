import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")



client = TavilyClient(api_key=api_key)


def tavily_search(query: str) -> str:
    response = client.search(
        query=query,
        max_results=2,
    )
    results = []

    for i, r in enumerate(response.get("results", []), 1):
        title = r.get("title", "unknown")
        url = r.get("url", "")
        snippet = r.get("content", "").strip()

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        results.append(f"{i}. **{title}**\n  {url}\n  {snippet}")

    return "\n\n".join(results) 