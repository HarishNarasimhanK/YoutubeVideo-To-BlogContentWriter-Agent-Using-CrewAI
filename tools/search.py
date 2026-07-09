from __future__ import annotations

import re
import urllib.request
import xml.etree.ElementTree as ET

import wikipedia
from langchain_core.tools import tool

def get_video_id(url: str) -> str:
    """Extract the video ID from any YouTube URL format."""
    pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/)|youtu\.be/)([\w-]+)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url

@tool("search_wikipedia_tool")
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a topic."""
    try:
        results = wikipedia.search(query)
        if not results:
            return f"No Wikipedia pages found for query '{query}'."
        page_title = results[0]
        page = wikipedia.page(page_title, auto_suggest=False)
        summary = wikipedia.summary(page_title, sentences=5, auto_suggest=False)
        return (
            f"Source: Wikipedia\n"
            f"Title: {page.title}\n"
            f"URL: {page.url}\n"
            f"Summary:\n{summary}"
        )
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
            return (
                f"Source: Wikipedia\n"
                f"Title: {page.title}\n"
                f"URL: {page.url}\n"
                f"Summary:\n{summary}"
            )
        except Exception as inner_e:
            return f"Disambiguation error. Options: {e.options[:5]}. Inner error: {inner_e}"
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {e}"

@tool("search_arxiv_tool")
def search_arxiv(query: str) -> str:
    """Search arXiv for academic papers on technical concepts."""
    clean_query = query.replace(" ", "+")
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{clean_query}&start=0&max_results=3"
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        ns = "http://www.w3.org/2005/Atom"
        entries = []
        for entry in root.findall(f"{{{ns}}}entry"):
            title   = entry.find(f"{{{ns}}}title").text.strip()
            id_url  = entry.find(f"{{{ns}}}id").text.strip()
            summary = entry.find(f"{{{ns}}}summary").text.strip().replace("\n", " ")
            entries.append(
                f"Paper Title: {title}\n"
                f"URL/Citation: {id_url}\n"
                f"Summary: {summary[:300]}...\n"
            )
        if not entries:
            return f"No arXiv papers found for '{query}'."
        return "\n---\n".join(entries)
    except Exception as e:
        return f"Error searching arXiv for '{query}': {e}"