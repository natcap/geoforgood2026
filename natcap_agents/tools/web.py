"""Web search + page fetch tools, usable by any specialist that needs to reach
outside whatever project-specific data source the crew otherwise relies on.

Uses smolagents' built-in web tools rather than custom ones — DuckDuckGo search
needs no API key, and VisitWebpageTool just fetches + converts a page to markdown.
"""
from __future__ import annotations

from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

WEB_TOOLS = [DuckDuckGoSearchTool(), VisitWebpageTool()]
