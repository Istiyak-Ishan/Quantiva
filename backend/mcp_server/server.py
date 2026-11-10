"""
backend/mcp_server/server.py
────────────────────────────────────────────────────────────────────────────────
Quantiva MCP (Model Context Protocol) Server

Exposes Quantiva's document intelligence as MCP tools so that ANY
MCP-compatible agent host (Claude Desktop, Antigravity IDE, Continue.dev,
custom agents) can call Quantiva without any bespoke integration.

Tools exposed:
  • search_documents   – hybrid semantic + keyword search over user docs
  • summarize_document – generate a concise summary of a single document
  • ask_question       – full reasoning-agent Q&A with citations

Transport: stdio  (run as a sidecar process, communicates over stdin/stdout)

Usage — add to Claude Desktop config (~/.claude/config.json):
  {
    "mcpServers": {
      "quantiva": {
        "command": "python",
        "args": ["-m", "mcp_server.server"],
        "cwd": "/path/to/quantiva/backend",
        "env": { "QUANTIVA_API_URL": "http://localhost:8000/api/v1",
                 "QUANTIVA_TOKEN": "your-jwt-token" }
      }
    }
  }

Usage — add to Antigravity IDE (settings.json):
  {
    "mcp.servers": [{
      "name": "quantiva",
      "command": "python -m mcp_server.server",
      "cwd": "I:\\\\AI Engineer Projects\\\\Quantiva\\\\backend",
      "env": { "QUANTIVA_API_URL": "http://localhost:8000/api/v1",
               "QUANTIVA_TOKEN": "your-jwt-token" }
    }]
  }
"""

import os
import sys
import json
import asyncio
import httpx
from typing import Optional

# ── FastMCP ───────────────────────────────────────────────────────────────────
# pip install fastmcp
from fastmcp import FastMCP

# ── Config ────────────────────────────────────────────────────────────────────
API_URL   = os.getenv("QUANTIVA_API_URL", "http://localhost:8000/api/v1")
API_TOKEN = os.getenv("QUANTIVA_TOKEN", "")

mcp = FastMCP(
    name="Quantiva",
    instructions=(
        "Quantiva is an enterprise AI document agent. "
        "Use search_documents to find specific information, "
        "summarize_document for high-level overviews, and "
        "ask_question for complex multi-step reasoning with citations."
    ),
)

# ── HTTP client helper ────────────────────────────────────────────────────────

def _headers() -> dict:
    if not API_TOKEN or API_TOKEN == "mock-token":
        raise ValueError("QUANTIVA_TOKEN must be a real access token from /auth/login")
    return {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type":  "application/json",
    }

async def _post(path: str, payload: dict) -> dict:
    """POST to the Quantiva REST API and return parsed JSON."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{API_URL}{path}", json=payload, headers=_headers())
        r.raise_for_status()
        return r.json()

async def _get(path: str, params: dict | None = None) -> dict:
    """GET from the Quantiva REST API."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{API_URL}{path}", params=params, headers=_headers())
        r.raise_for_status()
        return r.json()

# ── Tool 1: search_documents ──────────────────────────────────────────────────

@mcp.tool()
async def search_documents(
    query: str,
    doc_ids: list[str] | None = None,
    top_k: int = 5,
) -> str:
    """
    Perform hybrid semantic + keyword search across the user's uploaded documents.

    Returns the top matching text chunks with source metadata (filename, page/row,
    relevance score). Use this when you need specific passages or facts from documents.

    Args:
        query:   Natural language search query.
        doc_ids: Optional list of specific document IDs to restrict the search.
                  If omitted, searches the authenticated owner's documents.
        top_k:   Number of results to return (1–20, default 5).

    Returns:
        JSON string containing a list of matching chunks with metadata.
    """
    try:
        payload = {
            "query":   query,
            "doc_ids": doc_ids or [],
            "top_k":   min(max(top_k, 1), 20),
        }
        result = await _post("/retrieval/search", payload)
        chunks = result.get("chunks", [])

        if not chunks:
            return json.dumps({
                "status":  "no_results",
                "message": "No matching chunks found. Try a different query or check doc_ids.",
                "chunks":  [],
            })

        # Format chunks for readability
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append({
                "rank":      i,
                "doc_id":    chunk.get("doc_id"),
                "filename":  chunk.get("filename", "unknown"),
                "page":      chunk.get("page_or_row"),
                "score":     round(chunk.get("score", 0.0), 4),
                "text":      chunk.get("text", "")[:500],  # cap at 500 chars per chunk
            })

        return json.dumps({
            "status": "ok",
            "query":  query,
            "count":  len(formatted),
            "chunks": formatted,
        }, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({
            "status": "error",
            "code":   e.response.status_code,
            "detail": e.response.text[:200],
        })
    except Exception as e:
        return json.dumps({"status": "error", "detail": str(e)})


# ── Tool 2: summarize_document ────────────────────────────────────────────────

@mcp.tool()
async def summarize_document(
    doc_id: str,
    style: str = "concise",
) -> str:
    """
    Generate a summary of an entire document using the Quantiva reasoning agent.

    Use this when the user asks for an overview, executive summary, or wants to
    understand what a document is about before diving into specifics.

    Args:
        doc_id:  The document ID to summarize (get from search_documents results).
        style:   Summary style — "concise" (3-5 sentences) or "detailed" (full outline).

    Returns:
        A structured summary with key points and source attribution.
    """
    try:
        style_instruction = (
            "Provide a concise 3-5 sentence summary covering the main topic, "
            "key findings, and any action items or conclusions."
            if style == "concise"
            else
            "Provide a detailed structured summary with: 1) Document purpose, "
            "2) Key sections and their findings, 3) Important numbers/dates, "
            "4) Conclusions and recommendations."
        )

        # Create a session and run the agent
        session = await _post("/chat/sessions", {"document_ids": [doc_id]})
        session_id = session["session_id"]

        # We use the non-streaming endpoint for MCP (tools need a single return value)
        result = await _post(
            "/chat/message",
            {
                "session_id": session_id,
                "query":      f"Summarize this document. {style_instruction}",
            },
        )

        return json.dumps({
            "status":     "ok",
            "doc_id":     doc_id,
            "style":      style,
            "summary":    result.get("answer", ""),
            "confidence": result.get("confidence"),
            "citations":  result.get("citations", [])[:3],
        }, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({
            "status": "error",
            "code":   e.response.status_code,
            "detail": e.response.text[:200],
        })
    except Exception as e:
        return json.dumps({"status": "error", "detail": str(e)})


# ── Tool 3: ask_question ──────────────────────────────────────────────────────

@mcp.tool()
async def ask_question(
    question: str,
    doc_ids: list[str] | None = None,
    session_id: str | None = None,
) -> str:
    """
    Ask a complex question and get a fully cited, reasoned answer from Quantiva's
    multi-step reasoning agent.

    The agent will:
      1. Retrieve the most relevant document chunks using hybrid search
      2. Reason step-by-step over the evidence
      3. Return a structured answer with citations (filename + page number)
      4. Include a confidence score (0.0–1.0)

    Use this for analytical questions, comparisons, multi-document synthesis,
    or any question requiring reasoning beyond simple lookup.

    Args:
        question:   The question to answer.
        doc_ids:    Specific document IDs to restrict the search (optional).
        session_id: Reuse an existing session for conversational follow-up (optional).

    Returns:
        JSON with answer, citations, confidence score, and optional chart_url.
    """
    try:
        # Create or reuse session
        if not session_id:
            session = await _post("/chat/sessions", {"document_ids": doc_ids or []})
            session_id = session["session_id"]
        elif doc_ids is not None:
            existing = await _get(f"/chat/sessions/{session_id}")
            if set(existing["document_ids"]) != set(doc_ids):
                return json.dumps({"status": "error", "detail": "Document scope cannot override an existing session; create a new session"})

        result = await _post(
            "/chat/message",
            {
                "session_id": session_id,
                "query":      question,
            },
        )

        # Format citations for easy reading
        citations = result.get("citations", [])
        formatted_citations = [
            {
                "source":   c.get("filename", "unknown"),
                "page":     c.get("page_or_row"),
                "excerpt":  c.get("text_excerpt", "")[:200],
            }
            for c in citations
        ]

        return json.dumps({
            "status":       "ok",
            "question":     question,
            "answer":       result.get("answer", ""),
            "confidence":   result.get("confidence"),
            "citations":    formatted_citations,
            "chart_url":    result.get("chart_url"),
            "session_id":   session_id,  # return for follow-up questions
            "tokens_used":  result.get("tokens_used"),
        }, indent=2)

    except httpx.HTTPStatusError as e:
        return json.dumps({
            "status": "error",
            "code":   e.response.status_code,
            "detail": e.response.text[:200],
        })
    except Exception as e:
        return json.dumps({"status": "error", "detail": str(e)})


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run over stdio transport — required for MCP host compatibility
    mcp.run(transport="stdio")
