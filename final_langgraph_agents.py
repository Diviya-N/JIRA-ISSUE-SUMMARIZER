import os
import json
import asyncio
from typing import Dict, List, Any

import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, END
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness, ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from playwright_agents import capture_jira_ticket_screenshot

load_dotenv()

# Configuration from environment (fallbacks can be added)
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
API_TOKEN = os.getenv("jira_api_key")
EMAIL = os.getenv("jira_email")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq model to use (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _extract_text_from_atlassian_doc(node: dict) -> str:
    """Recursively extract plain text from Atlassian document nodes."""
    parts = []

    if not node:
        return ""

    if isinstance(node, dict) and "text" in node and isinstance(node["text"], str):
        parts.append(node["text"])

    if isinstance(node, dict) and "attrs" in node and isinstance(node["attrs"], dict):
        t = node["attrs"].get("text")
        if isinstance(t, str):
            parts.append(t)

    if isinstance(node, dict) and "content" in node and isinstance(node["content"], list):
        for child in node["content"]:
            parts.append(_extract_text_from_atlassian_doc(child))

    return "".join([p for p in parts if p])


def get_issue_details(issue_key: str) -> dict:
    if not (EMAIL and API_TOKEN):
        raise EnvironmentError("Both JIRA email and API token must be set in environment variables")

    url = f"{JIRA_BASE_URL.rstrip('/')}/rest/api/3/issue/{issue_key}"
    resp = requests.get(
        url,
        auth=HTTPBasicAuth(EMAIL, API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    issue = resp.json()

    title = issue.get("fields", {}).get("summary")
    description_node = issue.get("fields", {}).get("description")
    description = _extract_text_from_atlassian_doc(description_node)
    status = issue.get("fields", {}).get("status", {}).get("name")

    # Comments
    comments_url = f"{JIRA_BASE_URL.rstrip('/')}/rest/api/3/issue/{issue_key}/comment"
    c_resp = requests.get(
        comments_url,
        auth=HTTPBasicAuth(EMAIL, API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    c_resp.raise_for_status()
    comments_data = c_resp.json()
    comments = []
    for c in comments_data.get("comments", []):
        body = c.get("body")
        text = _extract_text_from_atlassian_doc(body)
        comments.append(text)

    # Attachments
    attachments = issue.get("fields", {}).get("attachment", []) or []
    attachment_contents = []
    for att in attachments:
        att_url = att.get("content")
        try:
            aresp = requests.get(att_url, auth=HTTPBasicAuth(EMAIL, API_TOKEN), timeout=30)
            aresp.raise_for_status()
            content = aresp.text
        except Exception:
            content = "<failed to download>"
        attachment_contents.append({"filename": att.get("filename"), "content": content})

    result = {
        "issue_key": issue_key,
        "title": title,
        "description": description,
        "status": status,
        "comments": comments,
        "attachments": attachment_contents,
    }

    return result


def get_issue_raw(issue_key: str) -> Dict[str, Any]:
    """
    Fetch the raw Jira issue JSON, including all fields.
    Used for character-length / token-length comparison vs preprocessed text.
    """
    if not (EMAIL and API_TOKEN):
        raise EnvironmentError("Both JIRA email and API token must be set in environment variables")

    url = f"{JIRA_BASE_URL.rstrip('/')}/rest/api/3/issue/{issue_key}"
    resp = requests.get(
        url,
        auth=HTTPBasicAuth(EMAIL, API_TOKEN),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def build_preprocessed_issue_text(issue: Dict) -> str:
    """
    Data-extractor agent output → single normalized string.
    This is what will later be chunked and sent to the summarizer agents.
    """
    parts: List[str] = []

    parts.append(f"ISSUE_KEY: {issue.get('issue_key') or ''}".strip())
    parts.append(f"TITLE: {issue.get('title') or ''}".strip())
    parts.append(f"STATUS: {issue.get('status') or ''}".strip())

    desc = (issue.get("description") or "").strip()
    if desc:
        parts.append("DESCRIPTION:")
        parts.append(desc)

    comments = issue.get("comments") or []
    if comments:
        parts.append("\nCOMMENTS (ordered oldest → newest):")
        for i, c in enumerate(comments, start=1):
            c = (c or "").strip()
            if not c:
                continue
            parts.append(f"- COMMENT {i}: {c}")

    attachments = issue.get("attachments") or []
    if attachments:
        parts.append("\nATTACHMENT EXCERPTS:")
        for att in attachments:
            name = (att.get("filename") or "").strip()
            content = (att.get("content") or "").strip()
            if not content:
                continue
            label = f"ATTACHMENT: {name}" if name else "ATTACHMENT"
            parts.append(f"{label}:\n{content}")

    return "\n".join(parts)


def _split_text_into_chunks(text: str, max_words: int = 900, overlap_words: int = 80) -> List[str]:
    """Very lightweight word-based chunking for context-window control."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + max_words, n)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == n:
            break
        start = max(0, end - overlap_words)

    return chunks


class State(TypedDict):
    text: str
    mini_summaries: List[str]
    trace: List[dict]


#=====================================================================
# Causal coherence evaluation: does the summary logically follow from the issue details?
#=====================================================================

def extract_sections(resp: str):
    summary = ""
    reason = ""
    trace = ""

    if "Summary:" in resp:
        temp = resp.split("Summary:", 1)[1]

        if "Reason Not Processed:" in temp:
            summary, temp2 = temp.split("Reason Not Processed:", 1)

            if "Trace:" in temp2:
                reason, trace = temp2.split("Trace:", 1)

    return summary.strip(), reason.strip(), trace.strip()


async def causal_coherence_eval(resp: str, issue_json: str):
    summary, reason, trace = extract_sections(resp)

    judge_prompt = f"""
You are evaluating reasoning quality.

ISSUE JSON:
{issue_json}

SUMMARY:
{summary}

REASON_NOT_PROCESSED:
{reason}

TRACE:
{trace}

Task:
1. Check whether the reasoning trace logically and correctly leads to the summary and reason_not_processed.
2. Ensure reasoning steps are consistent with the issue JSON.
3. Score from 1 to 5.

Respond ONLY with a number.
"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=GROQ_API_KEY,
    )

    response = await llm.ainvoke(judge_prompt)

    try:
        score = float(response.content.strip())
    except Exception:
        score = 3.0  # neutral fallback

    print("Causal Coherence Score:", score)

    return score


# =====================================================================
# Faithfulness: does the summary accurately reflect the content of the Jira issue?
# =====================================================================

def extract_trace(resp: str):
    if "Trace" in resp:
        trace = resp.split("Trace:", 1)[1].strip()
        return trace
    return ""


async def faithfullness_eval(resp: str, issue: str):
    # Use Groq via OpenAI-compatible endpoint for RAGAS
    groq_client = AsyncOpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
    llm = llm_factory(GROQ_MODEL, client=groq_client, max_tokens=4000)

    trace = extract_trace(resp)

    faith = Faithfulness(llm=llm)
    recall = ContextRecall(llm=llm)
    precision = ContextPrecision(llm=llm)

    response_faith = await faith.ascore(
        user_input="Generate a summary and reason_not_processed for the given Jira issue",
        response=resp,
        retrieved_contexts=[issue],
    )

    trace_faith = await faith.ascore(
        user_input="Evaluate reasoning trace grounding",
        response=trace,
        retrieved_contexts=[issue],
    )

    trace_recall = await recall.ascore(
        user_input="Evaluate reasoning trace completeness",
        retrieved_contexts=[trace],
        reference=issue,
    )

    trace_precision = await precision.ascore(
        user_input="Evaluate reasoning trace relevance",
        retrieved_contexts=[trace],
        reference=issue,
    )

    print("\n=== Faithfulness Metrics ===")
    print(f"Response faithfulness: {response_faith.value:.3f}")
    print(f"Trace faithfulness:    {trace_faith.value:.3f}")
    print(f"Trace recall:          {trace_recall.value:.3f}")
    print(f"Trace precision:       {trace_precision.value:.3f}")

    return {
        "response_faithfulness": response_faith.value,
        "trace_faithfulness": trace_faith.value,
        "trace_recall": trace_recall.value,
        "trace_precision": trace_precision.value,
    }


def agent1_playwright(state: State) -> dict:
    """agent1_playwright: capture a screenshot of the Jira ticket page using playwright.

    This agent performs a side-effect (saves a screenshot)
    """
    raw = state.get("text", "") if isinstance(state, dict) else ""
    try:
        issue = json.loads(raw)
        issue_key = issue.get("issue_key")
    except Exception:
        issue_key = os.getenv("JIRA_ISSUE_KEY") or "SCRUM-1"

    base = os.getenv("JIRA_BASE_URL")
    ticket_url = f"{base.rstrip('/')}/browse/{issue_key}"
    out_file = f"JIRA_ISSUE_SUMMARIZER\\{issue_key}_ticket.png"
    try:
        capture_jira_ticket_screenshot(ticket_url, output_file=out_file)
        print(f"Saved ticket screenshot: {out_file}")
        state["trace"].append({
        "agent": "agent1_playwright",
        "step": "Captured screenshot of Jira ticket",
        "data": {"file_path": out_file}
    })
    except Exception as e:
        print(f"playwright capture failed: {e}")

    return {"text": state.get("text", ""), "mini_summaries": state.get("mini_summaries", []), "trace": state.get("trace", [])}


def agent2_data_extractor(state: State) -> dict:
    """
    Agent 2: Data extractor from Jira using the existing REST calls.
    Takes an issue_key from state["text"], fetches Jira, and normalizes it.
    """
    raw = state.get("text", "")
    issue_key = None

    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "issue_key" in parsed:
                issue_key = parsed.get("issue_key")
            elif isinstance(parsed, str):
                issue_key = parsed
            else:
                issue_key = raw
        except Exception:
            issue_key = raw

    if not issue_key:
        issue_key = os.getenv("JIRA_ISSUE_KEY") or raw

    try:
        issue = get_issue_details(issue_key)
        preprocessed = build_preprocessed_issue_text(issue)
        print(f"[agent2] Retrieved and preprocessed Jira issue {issue_key} (chars={len(preprocessed)})")
        trace = state.get("trace", [])
        trace.append(
            {
                "agent": "agent2_data_extractor",
                "step": "Fetched and preprocessed Jira issue",
                "data": {"issue_key": issue_key, "chars": len(preprocessed)},
            }
        )
        return {"text": preprocessed, "mini_summaries": state.get("mini_summaries", []), "trace": trace}
    except Exception as e:
        err = f"ERROR: failed to fetch {issue_key}: {e}"
        print(f"[agent2] {err}")
        trace = state.get("trace", [])
        trace.append({"agent": "agent2_data_extractor", "step": "error", "data": {"error": str(e)}})
        return {"text": err, "mini_summaries": state.get("mini_summaries", []), "trace": trace}


def agent3_mini_summarizer(state: State) -> dict:
    """
    Agent 3: Mini-summarizer.
    - Chunks the normalized Jira text.
    - For each chunk, generates a short 3–4 line mini-summary.
    - Stores them in state["mini_summaries"].
    """
    raw = state.get("text", "")

    if raw.startswith("ERROR:") or not raw.strip():
        return {"text": raw, "mini_summaries": state.get("mini_summaries", []), "trace": state.get("trace", [])}

    chunks = _split_text_into_chunks(raw, max_words=900, overlap_words=80)
    print(f"[agent3] Chunk count for mini-summaries: {len(chunks)}")

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        api_key=GROQ_API_KEY,
    )

    mini_summaries: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = f"""
You are a mini summarization agent for a Jira issue.

Your task is to write a very concise 3–4 line summary of the following Jira-related text chunk.
Only use information present in the chunk. Do not hallucinate missing facts.

Chunk {idx}/{len(chunks)}:
\"\"\"
{chunk}
\"\"\"
"""
        resp = llm.invoke(prompt)
        mini_summaries.append(resp.content.strip())

    trace = state.get("trace", [])
    trace.append(
        {
            "agent": "agent3_mini_summarizer",
            "step": "Created mini summaries from normalized issue text",
            "data": {"chunk_count": len(chunks)},
        }
    )

    return {"text": raw, "mini_summaries": mini_summaries, "trace": trace}


def agent4_final_summarizer(state: State) -> dict:
    """
    Agent 4: Final summarizer.
    Consumes the mini-summaries and produces:
    - Summary
    - Reason Not Processed
    - Trace
    as a single formatted text block.
    """
    raw = state.get("text", "")
    mini_summaries = state.get("mini_summaries", [])

    if raw.startswith("ERROR:"):
        return {
            "text": f"Summary:\n\n- (none)\n\nReason Not Processed:\n\n- {raw}\n\nTrace:\n\n- (none)",
            "mini_summaries": mini_summaries,
            "trace": state.get("trace", []),
        }

    mini_summaries_text = "\n\n".join(
        f"- {s}" for s in mini_summaries if isinstance(s, str) and s.strip()
    )

    prompt = f"""
You are a helpful assistant.
You will receive a list of mini-summaries that were generated from a Jira issue and its related context.

Mini-Summaries:
{mini_summaries_text}

Produce three sections:

Summary:
- Write a concise 5-line summary of the issue (what it is, context, and what's required) based ONLY on the mini-summaries.

Reason Not Processed:
- Write a 5-line explanation of the blocker or reason why the issue is not processed, again ONLY using information from the mini-summaries.

Trace:
- Generate a bullet list (4–6 bullets) that reflects the actual reasoning steps you took using the mini-summaries.
- Each bullet should describe a concrete action or observation (e.g., "combined information from mini-summary 1 and 3 to identify main incident", "checked status-related sentences for blockers").
- The trace must be derived from the given mini-summaries, not a static template.

Format the output with clear section headers: "Summary:", "Reason Not Processed:", and "Trace:".
"""

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        api_key=GROQ_API_KEY,
    )

    resp = llm.invoke(prompt)
    text_out = resp.content.strip()

    trace = state.get("trace", [])
    trace.append(
        {
            "agent": "agent4_final_summarizer",
            "step": "Generated final summary / reason_not_processed / trace",
            "data": {"mini_summary_count": len(mini_summaries)},
        }
    )

    return {"text": text_out, "mini_summaries": mini_summaries, "trace": trace}


graph = StateGraph(State)
graph.add_node("agent1", agent1_playwright)
graph.add_node("agent2", agent2_data_extractor)
graph.add_node("agent3", agent3_mini_summarizer)
graph.add_node("agent4", agent4_final_summarizer)
graph.add_edge(START, "agent1")
graph.add_edge("agent1", "agent2")
graph.add_edge("agent2", "agent3")
graph.add_edge("agent3", "agent4")
graph.add_edge("agent4", END)


if __name__ == "__main__":
    try:
        issue_key = input("Enter JIRA issue key (or leave blank to use env/default): ").strip()
    except Exception:
        issue_key = ""

    if not issue_key:
        issue_key = os.getenv("JIRA_ISSUE_KEY", "SCRUM-2")

    print(f"Processing issue: {issue_key}\n")

    app = graph.compile()
    initial_state: State = {
        "text": json.dumps({"issue_key": issue_key}),
        "mini_summaries": [],
        "trace": [],
    }
    result = app.invoke(initial_state)

    print("\n=== Final Output (Summary / Reason / Trace) ===\n")
    print(result["text"])

    # ===============================
    # Context-window / reduction stats
    # ===============================
    pre_issue_for_eval: Dict[str, Any] | None = None
    try:
        raw_issue = get_issue_raw(issue_key)
        raw_text = json.dumps(raw_issue, ensure_ascii=False)
        raw_chars = len(raw_text)

        pre_issue = get_issue_details(issue_key)
        pre_issue_for_eval = pre_issue
        pre_text = build_preprocessed_issue_text(pre_issue)
        pre_chars = len(pre_text)

        reduced_chars = raw_chars - pre_chars
        reduction_pct = (reduced_chars / raw_chars * 100.0) if raw_chars else 0.0

        mini_chunks = _split_text_into_chunks(pre_text, max_words=900, overlap_words=80)

        print("\n=== Context Window / Reduction Stats ===\n")
        print(f"[INFO] Raw Jira JSON chars:              {raw_chars}")
        print(f"[INFO] Preprocessed text chars:          {pre_chars}")
        print(f"[INFO] Character reduction:              {reduced_chars} ({reduction_pct:.1f}% reduction)")
        print(f"[INFO] Mini-summarizer chunk count:      {len(mini_chunks)}")
    except Exception as e:
        print(f"[WARN] Failed to compute context-window stats: {e}")

    # ===============================
    # Evaluation: faithfulness & causal coherence
    # ===============================
    if pre_issue_for_eval is None:
        try:
            pre_issue_for_eval = get_issue_details(issue_key)
        except Exception as e:
            print(f"[WARN] Failed to fetch issue details for evaluation: {e}")

    if pre_issue_for_eval is not None:
        issue_json_str = json.dumps(pre_issue_for_eval, indent=2, ensure_ascii=False)
        try:
            asyncio.run(faithfullness_eval(resp=result["text"], issue=issue_json_str))
        except Exception as e:
            print(f"[WARN] Faithfulness evaluation failed: {e}")

        try:
            asyncio.run(causal_coherence_eval(result["text"], issue_json_str))
        except Exception as e:
            print(f"[WARN] Causal coherence evaluation failed: {e}")
