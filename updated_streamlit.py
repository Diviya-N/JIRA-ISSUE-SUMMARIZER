import os
import json
import importlib.util
from pathlib import Path
import asyncio

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent
MAIN_PATH = ROOT_DIR / "final_langgraph_agents.py"


def load_langgraph_module(path: Path):
    spec = importlib.util.spec_from_file_location("langgraph_main", str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


st.title("Jira Issue Inspector (LangGraph)")

issue_key = st.text_input("JIRA Issue Key", value=os.getenv("JIRA_ISSUE_KEY", ""))

# Top-right LangSmith traces link (reads URL from env)
trace_url = os.getenv("LANGSMITH_TRACE_URL") or os.getenv("LANGSMITH_RUN_URL") or ""

run = st.button("Inspect Issue")

if run and issue_key:
    with st.spinner("Running agents..."):
        try:
            mod = load_langgraph_module(MAIN_PATH)

            # Call agent2 logic directly (via main.get_issue_details) to get structured issue details
            try:
                agent2_output = mod.get_issue_details(issue_key)
            except Exception as e:
                agent2_output = {"error": str(e)}

            # Invoke the LangGraph flow to run agent1 -> agent2 -> agent3 -> agent4
            try:
                initial_state = {
                    "text": json.dumps({"issue_key": issue_key}),
                    "mini_summaries": [],
                    "trace": [],
                }
                result = mod.graph.compile().invoke(initial_state)
            except Exception as e:
                result = {"text": f"Graph invocation failed: {e}"}
        except Exception as e:
            result = {"text": f"Failed to load main.py module: {e}"}

    # Display results
    st.header("Results")

    # Agent1 image
    st.subheader("Image saved from agent1_playwright")
    # agent1 is expected to save to JIRA_ISSUE_SUMMARIZER/<issue_key>_ticket.png
    img_path = ROOT_DIR / f"{issue_key}_ticket.png"
    if not img_path.exists():
        img_path = ROOT_DIR / "JIRA_ISSUE_SUMMARIZER" / f"{issue_key}_ticket.png"
    if img_path.exists():
        st.image(str(img_path), caption=f"Screenshot for {issue_key}")
    else:
        st.warning("Screenshot not found. Agent1 may have failed to capture or saved to a different path.")

    # Agent2 output (structured Jira issue)
    st.subheader("Agent 2 (Jira data extractor) output")
    try:
        st.json(agent2_output)
    except Exception:
        st.write(str(agent2_output))
# ==================================================================
    # Agent3 output (intermediate summarizer)
    st.subheader("Agent 3 (Intermediate summarizer) output")
    try:
        agent3_output = None
        if isinstance(result, dict):
            agent3_output = (
                result.get("agent3")
                or result.get("agent_3")
                or result.get("agent3_output")
                or result.get("mini_summaries")
                or result.get("mini_summary")
                or result.get("intermediate")
                or result.get("trace")
            )

        if agent3_output is None:
            st.info("No explicit Agent 3 output found. Showing available result instead.")
            if isinstance(result, dict):
                try:
                    st.json(result)
                except Exception:
                    st.write(result)
            else:
                st.write(str(result))
        else:
            try:
                st.json(agent3_output)
            except Exception:
                st.write(agent3_output)
    except Exception as e:
        st.error(f"Failed to display Agent 3 output: {e}")
# ==================================================================
    # Final agent output (agent4_final_summarizer)
    st.subheader("Final agent (agent4_final_summarizer) output")
    final_text = result.get("text") if isinstance(result, dict) else str(result)
    st.text_area("Final summary / reason / trace", value=final_text or "", height=300)

    # Monitoring / LangSmith
    st.markdown("### Monitoring")
    trace_url = os.getenv("LANGSMITH_TRACE_URL") or os.getenv("LANGSMITH_RUN_URL") or "https://smith.langchain.com/"
    if trace_url:
        st.markdown(
            f'<a href="{trace_url}" target="_blank" rel="noopener noreferrer">'
            '<button style="width:100%;padding:6px;border-radius:6px;background:#0f62fe;'
            'color:white;border:none;">Open LangSmith Dashboard</button></a>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<button style="width:100%;padding:6px;border-radius:6px;background:#e0e0e0;'
            'color:#6c6c6c;border:none;" disabled>Open LangSmith Dashboard</button>',
            unsafe_allow_html=True,
        )

    # EVALUATION METRICS
    with st.sidebar:
        st.header("Evaluation Metrics")
        try:
            # Fetch issue details for evaluation context
            issue_data = mod.get_issue_details(issue_key)
            issue_json = json.dumps(issue_data, ensure_ascii=False)
            final_text = result.get("text") if isinstance(result, dict) else str(result)
# Run async evaluations from main.py
            faith_scores = asyncio.run(
                mod.faithfullness_eval(resp=final_text, issue=issue_json)
            )
            coherence_score = asyncio.run(
                mod.causal_coherence_eval(final_text, issue_json)
            )

            # Response Quality
            st.subheader("Response Quality")
            st.metric("Faithfulness", round(faith_scores["response_faithfulness"], 3))
            st.caption("How accurately the final summary reflects the Jira issue (groundedness of output).")

            # Trace Grounding
            st.subheader("Trace Grounding")
            st.metric("Trace Faithfulness", round(faith_scores["trace_faithfulness"], 3))
            st.caption("How well the reasoning trace is grounded in the actual Jira data.")
            st.metric("Trace Recall", round(faith_scores["trace_recall"], 3))
            st.caption("How much of the relevant Jira information the trace successfully used (completeness).")
            st.metric("Trace Precision", round(faith_scores["trace_precision"], 3))
            st.caption("How much of the reasoning trace is relevant and non-hallucinated (purity).")

            # Compute F1
            p = faith_scores["trace_precision"]
            r = faith_scores["trace_recall"]
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0
            st.metric("Trace F1 Score", round(f1, 3))
            st.caption("Balanced score combining trace precision and recall.")

            # Reasoning Quality
            st.subheader("Reasoning Quality")
            st.metric("Causal Coherence", round(coherence_score, 3))
            st.caption("How logically the reasoning steps lead to the summary and conclusions.")
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

else:
    st.info("Enter a JIRA issue key and click 'Inspect Issue' to run the agents.")