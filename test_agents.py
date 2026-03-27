import json
from pathlib import Path
from types import SimpleNamespace
import langgraph_agents as mod


# ---------------------------
# Agent1 (Playwright)
# ---------------------------

def test_agent1_success(monkeypatch, tmp_path):
    def fake_capture(output_file=None):
        out = tmp_path / Path(output_file).name
        out.write_text("pngdata")
        return str(out)

    monkeypatch.setattr(mod, "capture_jira_ticket_screenshot", fake_capture)

    payload = json.dumps({"issue_key": "TEST-1"})
    result = mod.agent1_playwright({"text": payload})

    assert result["text"] == payload


def test_agent1_failure(monkeypatch):

    def fake_browser_error(*args, **kwargs):
        raise Exception("Browser error")
    monkeypatch.setattr(
        mod,
        "capture_jira_ticket_screenshot",
        fake_browser_error,
    )

    payload = json.dumps({"issue_key": "TEST-1"})
    result = mod.agent1_playwright({"text": payload})

    # Should still pass state forward
    assert result["text"] == payload


# ---------------------------
# Agent2 (Jira fetch)
# ---------------------------

def test_agent2_success(monkeypatch):
    sample_issue = {
        "issue_key": "SCRUM-1",
        "title": "Test Title",
        "description": "D",
    }

    monkeypatch.setattr(mod, "get_issue_details", lambda k: sample_issue)

    result = mod.agent2({"text": "SCRUM-1", "trace": []})
    parsed = json.loads(result["text"])

    assert parsed["issue_key"] == "SCRUM-1"
    assert parsed["title"] == "Test Title"


def test_agent2_failure(monkeypatch):
    def fake_browser(*args, **kwargs):
        raise Exception("404")
    monkeypatch.setattr(
        mod,
        "get_issue_details",
        fake_browser,
    )

    result = mod.agent2({"text": "SCRUM-404", "trace": []})

    assert result["text"].startswith("ERROR:")


# ---------------------------
# Agent3 (LLM summary)
# ---------------------------

def test_agent3_success(monkeypatch):
    issue = {"issue_key": "TEST-1", "title": "T", "description": "D"}

    class DummyChain:
        def invoke(self, _):
            return SimpleNamespace(
                content="Summary:\n- short\n\nReason Not Processed:\n- none"
            )
    class DummyPrompt:
        def __or__(self, _):
            return DummyChain()

    monkeypatch.setattr(
        mod.PromptTemplate,
        "from_template",
        staticmethod(lambda _: DummyPrompt()),
    )

    result = mod.agent3({"text": json.dumps(issue)})

    assert "Summary:" in result["text"]
    assert "Reason Not Processed:" in result["text"]


def test_agent3_bad_json():
    result = mod.agent3({"text": "invalid-json"})

    assert "failed to parse" in result["text"] or "(none)" in result["text"]


def test_agent3_llm_failure(monkeypatch):
    class BadChain:
        def invoke(self, _):
            raise Exception("LLM down")

    class DummyPrompt:
        def __or__(self, _):
            return BadChain()

    monkeypatch.setattr(
        mod.PromptTemplate,
        "from_template",
        staticmethod(lambda _: DummyPrompt()),
    )

    result = mod.agent3({"text": json.dumps({"issue_key": "SCRUM-1"})})

    assert "LLM call failed" in result["text"]