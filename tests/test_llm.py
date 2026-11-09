"""Tests for the shared LLM layer (no network)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel

from matomeru.config import RuntimeConfig
from matomeru.llm import (
    MAX_COMMENT_CHARS,
    MAX_COMMENTS_INCLUDED,
    MAX_POST_CHARS,
    RedditItemSummary,
    build_reddit_prompt,
    make_agent,
    summarize_reddit_item,
)
from matomeru.models import RedditComment, RedditThread

PROFILE = "Interested in local Japan events, hiking, and machine learning."


def make_thread(
    *,
    n_comments: int = 2,
    body_chars: int = 10,
    selftext: str = "post body",
) -> RedditThread:
    return RedditThread(
        id="abc123",
        subreddit="r/japan",
        title="A thread title",
        score=42,
        permalink="https://reddit.com/r/japan/comments/abc123/",
        selftext=selftext,
        comments=[RedditComment(author=f"user{i}", score=i, body="c" * body_chars) for i in range(n_comments)],
    )


def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        sagashi_mcp_url="http://localhost:8000/mcp",
        sagashi_mcp_token=None,
        notes_mcp_url="http://localhost:8001/mcp",
        llm_base_url="http://localhost:9999/v1",
        llm_api_key=None,
        llm_model="test-model",
        smtp_host="localhost",
        smtp_port=25,
        smtp_user="",
        smtp_password="",
        smtp_from="a@b.c",
        smtp_to="d@e.f",
        tz_iana="Asia/Tokyo",
        state_db_path="state.db",
        arxiv_api_base="http://export.arxiv.org/api/query",
    )


# --- build_reddit_prompt ----------------------------------------------------


def test_prompt_sections_in_fixed_order():
    prompt = build_reddit_prompt(PROFILE, make_thread())
    roles = [prompt.index(s) for s in ("== ROLE ==", "== INTEREST PROFILE ==", "== RULES ==", "== CONTENT ==")]
    assert roles == sorted(roles)
    assert prompt.index("== REFERENCE METADATA ==") > roles[-1]


def test_prompt_includes_profile():
    prompt = build_reddit_prompt(PROFILE, make_thread())
    assert PROFILE in prompt


def test_prompt_truncates_selftext_with_marker():
    long_text = "x" * (MAX_POST_CHARS + 500)
    prompt = build_reddit_prompt(PROFILE, make_thread(selftext=long_text))
    assert "[truncated]" in prompt
    # The full untruncated text must not appear.
    assert long_text not in prompt


def test_prompt_keeps_short_selftext_untruncated():
    prompt = build_reddit_prompt(PROFILE, make_thread(selftext="short"))
    assert "[truncated]" not in prompt


def test_prompt_caps_comments():
    prompt = build_reddit_prompt(PROFILE, make_thread(n_comments=MAX_COMMENTS_INCLUDED + 5, body_chars=5))
    assert prompt.count("- author: user") == MAX_COMMENTS_INCLUDED
    assert "user9" in prompt
    assert f"user{MAX_COMMENTS_INCLUDED}" not in prompt


def test_prompt_truncates_comment_bodies():
    prompt = build_reddit_prompt(PROFILE, make_thread(n_comments=1, body_chars=MAX_COMMENT_CHARS + 100))
    assert prompt.count("[truncated]") == 1


def test_prompt_permalink_and_id_as_metadata_only():
    prompt = build_reddit_prompt(PROFILE, make_thread())
    metadata = prompt[prompt.index("== REFERENCE METADATA ==") :]
    assert "https://reddit.com/r/japan/comments/abc123/" in metadata
    assert "abc123" in metadata
    # Not elsewhere in the prompt (id appears once as metadata; the permalink
    # also contains it, so exactly two occurrences total).
    assert prompt.count("https://reddit.com/r/japan/comments/abc123/") == 1
    assert prompt.count("abc123") == 2
    assert "id: abc123" in metadata
    assert prompt.index("id: abc123") > prompt.index("== REFERENCE METADATA ==")


# --- summarize_reddit_item ---------------------------------------------------


def canned_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(
                content=json.dumps(
                    {
                        "summary": "A summary of the thread.",
                        "why_relevant": "Matches the interest profile.",
                    }
                )
            )
        ]
    )


def agent_with_canned_output(model: FunctionModel) -> Agent[None, RedditItemSummary]:
    return Agent(model, output_type=RedditItemSummary)


@pytest.mark.asyncio
async def test_summarize_reddit_item_returns_validated_output():
    agent = agent_with_canned_output(FunctionModel(canned_response))
    thread = make_thread()
    result = await summarize_reddit_item(agent, PROFILE, thread)
    assert isinstance(result, RedditItemSummary)
    assert result.summary == "A summary of the thread."
    assert result.why_relevant == "Matches the interest profile."


@pytest.mark.asyncio
async def test_summarize_reddit_item_prompt_reaches_model():
    captured: dict[str, str] = {}

    def capturing_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        user_parts = [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        captured["prompt"] = "\n".join(str(c) for c in user_parts)
        return canned_response(messages, info)

    agent = agent_with_canned_output(FunctionModel(capturing_response))
    thread = make_thread()
    await summarize_reddit_item(agent, PROFILE, thread)

    prompt = captured["prompt"]
    assert "A thread title" in prompt  # title text present
    assert "machine learning" in prompt  # profile keyword present


@pytest.mark.asyncio
async def test_summarize_reddit_item_propagates_model_errors():
    def failing_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    agent = agent_with_canned_output(FunctionModel(failing_response))
    with pytest.raises(RuntimeError, match="model exploded"):
        await summarize_reddit_item(agent, PROFILE, make_thread())


# --- make_agent --------------------------------------------------------------


def test_make_agent_builds_agent_instance():
    agent = make_agent(runtime_config())
    assert isinstance(agent, Agent)


def test_make_agent_model_name_and_base_url():
    agent = make_agent(runtime_config())
    model = agent.model
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "test-model"
    # Keep loose: only check the base url made it through the provider/client.
    assert "localhost:9999" in str(model.client.base_url)


# --- RedditItemSummary -------------------------------------------------------


def test_reddit_item_summary_rejects_non_string_summary():
    with pytest.raises(ValidationError):
        RedditItemSummary(summary=123, why_relevant="text")  # ty: ignore[invalid-argument-type]
