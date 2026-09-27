"""A caller-selected model replaces every model the Agent overlay resolved."""

from __future__ import annotations

from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xagent.core.model.chat.basic.base import BaseLLM
from xagent.core.model.chat.basic.call_boundary import UnavailableVisionModel
from xagent.web.models.task import TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_service_manager import AgentServiceManager
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskModelOverride,
    TaskModelOverrideError,
    TaskSetupSnapshot,
    _TaskFields,
    apply_task_model_override,
)


class NamedLLM(BaseLLM):
    def __init__(self, name: str, abilities: list[str] | None = None) -> None:
        self._name = name
        self._abilities = abilities or ["chat", "tool_calling"]

    @property
    def abilities(self) -> List[str]:
        return self._abilities

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def supports_thinking_mode(self) -> bool:
        return False

    async def chat(self, messages, **kwargs):  # type: ignore[override]
        raise AssertionError(f"{self._name} must not be called")


# The models the Agent overlay resolved -- an explicit platform model saved
# on the agent row.
OVERLAY_GENERAL = NamedLLM("platform-general")
OVERLAY_FAST = NamedLLM("platform-fast")
OVERLAY_VISION = NamedLLM("platform-vision", ["chat", "vision"])
OVERLAY_COMPACT = NamedLLM("platform-compact")


def _snapshot(tool_categories: list[str] | None) -> TaskSetupSnapshot:
    agent_config: dict[str, Any] = {
        "llms": [OVERLAY_GENERAL, OVERLAY_FAST, OVERLAY_VISION, OVERLAY_COMPACT],
        "saved_model_ids": {"general": "platform/x"},
        "saved_model_descriptors": {},
        "execution_mode": "balanced",
        "instructions": "Be helpful.",
        "skills": [],
        "knowledge_bases": [],
        "tool_categories": tool_categories,
    }
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=42,
            user_id=1,
            status=TaskStatus.PENDING,
            source="external",
            agent_id=7,
            agent_config={},
            model_name=None,
            compact_model_name=None,
            execution_mode="balanced",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="react",
        task_llm=OVERLAY_GENERAL,
        task_fast_llm=OVERLAY_FAST,
        task_vision_llm=OVERLAY_VISION,
        task_compact_llm=OVERLAY_COMPACT,
        agent=AgentRuntimeFields(
            id=7,
            name="Assistant",
            status="published",
            instructions="Be helpful.",
            agent_creator_user_id=1,
        ),
        agent_config=agent_config,
        excluded_agent_id=None,
    )


class TestApply:
    def test_every_model_slot_takes_the_override(self):
        selected = NamedLLM("selected", ["chat", "tool_calling", "vision"])
        applied = apply_task_model_override(
            _snapshot(["basic", "image"]),
            TaskModelOverride(llm=selected, vision_llm=selected),
        )
        assert applied.task_llm is selected
        assert applied.task_fast_llm is selected
        assert applied.task_compact_llm is selected
        assert applied.task_vision_llm is selected
        assert applied.agent_config["tool_categories"] == ["basic", "image"]

    def test_no_vision_model_means_vision_is_unavailable(self):
        selected = NamedLLM("selected")
        applied = apply_task_model_override(
            _snapshot(["basic"]), TaskModelOverride(llm=selected)
        )
        assert isinstance(applied.task_vision_llm, UnavailableVisionModel)
        assert applied.task_vision_llm is not OVERLAY_VISION

    def test_excluded_categories_leave_the_selection(self):
        selected = NamedLLM("selected")
        original = _snapshot(["web_search", "image", "video", "basic"])
        applied = apply_task_model_override(
            original,
            TaskModelOverride(
                llm=selected, excluded_tool_categories=frozenset({"image", "video"})
            ),
        )
        assert applied.agent_config["tool_categories"] == ["web_search", "basic"]
        # The caller's snapshot is not mutated.
        assert original.agent_config["tool_categories"] == [
            "web_search",
            "image",
            "video",
            "basic",
        ]

    def test_exclusion_needs_an_explicit_selection(self):
        with pytest.raises(TaskModelOverrideError):
            apply_task_model_override(
                _snapshot(None),
                TaskModelOverride(
                    llm=NamedLLM("selected"),
                    excluded_tool_categories=frozenset({"image"}),
                ),
            )


@pytest.mark.asyncio
async def test_the_built_service_and_its_tools_use_only_the_override() -> None:
    selected = NamedLLM("selected")
    snapshot = apply_task_model_override(
        _snapshot(["basic", "image", "vision"]),
        TaskModelOverride(llm=selected, excluded_tool_categories=frozenset({"image"})),
    )
    manager = AgentServiceManager()
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = MagicMock(
        status=TaskStatus.PENDING
    )
    create_tools = AsyncMock(return_value=([], MagicMock()))
    built: dict[str, Any] = {}

    class RecordingService(MagicMock):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__()
            built.update(kwargs)
            self.workspace = None

    with (
        patch(
            "xagent.web.services.agent_service_manager.load_task_setup_snapshot_sync",
            return_value=snapshot,
        ),
        patch.object(manager, "_load_persisted_conversation_history"),
        patch.object(manager, "_load_persisted_execution_context", new=AsyncMock()),
        patch(
            "xagent.web.services.agent_service_manager.create_task_tracer",
            return_value=MagicMock(),
        ),
        patch(
            "xagent.web.services.agent_service_manager.create_default_tools",
            new=create_tools,
        ),
        patch("xagent.web.sandbox_manager.get_sandbox_manager", return_value=None),
        patch(
            "xagent.web.services.agent_service_manager.AgentService",
            new=RecordingService,
        ),
    ):
        await manager.get_agent_for_task(
            task_id=42,
            db=db,
            user=User(id=1, username="u", password_hash="h", is_admin=False),
            task_setup_snapshot=snapshot,
        )

    tool_kwargs = create_tools.await_args.kwargs
    assert tool_kwargs["llm"] is selected
    assert isinstance(tool_kwargs["vision_model"], UnavailableVisionModel)
    spec = tool_kwargs["tool_selection_spec"]
    assert "image" not in spec.categories
    assert "basic" in spec.categories
    assert built["llm"] is selected
    assert built["fast_llm"] is selected
    assert built["compact_llm"] is selected
    assert isinstance(built["vision_llm"], UnavailableVisionModel)
    overlay = {OVERLAY_GENERAL, OVERLAY_FAST, OVERLAY_VISION, OVERLAY_COMPACT}
    assert not overlay & {
        built["llm"],
        built["fast_llm"],
        built["compact_llm"],
        built["vision_llm"],
        tool_kwargs["llm"],
        tool_kwargs["vision_model"],
    }
