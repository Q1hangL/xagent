"""Adapter and real factory-registration tests for the slide render tool."""

import asyncio
import base64
import pathlib
import shutil
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from xagent.core.tools.adapters.vibe.base import ToolCategory
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.core.workspace import TaskWorkspace

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aXioAAAAASUVORK5CYII="
)
RENDER_TOOL_NAME = "render_pptx_to_images"


def _load_adapter_module():
    try:
        from xagent.core.tools.adapters.vibe import pptx_render_tool
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"pptx render adapter module is missing: {exc}")
    return pptx_render_tool


def _load_render_module():
    try:
        from xagent.core.tools.core import pptx_render
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"pptx_render module is missing: {exc}")
    return pptx_render


def _find_creator(module):
    ToolRegistry._import_tool_modules()
    for creator, categories, selection_gate in ToolRegistry._tool_creators:
        if creator is module.create_pptx_render_tools:
            return categories, selection_gate
    return None


def _unwrap(tool: Any) -> Any:
    seen = 0
    while hasattr(tool, "_target"):
        tool = tool._target
        seen += 1
        assert seen < 5, "tool wrapper chain is unexpectedly deep"
    return tool


class _RecordingSandbox:
    """Sandbox double that only reports host-mounted workspace coverage."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def workspace_dirs_are_host_mounted(self, directories) -> bool:
        self.calls.append(("host_mount_check", tuple(directories)))
        return True


class _SandboxEnabledToolConfig:
    """Duck-typed tool config with a sandbox for the real tool factory."""

    def __init__(self, workspace, sandbox, spec):
        self._workspace = workspace
        self._sandbox = sandbox
        self._spec = spec
        self.released = 0

    def get_tool_selection_spec(self):
        return self._spec

    def get_task_runtime_workspace(self):
        return self._workspace

    def set_task_runtime_workspace(self, workspace) -> None:
        self._workspace = workspace

    def get_workspace_config(self):
        return None

    def get_sandbox(self):
        return self._sandbox

    def get_allowed_tools(self):
        return None

    def get_user_tool_overrides(self):
        return {}

    def get_user_tool_allowlist(self):
        return None

    def get_max_output_length(self) -> int:
        return 10000

    def get_max_field_count(self) -> int:
        return 100

    def get_max_recursion_depth(self) -> int:
        return 5

    def release_db_connection(self) -> None:
        self.released += 1


def test_render_creator_is_registered_for_the_ppt_category():
    module = _load_adapter_module()

    found = _find_creator(module)

    assert found is not None, "create_pptx_render_tools is not in the real registry"
    categories, selection_gate = found
    assert categories == frozenset({"ppt"})
    assert selection_gate is None
    assert module.__name__ in sys.modules


def test_registry_dispatches_the_render_creator_only_for_the_ppt_category(tmp_path):
    _load_adapter_module()
    _load_render_module()
    workspace = TaskWorkspace(id="4242", base_dir=str(tmp_path / "uploads"))
    try:
        config = _SandboxEnabledToolConfig(
            workspace,
            _RecordingSandbox(),
            ToolSelectionSpec(categories=frozenset({"ppt"})),
        )
        tools = asyncio.run(ToolRegistry.create_registered_tools(config))
        render_tools = [tool for tool in tools if tool.name == RENDER_TOOL_NAME]
        assert len(render_tools) == 1
        tool = render_tools[0]
        assert tool.metadata.category == ToolCategory.PPT
        assert tool.description
        assert tool.args_type() is not None

        basic_config = _SandboxEnabledToolConfig(
            workspace,
            _RecordingSandbox(),
            ToolSelectionSpec(categories=frozenset({"basic"})),
        )
        basic_names = [
            tool.name
            for tool in asyncio.run(ToolRegistry.create_registered_tools(basic_config))
        ]
        assert RENDER_TOOL_NAME not in basic_names
    finally:
        shutil.rmtree(workspace.workspace_dir.parent, ignore_errors=True)


def test_tool_schema_rejects_non_integer_page_arguments():
    module = _load_adapter_module()
    tool = module.PPTXRenderTool(None).get_tools()[0]

    assert tool.name == RENDER_TOOL_NAME
    schema = tool.args_type().model_json_schema()
    start_schema = schema["properties"]["start_page"]
    assert "integer" in repr(start_schema)
    assert "string" not in repr(start_schema)
    for bad in (True, 1.5, "3", [3], {"page": 3}):
        with pytest.raises(ValidationError):
            tool.args_type()(pptx_path="deck.pptx", start_page=bad)
        with pytest.raises(ValidationError):
            tool.args_type()(pptx_path="deck.pptx", end_page=bad)
    parsed = tool.args_type()(pptx_path="deck.pptx", start_page=2, end_page=4)
    assert (parsed.start_page, parsed.end_page) == (2, 4)


def test_tool_description_states_limits_range_and_host_requirement():
    module = _load_adapter_module()
    description = module.RENDER_PPTX_TO_IMAGES_DESCRIPTION
    tool = module.PPTXRenderTool(None).get_tools()[0]

    assert tool.description == description
    lowered = description.lower()
    assert "20" in description
    assert "first" in lowered and "slide" in lowered
    assert "start_page" in description
    assert "end_page" in description
    assert "4,000,000" in description
    assert "60" in description
    assert "libreoffice" in lowered
    assert "soffice" in lowered
    assert "backend" in lowered
    assert "sandbox" in lowered
    assert "truncated" in lowered
    assert "pixel-perfect" not in lowered
    assert "fidelity" not in lowered
    assert "file_id" in description


@pytest.mark.asyncio
async def test_factory_build_with_sandbox_keeps_the_tool_on_the_host(
    tmp_path, monkeypatch
):
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandbox_config import (
        resolve_sandbox_config,
    )
    from xagent.core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
        SandboxedToolWrapper,
    )

    module = _load_adapter_module()
    render = _load_render_module()
    assert _find_creator(module) is not None

    # The factory's existing workspace bootstrap creates the real directories
    # the render tool writes into.
    workspace = ToolFactory.create_workspace(
        {"base_dir": str(tmp_path / "uploads"), "task_id": "4242"}
    )
    assert isinstance(workspace, TaskWorkspace)
    assert workspace.output_dir.is_dir()

    sandbox = _RecordingSandbox()
    spec = ToolSelectionSpec(categories=frozenset({"ppt"}))
    config = _SandboxEnabledToolConfig(workspace, sandbox, spec)

    tools = await ToolFactory.create_all_tools(config)

    names = [tool.name for tool in tools]
    assert names.count(RENDER_TOOL_NAME) == 1
    tool = _unwrap(next(tool for tool in tools if tool.name == RENDER_TOOL_NAME))
    assert isinstance(tool, SandboxedToolWrapper) is False
    assert resolve_sandbox_config(tool) is None
    assert tool.metadata.category == ToolCategory.PPT
    assert tool.func.__self__._workspace is workspace
    assert sandbox.calls == [("host_mount_check", tuple(workspace.get_allowed_dirs()))]

    # Reset the sandbox spy and exercise a real render call through the tool.
    sandbox.calls.clear()
    deck = workspace.input_dir / "deck.pptx"
    deck.write_bytes(b"PK\x03\x04 synthetic")

    async def convert_pptx_to_pdf(pptx_path, output_dir, *, timeout_seconds=60.0):
        pdf_path = pathlib.Path(output_dir) / (pathlib.Path(pptx_path).stem + ".pdf")
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")
        return pdf_path

    def render_pdf_pages_to_pngs(
        pdf_path, page_numbers, output_dir, *, scale=2.0, max_pixels=4_000_000
    ):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = []
        for number in page_numbers:
            path = output_dir / f"slide_{number:03d}.png"
            path.write_bytes(PNG_1X1)
            pages.append(
                render.RenderedPage(page_number=number, path=path, width=1, height=1)
            )
        return len(page_numbers), pages

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)
    monkeypatch.setattr(render, "render_pdf_pages_to_pngs", render_pdf_pages_to_pngs)
    monkeypatch.setattr(render, "read_pdf_page_count", lambda _path: 1)
    file_ids: dict[str, str] = {}

    def register_files(files, *, db_session=None):
        ids = []
        for path, _file_id in files:
            file_id = f"00000000-0000-4000-8000-{len(ids) + 1:012d}"
            file_ids[str(path)] = file_id
            ids.append(file_id)
        return tuple(ids)

    monkeypatch.setattr(
        workspace,
        "register_files",
        register_files,
    )
    monkeypatch.setattr(
        workspace,
        "get_registered_file_id",
        lambda path: file_ids.get(str(path)),
    )

    result = await tool.run_json_async({"pptx_path": str(deck)})

    assert result["success"] is True, result
    assert len(result["pages"]) == 1
    assert sandbox.calls == []
