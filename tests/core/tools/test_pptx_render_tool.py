"""Tests for rendering a workspace .pptx into one PNG per slide.

The conversion subprocess and (for most cases) the PDF raster step are mocked
so the selection, pixel budget, registration and cleanup contracts can be
checked deterministically. ``test_render_two_page_deck_with_real_soffice``
covers the real LibreOffice + PyMuPDF path and only skips when LibreOffice is
genuinely absent from this host's PATH.
"""

import asyncio
import base64
import hashlib
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import xagent.core.storage.manager as storage_manager
import xagent.web.models.database as database_module
from xagent.core.file_storage.factory import get_unscoped_file_storage
from xagent.core.workspace import TaskWorkspace
from xagent.web.models import Base
from xagent.web.models.task import Task
from xagent.web.models.uploaded_file import UploadedFile
from xagent.web.models.user import User

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO+aXioAAAAASUVORK5CYII="
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_render_module():
    try:
        from xagent.core.tools.core import pptx_render
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"pptx_render module is missing: {exc}")
    return pptx_render


def _load_conversion_module():
    try:
        from xagent.core.tools.core import pptx_conversion
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"pptx_conversion module is missing: {exc}")
    return pptx_conversion


@pytest.fixture
def durable_workspace(monkeypatch, tmp_path):
    """A real task-owned workspace backed by an isolated sqlite database."""
    uploads = tmp_path / "uploads"
    objects = tmp_path / "objects"
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(uploads))
    monkeypatch.setenv("XAGENT_FILE_STORAGE_URI", objects.as_uri())
    monkeypatch.setenv("XAGENT_STORAGE_ROOT", str(storage_root))
    get_unscoped_file_storage.cache_clear()

    engine = create_engine(
        f"sqlite:///{tmp_path / 'metadata.db'}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    with session_factory() as db:
        user = User(username="render-owner", password_hash="hash")
        db.add(user)
        db.flush()
        db.add(Task(id=4242, user_id=user.id, title="render test task"))
        db.commit()
        user_pk = int(user.id)
    monkeypatch.setattr(database_module, "_SessionLocal", session_factory)
    monkeypatch.setattr(storage_manager, "create_db_session", session_factory)
    workspace = TaskWorkspace(
        id="web_task_4242",
        base_dir=str(uploads / f"user_{user_pk}"),
    )
    yield SimpleNamespace(
        workspace=workspace,
        session_factory=session_factory,
        base_dir=uploads / f"user_{user_pk}",
        storage_root=storage_root,
    )
    engine.dispose()
    get_unscoped_file_storage.cache_clear()


def _write_deck(directory: pathlib.Path, name: str = "deck.pptx") -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"PK\x03\x04 synthetic presentation bytes")
    return path


def _arg_value(arguments, flag):
    return arguments[arguments.index(flag) + 1]


class _FakeMatrix:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.a = x
        self.b = y

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _FakeMatrix) and (self.x, self.y) == (other.x, other.y)


class _FakePixelExtent:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def save(self, path, output=None) -> None:
        if hasattr(path, "write"):
            path.write(PNG_1X1)
        else:
            pathlib.Path(path).write_bytes(PNG_1X1)

    def pil_save(self, stream, **kwargs) -> None:
        self.save(stream)


class _FakeRect:
    """Rect double that mirrors PyMuPDF's matrix/irect round trip."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    def __mul__(self, matrix):
        width = max(1, math.ceil(self.width * matrix.a - 1e-9))
        height = max(1, math.ceil(self.height * matrix.b - 1e-9))
        return SimpleNamespace(irect=SimpleNamespace(width=width, height=height))

    __rmul__ = __mul__


class _FakePage:
    def __init__(self, width: float, height: float, events: list, number: int):
        self.rect = _FakeRect(width, height)
        self._events = events
        self._number = number

    def get_pixmap(self, *, matrix, alpha):
        self._events.append(("get_pixmap", self._number, matrix))
        width = max(1, math.ceil(self.rect.width * matrix.a - 1e-9))
        height = max(1, math.ceil(self.rect.height * matrix.b - 1e-9))
        return _FakePixelExtent(width, height)


class _FakeDocument:
    def __init__(self, pages: list[_FakePage]):
        self._pages = pages
        self.page_count = len(pages)
        self.closed = False

    def __getitem__(self, index: int) -> _FakePage:
        return self._pages[index]

    def close(self) -> None:
        self.closed = True


class _FakeFitz:
    def __init__(self, pages: list[tuple[float, float]], events: list):
        self.pages = [
            _FakePage(width, height, events, number)
            for number, (width, height) in enumerate(pages, start=1)
        ]
        self.events = events
        self.documents: list[_FakeDocument] = []

    def open(self, _path):
        document = _FakeDocument(self.pages)
        self.documents.append(document)
        return document

    @staticmethod
    def Matrix(x: float, y: float) -> _FakeMatrix:  # noqa: N802 - mirrors PyMuPDF
        return _FakeMatrix(x, y)


def _install_conversion(monkeypatch, render, *, calls, timeout_records):
    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        calls.append(pathlib.Path(pptx_path))
        timeout_records.append(timeout_seconds)
        pdf_path = output_dir / (pathlib.Path(pptx_path).stem + ".pdf")
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")
        return pdf_path

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)
    return calls


def _install_rasterizer(monkeypatch, render, *, page_count, requested, rendered):
    from xagent.core.tools.core.pptx_render import RenderedPage

    def render_pdf_pages_to_pngs(
        pdf_path, page_numbers, output_dir, *, scale=2.0, max_pixels=4_000_000
    ):
        requested.append(list(page_numbers))
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pages = []
        for number in page_numbers:
            path = output_dir / f"slide_{number:03d}.png"
            path.write_bytes(PNG_1X1)
            pages.append(RenderedPage(page_number=number, path=path, width=1, height=1))
        rendered.append(pages)
        return page_count, pages

    monkeypatch.setattr(render, "render_pdf_pages_to_pngs", render_pdf_pages_to_pngs)
    monkeypatch.setattr(render, "read_pdf_page_count", lambda _path: page_count)


def _output_rows(session_factory):
    with session_factory() as db:
        return list(db.query(UploadedFile).order_by(UploadedFile.filename).all())


def _register_input_deck(workspace, deck_path):
    return workspace.register_files([(str(deck_path), None)])[0]


def test_compute_page_geometry_keeps_scale_within_the_budget():
    render = _load_render_module()

    geometry = render.compute_page_geometry(720.0, 540.0, scale=2.0)

    assert (geometry.width, geometry.height) == (1440, 1080)
    assert geometry.scale == 2.0
    assert geometry.width * geometry.height < render.MAX_PAGE_PIXELS


def test_compute_page_geometry_allows_exactly_the_pixel_budget():
    render = _load_render_module()

    geometry = render.compute_page_geometry(
        2000.0, 2000.0, scale=1.0, max_pixels=render.MAX_PAGE_PIXELS
    )

    assert (geometry.width, geometry.height) == (2000, 2000)
    assert geometry.scale == 1.0
    assert geometry.width * geometry.height == render.MAX_PAGE_PIXELS


def test_compute_page_geometry_clamps_one_pixel_over_the_budget():
    render = _load_render_module()

    geometry = render.compute_page_geometry(
        2001.0, 2000.0, scale=1.0, max_pixels=render.MAX_PAGE_PIXELS
    )

    assert geometry.scale < 1.0
    assert geometry.width * geometry.height <= render.MAX_PAGE_PIXELS
    assert geometry.width == 2000
    # Rounding must not distort the aspect ratio beyond one pixel per side.
    assert abs(geometry.width / geometry.height - 2001.0 / 2000.0) < 0.002


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1_000_000.0, 1_000_000.0),
        (1_000_000.0, 1.0),
        (1.0, 1_000_000.0),
        (1.0e6, 1.0e6),
    ],
)
def test_compute_page_geometry_clamps_huge_and_extreme_pages(width, height):
    render = _load_render_module()

    geometry = render.compute_page_geometry(
        width, height, scale=2.0, max_pixels=render.MAX_PAGE_PIXELS
    )

    assert geometry.width >= 1
    assert geometry.height >= 1
    assert geometry.width * geometry.height <= render.MAX_PAGE_PIXELS
    assert geometry.scale > 0


@pytest.mark.parametrize(
    ("width", "height", "scale"),
    [
        (0.0, 540.0, 2.0),
        (-1.0, 540.0, 2.0),
        (720.0, float("nan"), 2.0),
        (720.0, float("inf"), 2.0),
        (720.0, 540.0, 0.0),
        (720.0, 540.0, -2.0),
        (720.0, 540.0, float("nan")),
        (float("inf"), 540.0, 2.0),
    ],
)
def test_compute_page_geometry_rejects_invalid_dimensions(width, height, scale):
    render = _load_render_module()

    with pytest.raises(render.PptxRenderError):
        render.compute_page_geometry(width, height, scale=scale)


def test_resolve_page_selection_defaults_to_the_first_twenty():
    render = _load_render_module()

    selection = render.resolve_page_selection(25, None, None)

    assert list(selection.page_numbers) == list(range(1, 21))
    assert selection.truncated is True
    assert selection.total_pages == 25

    small = render.resolve_page_selection(2, None, None)
    assert list(small.page_numbers) == [1, 2]
    assert small.truncated is False


def test_resolve_page_selection_honours_explicit_ranges():
    render = _load_render_module()

    assert list(render.resolve_page_selection(21, 1, 1).page_numbers) == [1]
    assert list(render.resolve_page_selection(21, 3, 5).page_numbers) == [3, 4, 5]
    assert list(render.resolve_page_selection(21, 21, 21).page_numbers) == [21]
    assert list(render.resolve_page_selection(21, 20, None).page_numbers) == [
        20,
        21,
    ]
    assert list(render.resolve_page_selection(21, None, 2).page_numbers) == [1, 2]


@pytest.mark.parametrize(
    ("start_page", "end_page"),
    [
        (0, None),
        (-1, None),
        (5, 3),
        (1, 21),
        (2, 22),
        (True, None),
        (1.0, None),
        ("1", None),
        (None, "2"),
        ([1], None),
        ({"page": 1}, None),
    ],
)
def test_page_arguments_reject_invalid_and_mixed_types(start_page, end_page):
    render = _load_render_module()

    with pytest.raises(render.PptxRenderError):
        render.validate_page_arguments(start_page, end_page)


def test_page_arguments_reject_a_range_over_the_page_limit():
    render = _load_render_module()

    with pytest.raises(render.PptxRenderError) as fault:
        render.validate_page_arguments(1, 21)

    assert str(render.MAX_RENDERED_PAGES) in str(fault.value)


def test_resolve_page_selection_rejects_out_of_bounds_ranges():
    render = _load_render_module()

    with pytest.raises(render.PptxRenderError):
        render.resolve_page_selection(5, 4, 6)
    with pytest.raises(render.PptxRenderError):
        render.resolve_page_selection(5, 6, 6)
    with pytest.raises(render.PptxRenderError):
        render.resolve_page_selection(0, None, None)


def test_unrendered_page_ranges_report_skipped_pages():
    render = _load_render_module()

    assert render.unrendered_page_ranges(25, list(range(1, 21))) == [[21, 25]]
    assert render.unrendered_page_ranges(10, [3, 4, 5]) == [[1, 2], [6, 10]]
    assert render.unrendered_page_ranges(5, [1, 2, 3, 4, 5]) == []


@pytest.mark.asyncio
async def test_render_default_renders_every_page_of_a_small_deck(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    conversion_calls: list[pathlib.Path] = []
    timeouts: list[float] = []
    requested: list[list[int]] = []
    rendered: list[list] = []
    _install_conversion(
        monkeypatch,
        render,
        calls=conversion_calls,
        timeout_records=timeouts,
    )
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=requested, rendered=rendered
    )

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    assert result["total_pages"] == 2
    assert result["page_numbers"] == [1, 2]
    assert result["rendered_pages"] == 2
    assert result["truncated"] is False
    assert result["unrendered_pages"] == 0
    assert result["unrendered_page_ranges"] == []
    assert result["page_limit"] == render.MAX_RENDERED_PAGES
    assert result["max_pixels_per_page"] == render.MAX_PAGE_PIXELS
    assert result["conversion_timeout_seconds"] == 60
    assert requested == [[1, 2]]
    assert [page["page_number"] for page in result["pages"]] == [1, 2]
    for page in result["pages"]:
        assert page["file_ref"]["mime_type"] == "image/png"
        assert page["file_ref"]["filename"] == f"slide_{page['page_number']:03d}.png"
        assert page["file_ref"]["file_id"]
        assert page["artifacts"] == [
            {
                "type": "image",
                "file_id": page["file_ref"]["file_id"],
                "filename": page["file_ref"]["filename"],
                "mime_type": "image/png",
                "display": "inline",
            }
        ]
    assert len(result["artifacts"]) == 2
    assert timeouts == [60.0]


@pytest.mark.asyncio
async def test_render_default_caps_at_twenty_pages_without_rasterizing_more(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    requested: list[list[int]] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=25, requested=requested, rendered=[]
    )

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    assert requested == [list(range(1, 21))]
    assert result["page_limit"] == 20
    assert result["rendered_pages"] == 20
    assert result["truncated"] is True
    assert result["total_pages"] == 25
    assert result["unrendered_pages"] == 5
    assert result["unrendered_page_ranges"] == [[21, 25]]
    assert len(_output_rows(durable_workspace.session_factory)) == 20


@pytest.mark.asyncio
async def test_render_rejects_an_explicit_range_over_the_page_limit_before_converting(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    conversions: list[pathlib.Path] = []
    _install_conversion(monkeypatch, render, calls=conversions, timeout_records=[])
    requested: list[list[int]] = []
    _install_rasterizer(
        monkeypatch, render, page_count=30, requested=requested, rendered=[]
    )

    result = await render.render_pptx_to_images(
        str(deck), durable_workspace.workspace, start_page=1, end_page=21
    )

    assert result["success"] is False
    assert result["error_code"] == "invalid_argument"
    assert "20" in result["error"]
    assert conversions == []
    assert requested == []
    assert not list(durable_workspace.workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_honours_an_explicit_page_range(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    requested: list[list[int]] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=10, requested=requested, rendered=[]
    )

    result = await render.render_pptx_to_images(
        str(deck), durable_workspace.workspace, start_page=3, end_page=5
    )

    assert result["success"] is True
    assert requested == [[3, 4, 5]]
    assert result["page_numbers"] == [3, 4, 5]
    assert result["truncated"] is False
    assert result["unrendered_pages"] == 7
    assert result["unrendered_page_ranges"] == [[1, 2], [6, 10]]
    assert [page["file_ref"]["filename"] for page in result["pages"]] == [
        "slide_003.png",
        "slide_004.png",
        "slide_005.png",
    ]


@pytest.mark.asyncio
async def test_render_rejects_an_explicit_range_beyond_the_deck(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    requested: list[list[int]] = []
    _install_rasterizer(
        monkeypatch, render, page_count=5, requested=requested, rendered=[]
    )

    result = await render.render_pptx_to_images(
        str(deck), durable_workspace.workspace, start_page=4, end_page=6
    )

    assert result["success"] is False
    assert result["error_code"] == "page_range_invalid"
    assert "6" in result["error"]
    assert requested == []
    assert not list(durable_workspace.workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_reports_an_empty_presentation(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    requested: list[list[int]] = []
    _install_rasterizer(
        monkeypatch, render, page_count=0, requested=requested, rendered=[]
    )

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "empty_presentation"
    assert requested == []
    assert not list(durable_workspace.workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_page_limit_is_applied_before_any_pixmap_allocation(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    events: list = []
    fake_fitz = _FakeFitz([(720.0, 540.0)] * 25, events)
    monkeypatch.setattr(render, "fitz", fake_fitz)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    assert result["rendered_pages"] == 20
    pixmap_pages = [event[1] for event in events if event[0] == "get_pixmap"]
    assert pixmap_pages == list(range(1, 21))
    assert len(_output_rows(durable_workspace.session_factory)) == 20


@pytest.mark.asyncio
async def test_render_limits_pixels_before_requesting_a_huge_pixmap(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    events: list = []
    fake_fitz = _FakeFitz([(1_000_000.0, 1_000_000.0)], events)
    monkeypatch.setattr(render, "fitz", fake_fitz)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    pixmap_events = [event for event in events if event[0] == "get_pixmap"]
    assert len(pixmap_events) == 1
    matrix = pixmap_events[0][2]
    assert matrix.a == matrix.b
    predicted_width = math.ceil(1_000_000.0 * matrix.a - 1e-9)
    predicted_height = math.ceil(1_000_000.0 * matrix.b - 1e-9)
    assert predicted_width * predicted_height <= render.MAX_PAGE_PIXELS
    assert matrix.a < 1.0


def test_real_rasterizer_stays_within_the_pixel_budget(tmp_path):
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        page = document.new_page(width=720, height=540)
        page.insert_text((72, 72), "budget")
        pdf_path = tmp_path / "budget.pdf"
        document.save(str(pdf_path))
    finally:
        document.close()
    output_dir = tmp_path / "pngs"
    output_dir.mkdir()

    total_pages, pages = render.render_pdf_pages_to_pngs(
        pdf_path, [1], output_dir, scale=100.0
    )

    assert total_pages == 1
    assert len(pages) == 1
    rendered_page = pages[0]
    assert rendered_page.page_number == 1
    assert rendered_page.path.read_bytes().startswith(PNG_MAGIC)
    assert rendered_page.width * rendered_page.height <= render.MAX_PAGE_PIXELS
    assert rendered_page.width > 720
    assert abs(rendered_page.width / rendered_page.height - 720 / 540) < 0.01


def test_real_rasterizer_renders_only_the_selected_pages(tmp_path):
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        for number in range(5):
            page = document.new_page(width=720, height=540)
            page.insert_text((72, 72), f"slide {number + 1}")
        pdf_path = tmp_path / "deck.pdf"
        document.save(str(pdf_path))
    finally:
        document.close()
    output_dir = tmp_path / "pngs"
    output_dir.mkdir()

    total_pages, pages = render.render_pdf_pages_to_pngs(pdf_path, [2, 4], output_dir)

    assert total_pages == 5
    assert [page.page_number for page in pages] == [2, 4]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "slide_002.png",
        "slide_004.png",
    ]
    for page in pages:
        data = page.path.read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert page.width > 0 and page.height > 0


@pytest.mark.asyncio
async def test_render_each_page_is_durably_registered_and_resolvable(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=2, requested=[], rendered=[])

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    rows = _output_rows(durable_workspace.session_factory)
    assert [row.filename for row in rows] == ["slide_001.png", "slide_002.png"]
    assert {row.mime_type for row in rows} == {"image/png"}
    assert {row.task_id for row in rows} == {4242}
    assert not any(str(row.filename).endswith(".pdf") for row in rows)
    fresh_workspace = TaskWorkspace(
        id="web_task_4242", base_dir=str(durable_workspace.base_dir)
    )
    for page in result["pages"]:
        file_id = page["file_ref"]["file_id"]
        resolved = fresh_workspace.resolve_file_id_detached(file_id)
        assert resolved is not None
        assert resolved.is_file()
        assert resolved.name == page["file_ref"]["filename"]
        assert page["file_ref"]["preview_url"] == f"/api/files/preview/{file_id}"
        assert page["file_ref"]["markdown_link"] is not None
        assert f"file:{file_id}" in page["file_ref"]["markdown_link"]
        assert page["artifacts"][0]["file_id"] == file_id


@pytest.mark.asyncio
async def test_render_requires_a_persistent_task_owner(monkeypatch, tmp_path):
    render = _load_render_module()
    deck = _write_deck(tmp_path)
    workspace = TaskWorkspace(id="scratch_workspace", base_dir=str(tmp_path / "ws"))
    conversions: list[pathlib.Path] = []
    _install_conversion(monkeypatch, render, calls=conversions, timeout_records=[])

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "durable_registration_unavailable"
    assert "persistent" in result["error"] or "task owner" in result["error"]
    assert conversions == []
    assert not list(workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_requires_a_workspace(monkeypatch, tmp_path):
    render = _load_render_module()
    deck = _write_deck(tmp_path)

    result = await render.render_pptx_to_images(str(deck), None)

    assert result["success"] is False
    assert result["error_code"] == "workspace_unavailable"


@pytest.mark.asyncio
async def test_render_accepts_persistent_file_ids_and_allowed_paths(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    deck_bytes = deck.read_bytes()
    file_id = _register_input_deck(durable_workspace.workspace, deck)
    conversions: list[pathlib.Path] = []
    _install_conversion(monkeypatch, render, calls=conversions, timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])
    selectors = [
        f"file:{file_id}",
        file_id,
        "input/deck.pptx",
        str(deck),
    ]

    for selector in selectors:
        result = await render.render_pptx_to_images(
            selector, durable_workspace.workspace
        )
        assert result["success"] is True, (selector, result)
        assert pathlib.Path(result["pptx_path"]).resolve() == deck.resolve()

    assert {path.resolve() for path in conversions} == {deck.resolve()}
    assert (
        hashlib.sha256(deck.read_bytes()).hexdigest()
        == hashlib.sha256(deck_bytes).hexdigest()
    )


@pytest.mark.asyncio
async def test_render_rejects_non_pptx_missing_outside_and_symlink_inputs(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    conversions: list[pathlib.Path] = []
    _install_conversion(monkeypatch, render, calls=conversions, timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])
    workspace = durable_workspace.workspace
    not_pptx = workspace.input_dir / "notes.txt"
    not_pptx.write_bytes(b"not a presentation")
    directory_input = workspace.input_dir / "folder.pptx"
    directory_input.mkdir()
    missing = workspace.input_dir / "missing.pptx"
    outside = tmp_path / "outside" / "outside.pptx"
    outside.parent.mkdir()
    outside.write_bytes(b"PK\x03\x04 outside")
    escape = workspace.input_dir / "escape.pptx"
    escape.symlink_to(outside)

    cases = [
        (str(not_pptx), "unsupported_format"),
        (str(directory_input), "input_not_a_file"),
        (str(missing), "input_not_found"),
        (str(outside), "input_outside_workspace"),
        (str(escape), "input_outside_workspace"),
        ("file:00000000-0000-0000-0000-000000000000", "input_not_found"),
    ]
    for selector, expected_code in cases:
        result = await render.render_pptx_to_images(selector, workspace)
        assert result["success"] is False, selector
        assert result["error_code"] == expected_code, (selector, result)

    assert conversions == []
    assert not list(workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_accepts_an_uppercase_pptx_suffix(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    _write_deck(durable_workspace.workspace.input_dir, "DECK.PPTX")
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])

    result = await render.render_pptx_to_images(
        "input/DECK.PPTX", durable_workspace.workspace
    )

    assert result["success"] is True


@pytest.mark.asyncio
async def test_render_rejects_mixed_argument_types_without_converting(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    conversions: list[pathlib.Path] = []
    _install_conversion(monkeypatch, render, calls=conversions, timeout_records=[])

    for kwargs in (
        {"pptx_path": 3},
        {"pptx_path": None},
        {"pptx_path": [str(deck)]},
        {"pptx_path": str(deck), "start_page": True},
        {"pptx_path": str(deck), "start_page": 1.5},
        {"pptx_path": str(deck), "end_page": "2"},
        {"pptx_path": str(deck), "end_page": [2]},
    ):
        result = await render.render_pptx_to_images(
            kwargs.pop("pptx_path"), durable_workspace.workspace, **kwargs
        )
        assert result["success"] is False, kwargs
        assert result["error_code"] == "invalid_argument", kwargs

    assert conversions == []


@pytest.mark.asyncio
async def test_render_passes_the_fixed_timeout_to_the_converter(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    conversion = _load_conversion_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    timeouts: list[float] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=timeouts)
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    assert timeouts == [conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS]


@pytest.mark.asyncio
async def test_render_cancellation_reclaims_the_subprocess_and_cleans_up(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    started = asyncio.Event()
    seen: dict[str, pathlib.Path] = {}

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        seen["temp_root"] = pathlib.Path(output_dir).parent
        seen["pdf_dir"] = pathlib.Path(output_dir)
        started.set()
        await asyncio.sleep(60)
        raise AssertionError("conversion must be cancelled")

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)

    task = asyncio.create_task(
        render.render_pptx_to_images(str(deck), durable_workspace.workspace)
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    output_dirs = list(durable_workspace.workspace.output_dir.iterdir())
    assert len(output_dirs) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not seen["pdf_dir"].exists()
    assert not seen["temp_root"].exists()
    assert not output_dirs[0].exists()
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_concurrent_renders_use_isolated_directories(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    first = _write_deck(durable_workspace.workspace.input_dir, "first.pptx")
    second = _write_deck(durable_workspace.workspace.input_dir, "second.pptx")
    seen: list[dict] = []
    release = asyncio.Event()
    arrived = asyncio.Event()
    in_flight = 0

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        nonlocal in_flight
        in_flight += 1
        if in_flight == 2:
            arrived.set()
        seen.append(
            {
                "source": pathlib.Path(pptx_path),
                "output_dir": pathlib.Path(output_dir),
                "temp_root": pathlib.Path(output_dir).parent,
            }
        )
        await release.wait()
        pdf_path = pathlib.Path(output_dir) / (pathlib.Path(pptx_path).stem + ".pdf")
        pdf_path.write_bytes(b"%PDF-1.4 synthetic")
        return pdf_path

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])

    both = asyncio.gather(
        render.render_pptx_to_images(str(first), durable_workspace.workspace),
        render.render_pptx_to_images(str(second), durable_workspace.workspace),
    )
    await asyncio.wait_for(arrived.wait(), timeout=5)
    release.set()
    results = await asyncio.wait_for(both, timeout=10)

    assert [result["success"] for result in results] == [True, True]
    directories = [pathlib.Path(result["output_dir"]) for result in results]
    pdf_directories = [entry["output_dir"] for entry in seen]
    temp_roots = [entry["temp_root"] for entry in seen]
    assert len({str(path) for path in directories}) == 2
    assert len({str(path) for path in pdf_directories}) == 2
    assert len({str(path) for path in temp_roots}) == 2
    for path in directories:
        assert path.is_dir()
        assert (path / "slide_001.png").is_file()
    for path in temp_roots:
        assert not path.exists()
    assert len(_output_rows(durable_workspace.session_factory)) == 2


@pytest.mark.asyncio
async def test_render_fails_clearly_when_libreoffice_is_missing(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    conversion = _load_conversion_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        raise conversion.PptxConverterUnavailableError(
            "LibreOffice is not installed or not on PATH"
        )

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "pdf_conversion_unavailable"
    assert "LibreOffice" in result["error"]
    assert not list(durable_workspace.workspace.output_dir.iterdir())
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_render_reports_conversion_timeout_and_cleans_up(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    conversion = _load_conversion_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        raise conversion.PptxConversionTimeoutError(
            "LibreOffice conversion timed out after 60 seconds"
        )

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "pdf_conversion_timeout"
    assert not list(durable_workspace.workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_reports_conversion_failure_and_cleans_up(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    conversion = _load_conversion_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        raise conversion.PptxConversionFailedError("soffice exited 1")

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "pdf_conversion_failed"
    assert not list(durable_workspace.workspace.output_dir.iterdir())


@pytest.mark.asyncio
async def test_render_reports_a_bad_pdf_and_cleans_up(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)

    async def convert_pptx_to_pdf(
        pptx_path, output_dir, *, timeout_seconds=60.0, include_hidden_slides=False
    ):
        pdf_path = pathlib.Path(output_dir) / (pathlib.Path(pptx_path).stem + ".pdf")
        pdf_path.write_bytes(b"this is not a pdf")
        return pdf_path

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert_pptx_to_pdf)

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] in {"pdf_unreadable", "raster_failed"}
    assert not list(durable_workspace.workspace.output_dir.iterdir())
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_render_cleans_up_when_rasterization_fails_midway(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])

    def failing_raster(pdf_path, page_numbers, output_dir, *, scale=2.0, max_pixels=0):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "slide_001.png").write_bytes(PNG_1X1)
        raise render.PptxRenderRasterError("page 2 could not be rasterized")

    monkeypatch.setattr(render, "render_pdf_pages_to_pngs", failing_raster)
    monkeypatch.setattr(render, "read_pdf_page_count", lambda _path: 2)

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "raster_failed"
    assert not list(durable_workspace.workspace.output_dir.iterdir())
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_render_cleans_up_when_a_png_cannot_be_written(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    events: list = []
    fake_fitz = _FakeFitz([(720.0, 540.0)], events)

    class _UnwritableExtent(_FakePixelExtent):
        def save(self, path) -> None:
            raise OSError("no space left on device")

    fake_fitz.pages[0].get_pixmap = lambda *, matrix, alpha: _UnwritableExtent(
        1440, 1080
    )
    monkeypatch.setattr(render, "fitz", fake_fitz)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is False
    assert result["error_code"] == "raster_failed"
    assert not list(durable_workspace.workspace.output_dir.iterdir())
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_render_reports_registration_failure_and_removes_unregistered_pngs(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=3, requested=[], rendered=[])
    workspace = durable_workspace.workspace

    def failing_registration(*_args, **_kwargs):
        raise RuntimeError("metadata database unavailable")

    monkeypatch.setattr(workspace, "register_files", failing_registration)

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert "metadata database unavailable" in result["error"]
    assert not list(workspace.output_dir.iterdir())
    assert _output_rows(durable_workspace.session_factory) == []


@pytest.mark.asyncio
async def test_render_keeps_already_registered_pngs_when_registration_fails_partway(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=3, requested=[], rendered=[])
    workspace = durable_workspace.workspace
    real_register = workspace.register_files
    calls = {"count": 0}

    def partial_registration(files, *, db_session=None):
        calls["count"] += 1
        real_register(files[:1], db_session=db_session)
        raise RuntimeError("registration interrupted")

    monkeypatch.setattr(workspace, "register_files", partial_registration)

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert result["registered_pages"] == [1]
    output_dirs = [path for path in workspace.output_dir.iterdir() if path.is_dir()]
    assert len(output_dirs) == 1
    remaining = sorted(path.name for path in output_dirs[0].iterdir())
    assert remaining == ["slide_001.png"]
    rows = _output_rows(durable_workspace.session_factory)
    assert [row.filename for row in rows] == ["slide_001.png"]


@pytest.mark.asyncio
async def test_render_two_page_deck_with_real_soffice(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    if shutil.which("soffice") is None:
        pytest.skip("LibreOffice (soffice) is not installed on this host")

    try:
        import pptx
    except (
        ImportError
    ) as exc:  # pragma: no cover - dependency is part of the backend image
        pytest.fail(
            "LibreOffice is available but python-pptx is not installed, so the "
            f"real conversion cannot run: {exc}"
        )
    deck_path = durable_workspace.workspace.input_dir / "synthetic-two-slides.pptx"
    presentation = pptx.Presentation()
    for index in range(2):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        textbox = slide.shapes.add_textbox(72, 72, 400000, 800000)
        textbox.text_frame.text = f"synthetic slide {index + 1}"
    presentation.save(str(deck_path))
    deck_hash = hashlib.sha256(deck_path.read_bytes()).hexdigest()

    result = await render.render_pptx_to_images(
        str(deck_path), durable_workspace.workspace
    )

    assert result["success"] is True, result
    assert result["total_pages"] == 2
    assert result["page_numbers"] == [1, 2]
    assert len(result["pages"]) == 2
    for page in result["pages"]:
        path = pathlib.Path(page["file_ref"]["file_path"])
        assert path.read_bytes().startswith(PNG_MAGIC)
        assert page["width"] * page["height"] <= render.MAX_PAGE_PIXELS
        assert page["width"] > 0 and page["height"] > 0
        resolved = durable_workspace.workspace.resolve_file_id_detached(
            page["file_ref"]["file_id"]
        )
        assert resolved is not None
        assert resolved.read_bytes().startswith(PNG_MAGIC)
    image_hashes = [
        hashlib.sha256(
            pathlib.Path(page["file_ref"]["file_path"]).read_bytes()
        ).hexdigest()
        for page in result["pages"]
    ]
    assert len(set(image_hashes)) == 2, (
        "the two synthetic slides must render to different images"
    )
    assert [
        pathlib.Path(page["file_ref"]["file_path"]).name for page in result["pages"]
    ] == [
        "slide_001.png",
        "slide_002.png",
    ]
    assert hashlib.sha256(deck_path.read_bytes()).hexdigest() == deck_hash
    rows = _output_rows(durable_workspace.session_factory)
    assert len(rows) == 2
    assert not list(durable_workspace.storage_root.rglob("*.pdf"))
    assert not list(pathlib.Path(tempfile.gettempdir()).glob("xagent-pptx-profile-*"))


@pytest.mark.parametrize(
    ("width", "height"),
    [(1430.0, 1670.0), (1810.0, 1080.0), (720.0, 540.0)],
)
def test_compute_page_geometry_keeps_common_page_sizes_within_budget(width, height):
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")

    geometry = render.compute_page_geometry(width, height)

    assert geometry.scale > 0
    assert geometry.width * geometry.height <= render.MAX_PAGE_PIXELS
    # The predicted with-MuPDF dimensions must not exceed the budget either.
    actual = (
        fitz.Rect(0.0, 0.0, width, height) * fitz.Matrix(geometry.scale, geometry.scale)
    ).irect
    assert actual.width * actual.height <= render.MAX_PAGE_PIXELS
    assert geometry.width >= actual.width
    assert geometry.height >= actual.height
    assert abs(geometry.width / geometry.height - width / height) < 0.01


def test_compute_page_geometry_handles_exactly_one_pixel_over_budget():
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")

    # 41 * 97,561 = 4,000,001: one pixel over the page budget, so the scale
    # must be lowered instead of rejecting the page or exceeding the budget.
    geometry = render.compute_page_geometry(41.0, 97561.0, scale=1.0)

    assert geometry.scale < 1.0
    assert geometry.width * geometry.height <= render.MAX_PAGE_PIXELS
    actual = (
        fitz.Rect(0.0, 0.0, 41.0, 97561.0) * fitz.Matrix(geometry.scale, geometry.scale)
    ).irect
    assert actual.width * actual.height <= render.MAX_PAGE_PIXELS


def test_compute_page_geometry_stays_within_budget_for_mupdf_rounding():
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")
    import random

    rng = random.Random(20261027)
    cases = [
        (1430.0, 1670.0),
        (1810.0, 1080.0),
        (41.0, 97561.0),
        (1_000_000.0, 1.0),
        (1_000_000.0, 1_000_000.0),
    ]
    cases.extend(
        (rng.uniform(200.0, 6000.0), rng.uniform(200.0, 6000.0)) for _ in range(200)
    )
    for width, height in cases:
        geometry = render.compute_page_geometry(width, height)
        rectangle = fitz.Rect(0.0, 0.0, width, height)
        actual = (rectangle * fitz.Matrix(geometry.scale, geometry.scale)).irect
        assert actual.width * actual.height <= render.MAX_PAGE_PIXELS, (
            width,
            height,
            geometry,
            actual,
        )


def _write_pdf_page(tmp_path, width: float, height: float, name: str):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    try:
        page = document.new_page(width=width, height=height)
        page.insert_text((72, 72), f"{width:g}x{height:g}")
        pdf_path = tmp_path / name
        document.save(str(pdf_path))
    finally:
        document.close()
    return pdf_path


def test_real_rasterizer_default_scale_renders_tall_page(tmp_path):
    render = _load_render_module()
    pdf_path = _write_pdf_page(tmp_path, 1430.0, 1670.0, "tall.pdf")
    output_dir = tmp_path / "pngs"
    output_dir.mkdir()

    total_pages, pages = render.render_pdf_pages_to_pngs(pdf_path, [1], output_dir)

    assert total_pages == 1
    assert len(pages) == 1
    assert pages[0].path.read_bytes().startswith(PNG_MAGIC)
    assert pages[0].width * pages[0].height <= render.MAX_PAGE_PIXELS
    assert abs(pages[0].width / pages[0].height - 1430 / 1670) < 0.01


def test_real_rasterizer_default_scale_renders_wide_page(tmp_path):
    render = _load_render_module()
    pdf_path = _write_pdf_page(tmp_path, 1810.0, 1080.0, "wide.pdf")
    output_dir = tmp_path / "pngs"
    output_dir.mkdir()

    total_pages, pages = render.render_pdf_pages_to_pngs(pdf_path, [1], output_dir)

    assert total_pages == 1
    assert len(pages) == 1
    assert pages[0].path.read_bytes().startswith(PNG_MAGIC)
    assert pages[0].width * pages[0].height <= render.MAX_PAGE_PIXELS
    assert abs(pages[0].width / pages[0].height - 1810 / 1080) < 0.01


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("total_pages", "rendered_count", "truncated", "unrendered"),
    [(20, 20, False, 0), (21, 20, True, 1)],
)
async def test_render_default_page_boundaries(
    durable_workspace,
    monkeypatch,
    tmp_path,
    total_pages,
    rendered_count,
    truncated,
    unrendered,
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir)
    requested: list[list[int]] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch,
        render,
        page_count=total_pages,
        requested=requested,
        rendered=[],
    )

    result = await render.render_pptx_to_images(str(deck), durable_workspace.workspace)

    assert result["success"] is True
    assert result["total_pages"] == total_pages
    assert result["rendered_pages"] == rendered_count
    assert result["truncated"] is truncated
    assert result["unrendered_pages"] == unrendered
    assert requested == [list(range(1, rendered_count + 1))]
    if truncated:
        assert result["unrendered_page_ranges"] == [[21, 21]]
    else:
        assert result["unrendered_page_ranges"] == []


@pytest.mark.parametrize(
    ("start_page", "end_page"),
    [
        (True, 2),
        (2, True),
        (1.5, 2),
        (2, 1.5),
        ("1", 2),
        (2, "3"),
        ([1], 2),
        (2, [3]),
        ({"page": 1}, 2),
        (2, {"page": 3}),
        (1, 3.0),
    ],
)
def test_page_arguments_reject_mixed_numeric_and_bad_types(start_page, end_page):
    render = _load_render_module()

    with pytest.raises(render.PptxRenderArgumentError):
        render.validate_page_arguments(start_page, end_page)


def test_one_sided_explicit_ranges_cannot_escape_the_page_limit():
    render = _load_render_module()

    with pytest.raises(render.PptxRenderArgumentError):
        render.resolve_page_selection(21, 1, None)
    with pytest.raises(render.PptxRenderArgumentError):
        render.resolve_page_selection(21, None, 21)
    assert list(render.resolve_page_selection(21, 21, None).page_numbers) == [21]
    assert list(render.resolve_page_selection(21, None, 20).page_numbers) == list(
        range(1, 21)
    )


@pytest.mark.asyncio
async def test_cancel_during_render_temp_directory_creation_removes_it(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    deck = _write_deck(durable_workspace.workspace.input_dir, "cancel.pptx")
    real_mkdtemp = tempfile.mkdtemp
    started = threading.Event()
    release = threading.Event()
    created: list[pathlib.Path] = []

    def delayed_mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") != "xagent-pptx-render-":
            return real_mkdtemp(*args, **kwargs)
        started.set()
        assert release.wait(5)
        kwargs["dir"] = str(tmp_path)
        path = pathlib.Path(real_mkdtemp(*args, **kwargs))
        created.append(path)
        return str(path)

    monkeypatch.setattr(tempfile, "mkdtemp", delayed_mkdtemp)
    task = asyncio.create_task(
        render.render_pptx_to_images(str(deck), durable_workspace.workspace)
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        release.set()
        await asyncio.sleep(0.05)
        assert created, "the controlled render temp root was never created"
        assert all(not path.exists() for path in created), (
            "cancellation during render temp-directory creation leaked the "
            "temporary root"
        )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        for path in created:
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_output_symlink_escape_is_refused_before_writing(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace.output_dir.rmdir()
    workspace.output_dir.symlink_to(outside, target_is_directory=True)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    requested: list[list[int]] = []
    rendered: list[list] = []
    _install_rasterizer(
        monkeypatch,
        render,
        page_count=2,
        requested=requested,
        rendered=rendered,
    )

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "output_unavailable"
    assert rendered == []
    assert requested == []
    assert list(outside.rglob("*")) == []


@pytest.mark.asyncio
async def test_preexisting_output_directory_is_not_deleted_on_collision(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    existing = workspace.output_dir / "pptx_render_000000000000"
    existing.mkdir()
    sentinel = existing / "existing.png"
    sentinel.write_bytes(b"pre-existing output")
    monkeypatch.setattr(
        render.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0" * 32),
    )
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=1, requested=[], rendered=[])

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert sentinel.exists(), (
        "a failed mkdir must not delete a directory this call never created"
    )
    assert sentinel.read_bytes() == b"pre-existing output"
    assert result["success"] is False
    assert result["error_code"] == "output_unavailable"


@pytest.mark.asyncio
async def test_postcommit_verification_outage_keeps_registered_pngs(
    durable_workspace, monkeypatch, tmp_path
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=2, requested=[], rendered=[])
    real_register = workspace.register_files
    committed: list[str] = []

    def register_then_break_verification(files, *, db_session=None):
        ids = real_register(files, db_session=db_session)
        committed.extend(ids)

        def failing_session():
            raise RuntimeError("synthetic post-commit verification outage")

        monkeypatch.setattr(storage_manager, "create_db_session", failing_session)
        return ids

    monkeypatch.setattr(workspace, "register_files", register_then_break_verification)

    result = await render.render_pptx_to_images(str(deck), workspace)

    rows = _output_rows(durable_workspace.session_factory)
    assert len(committed) == 2
    assert len(rows) == 2
    assert all(pathlib.Path(str(row.storage_path)).is_file() for row in rows), (
        "a verification outage must not delete already-registered PNGs"
    )
    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert result["registered_pages"] == [1, 2]
    assert result["unverified_pages"] == [1, 2]


@pytest.mark.asyncio
async def test_file_ref_build_runs_off_the_event_loop(durable_workspace, monkeypatch):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=2, requested=[], rendered=[])
    loop_thread = threading.get_ident()
    seen_threads: list[int] = []
    original = render.build_workspace_file_ref

    def observed_build(**kwargs):
        seen_threads.append(threading.get_ident())
        return original(**kwargs)

    monkeypatch.setattr(render, "build_workspace_file_ref", observed_build)

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is True, result
    assert len(seen_threads) == 2
    assert all(thread != loop_thread for thread in seen_threads), (
        "FileRef resolve/stat/is_file ran on the main event-loop thread"
    )


def test_fit_scale_to_page_lowers_the_scale_before_rasterizing(tmp_path):
    render = _load_render_module()
    fitz = pytest.importorskip("fitz")
    pdf_path = _write_pdf_page(tmp_path, 720.0, 540.0, "budget.pdf")
    document = fitz.open(str(pdf_path))
    try:
        page = document[0]
        requested_scale = 100.0
        effective_scale = render._fit_scale_to_page(
            page, requested_scale, render.MAX_PAGE_PIXELS
        )
        extents = render._mupdf_pixel_extents(page, effective_scale)
    finally:
        document.close()

    assert effective_scale < requested_scale
    assert extents[0] * extents[1] <= render.MAX_PAGE_PIXELS


@pytest.mark.asyncio
async def test_cancel_after_output_creation_does_not_leak(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    started = threading.Event()
    release = threading.Event()
    done = threading.Event()
    created: list[pathlib.Path] = []
    original_resolve = pathlib.Path.resolve

    def delayed_resolve(path, *args, **kwargs):
        if (
            path.name.startswith("pptx_render_")
            and path.parent == workspace.output_dir
            and args == (True,)
        ):
            created.append(path)
            started.set()
            assert release.wait(5)
            try:
                return original_resolve(path, *args, **kwargs)
            finally:
                done.set()
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", delayed_resolve)
    task = asyncio.create_task(render.render_pptx_to_images(str(deck), workspace))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        release.set()
        assert await asyncio.to_thread(done.wait, 5)
        await asyncio.sleep(0.05)
        leaked = [str(path) for path in created if path.exists()]
        assert leaked == [], (
            "cancellation after the exclusive output-directory creation lost "
            f"cleanup ownership: {leaked}"
        )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.to_thread(done.wait, 5)
        for path in created:
            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_output_validation_filesystem_calls_run_off_event_loop(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=2, requested=[], rendered=[])
    loop_thread = threading.get_ident()
    seen: list[tuple[str, int]] = []
    original_resolve = pathlib.Path.resolve
    original_is_dir = pathlib.Path.is_dir

    def observed_resolve(path, *args, **kwargs):
        if path == workspace.workspace_dir:
            seen.append(("resolve_workspace", threading.get_ident()))
        return original_resolve(path, *args, **kwargs)

    def observed_is_dir(path, *args, **kwargs):
        if path == workspace.output_dir:
            seen.append(("is_dir_output", threading.get_ident()))
        return original_is_dir(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", observed_resolve)
    monkeypatch.setattr(pathlib.Path, "is_dir", observed_is_dir)

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is True, result
    assert seen, "the output validation calls were not observed"
    assert all(thread != loop_thread for _name, thread in seen), (
        "output validation filesystem calls ran on the event loop"
    )


@pytest.mark.asyncio
async def test_delegated_workspace_registration_matches_persisted_record(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    delegated = TaskWorkspace(
        id="synthetic_delegated_render",
        base_dir=str(durable_workspace.base_dir),
        db_task_id=4242,
    )
    deck = _write_deck(delegated.input_dir)
    rendered: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=rendered
    )

    result = await render.render_pptx_to_images(str(deck), delegated)

    rows = _output_rows(durable_workspace.session_factory)
    originals = [page.path for batch in rendered for page in batch]
    assert rows and len(rows) == 2
    assert result["success"] is True, (
        "committed delegated outputs were classified as unregistered by "
        f"their old local paths: {result}"
    )
    assert all(path.exists() for path in originals), (
        "cleanup removed rendered PNGs whose records had committed under the "
        "parent task's canonical directory"
    )
    parent_output = (
        pathlib.Path(durable_workspace.base_dir) / "web_task_4242" / "output"
    )
    for row in rows:
        assert row.task_id == 4242
        assert pathlib.Path(str(row.storage_path)).is_relative_to(parent_output)
    fresh_parent = TaskWorkspace(
        id="web_task_4242", base_dir=str(durable_workspace.base_dir)
    )
    for page in result["pages"]:
        assert fresh_parent.resolve_file_id_detached(page["file_id"]) is not None


@pytest.mark.asyncio
async def test_fake_registration_ids_are_reported_as_unregistered(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    rendered: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=rendered
    )

    def fake_registration(files, *, db_session=None):
        del db_session
        return tuple(str(uuid.uuid4()) for _entries in files)

    monkeypatch.setattr(workspace, "register_files", fake_registration)

    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert result["registered_pages"] == []
    assert result["unregistered_pages"] == [1, 2]
    assert _output_rows(durable_workspace.session_factory) == []
    originals = [page.path for batch in rendered for page in batch]
    assert len(originals) == 2
    assert not any(path.exists() for path in originals)


def _render_lifecycle_source_root() -> str:
    import xagent

    return str(pathlib.Path(xagent.__path__[0]).resolve().parent)


_RENDER_OUTPUT_CREATION_CHILD = r"""
import asyncio
import os
import pathlib
import threading
import time
from pathlib import Path

from xagent.core.tools.core import pptx_render as render
from xagent.core.workspace import TaskWorkspace

root = Path(os.environ["LIFECYCLE_DIR"])
workspace = TaskWorkspace(id="web_task_1", base_dir=str(root / "uploads"))
deck = workspace.input_dir / "deck.pptx"
deck.write_bytes(b"PK\x03\x04 synthetic")
original_resolve = pathlib.Path.resolve
started = threading.Event()


def delayed_resolve(path, *args, **kwargs):
    if (
        path.name.startswith("pptx_render_")
        and path.parent == workspace.output_dir
        and args == (True,)
    ):
        started.set()
        time.sleep(0.3)
    return original_resolve(path, *args, **kwargs)


pathlib.Path.resolve = delayed_resolve


async def main():
    task = asyncio.create_task(render.render_pptx_to_images(str(deck), workspace))
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


asyncio.run(main())
print("CHILD_DONE")
"""


def test_output_creation_cleanup_survives_asyncio_run_exit(tmp_path):
    environment = dict(os.environ)
    ambient = os.environ.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = ":".join(
        part for part in (_render_lifecycle_source_root(), ambient) if part
    )
    for key in (
        "XAGENT_TEST_REDIS_URL",
        "MILVUS_URI",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        environment.pop(key, None)
    lifecycle_root = tmp_path / "lifecycle"
    lifecycle_root.mkdir(parents=True, exist_ok=True)
    environment["LIFECYCLE_DIR"] = str(lifecycle_root)
    environment["XAGENT_STORAGE_ROOT"] = str(tmp_path / "storage")
    environment["HOME"] = str(tmp_path / "home")
    (tmp_path / "home").mkdir(exist_ok=True)

    completed = subprocess.run(
        [sys.executable, "-B", "-c", _RENDER_OUTPUT_CREATION_CHILD],
        cwd=str(pathlib.Path(_render_lifecycle_source_root()).parent),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CHILD_DONE" in completed.stdout
    output_root = lifecycle_root / "uploads" / "web_task_1" / "output"
    leftover = (
        [str(path) for path in output_root.iterdir()] if output_root.is_dir() else []
    )
    assert leftover == [], (
        f"asyncio.run() shutdown left a render output directory behind: {leftover}"
    )
    combined = completed.stdout + completed.stderr
    for marker in (
        "Task was destroyed but it is pending",
        "Task exception was never retrieved",
    ):
        assert marker not in combined, combined


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize(
    "problem", ["owner", "task", "detached", "status", "outside", "sibling", "swapped"]
)
async def test_registration_rejects_invalid_persisted_records(
    durable_workspace, monkeypatch, tmp_path, delegated, problem
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    if delegated:
        workspace = TaskWorkspace(
            id="synthetic_delegated_render",
            base_dir=str(durable_workspace.base_dir),
            db_task_id=4242,
        )
    deck = _write_deck(workspace.input_dir)
    batches: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=batches
    )
    register = workspace.register_files

    def register_then_change_record(files, *, db_session=None):
        ids = register(files, db_session=db_session)
        with durable_workspace.session_factory() as db:
            record = db.query(UploadedFile).filter(UploadedFile.file_id == ids[0]).one()
            if problem == "owner":
                record.user_id += 1
            elif problem == "task":
                record.task_id += 1
            elif problem == "detached":
                record.detached_reason = "synthetic retained file"
            elif problem == "status":
                record.storage_status = "compensating"
            elif problem in {"outside", "sibling"}:
                root = tmp_path if problem == "outside" else durable_workspace.base_dir
                other = root / "unrelated" / "slide_001.png"
                other.parent.mkdir(parents=True)
                other.write_bytes(PNG_1X1)
                record.storage_path = str(other)
            db.commit()
        return tuple(reversed(ids)) if problem == "swapped" else ids

    monkeypatch.setattr(workspace, "register_files", register_then_change_record)
    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False, result
    assert result["error_code"] == "registration_failed"
    assert result["unverified_pages"] == ([1, 2] if problem == "swapped" else [1])
    assert result["unregistered_pages"] == []
    originals = [page.path for batch in batches for page in batch]
    assert len(originals) == 2
    assert all(path.is_file() for path in originals)
    assert len(_output_rows(durable_workspace.session_factory)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
@pytest.mark.parametrize("lookup", ["record", "owner"])
async def test_registration_lookup_outage_cannot_be_overridden_by_path_hit(
    durable_workspace, monkeypatch, delegated, lookup
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    if delegated:
        workspace = TaskWorkspace(
            id="synthetic_delegated_render",
            base_dir=str(durable_workspace.base_dir),
            db_task_id=4242,
        )
    deck = _write_deck(workspace.input_dir)
    batches: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=batches
    )

    def unavailable(*args):
        raise RuntimeError("synthetic metadata lookup outage")

    target = (
        "_lookup_uploaded_file_record"
        if lookup == "record"
        else "_expected_registration_owner"
    )
    monkeypatch.setattr(render, target, unavailable)
    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False, result
    assert result["error_code"] == "registration_failed"
    assert result["unverified_pages"] == [1, 2]
    assert result["unregistered_pages"] == []
    assert all(page.path.is_file() for batch in batches for page in batch)
    assert len(_output_rows(durable_workspace.session_factory)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_committed_images_survive_missing_returned_ids(
    durable_workspace, monkeypatch, delegated
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    if delegated:
        workspace = TaskWorkspace(
            id="synthetic_delegated_render",
            base_dir=str(durable_workspace.base_dir),
            db_task_id=4242,
        )
    deck = _write_deck(workspace.input_dir)
    batches: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=batches
    )
    register = workspace.register_files

    def commit_with_wrong_ids(files, *, db_session=None):
        register(files, db_session=db_session)
        return (str(uuid.uuid4()), str(uuid.uuid4()))

    monkeypatch.setattr(workspace, "register_files", commit_with_wrong_ids)
    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert result["unverified_pages"] == [1, 2]
    assert result["unregistered_pages"] == []
    originals = [page.path for batch in batches for page in batch]
    assert len(originals) == 2 and all(path.is_file() for path in originals)
    rows = _output_rows(durable_workspace.session_factory)
    assert len(rows) == 2
    assert all(pathlib.Path(str(row.storage_path)).is_file() for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [OSError, RuntimeError])
async def test_created_output_is_removed_when_validation_raises(
    durable_workspace, monkeypatch, failure
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    original_resolve = pathlib.Path.resolve
    created: list[pathlib.Path] = []

    def invalid(path, *args, **kwargs):
        if path.name.startswith("pptx_render_") and args == (True,):
            created.append(path)
            raise failure("synthetic directory validation failure")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", invalid)
    with pytest.raises((render.PptxRenderOutputError, RuntimeError)):
        await render._create_render_output_directory(workspace)
    assert len(created) == 1
    assert not created[0].exists()


@pytest.mark.asyncio
async def test_delegated_registration_accepts_canonical_collision_rename(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = TaskWorkspace(
        id="synthetic_delegated_render",
        base_dir=str(durable_workspace.base_dir),
        db_task_id=4242,
    )
    deck = _write_deck(workspace.input_dir)
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(monkeypatch, render, page_count=2, requested=[], rendered=[])
    register = workspace.register_files
    collisions: list[pathlib.Path] = []

    def register_after_collision(files, *, db_session=None):
        for path, _file_id in files:
            relative = pathlib.Path(path).relative_to(workspace.output_dir)
            occupied = durable_workspace.workspace.output_dir / relative
            occupied.parent.mkdir(parents=True, exist_ok=True)
            occupied.write_bytes(b"existing image")
            collisions.append(occupied)
        return register(files, db_session=db_session)

    monkeypatch.setattr(workspace, "register_files", register_after_collision)
    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is True, result
    assert [path.read_bytes() for path in collisions] == [b"existing image"] * 2
    rows = _output_rows(durable_workspace.session_factory)
    assert sorted(pathlib.Path(str(row.storage_path)).name for row in rows) == [
        "slide_001_1.png",
        "slide_002_1.png",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("commit_first", [False, True])
async def test_delegated_registration_exception_keeps_uncertain_images(
    durable_workspace, monkeypatch, commit_first
):
    render = _load_render_module()
    workspace = TaskWorkspace(
        id="synthetic_delegated_render",
        base_dir=str(durable_workspace.base_dir),
        db_task_id=4242,
    )
    deck = _write_deck(workspace.input_dir)
    batches: list[list] = []
    _install_conversion(monkeypatch, render, calls=[], timeout_records=[])
    _install_rasterizer(
        monkeypatch, render, page_count=2, requested=[], rendered=batches
    )
    register = workspace.register_files

    def interrupted(files, *, db_session=None):
        if commit_first:
            register(files[:1], db_session=db_session)
        raise RuntimeError("synthetic interrupted registration")

    monkeypatch.setattr(workspace, "register_files", interrupted)
    result = await render.render_pptx_to_images(str(deck), workspace)

    assert result["success"] is False
    assert result["error_code"] == "registration_failed"
    assert result["unregistered_pages"] == []
    assert result["unverified_pages"] == [1, 2]
    assert all(page.path.is_file() for batch in batches for page in batch)
    assert len(_output_rows(durable_workspace.session_factory)) == int(commit_first)


_REGISTRATION_SHUTDOWN_CHILD = r"""
import asyncio
import json
import os
import threading
from pathlib import Path

from pytest import MonkeyPatch

from xagent.core.tools.core import pptx_render as render
from tests.core.tools.test_pptx_render_tool import (
    durable_workspace,
    _write_deck,
    _install_conversion,
    _install_rasterizer,
    _output_rows,
)

root = Path(os.environ["REGISTRATION_TEST_ROOT"])
mode = os.environ["REGISTRATION_TEST_MODE"]
fail_after_commit = os.environ["REGISTRATION_TEST_FAILURE"] == "true"
patch = MonkeyPatch()
fixture_generator = durable_workspace.__wrapped__(monkeypatch=patch, tmp_path=root)
fixture = next(fixture_generator)
workspace = fixture.workspace
started = threading.Event()
release = threading.Event()
committed = threading.Event()
worker_finished = threading.Event()
observed = {}
tasks = []
release_timer = None

_install_conversion(patch, render, calls=[], timeout_records=[])
_install_rasterizer(patch, render, page_count=2, requested=[], rendered=[])
deck = _write_deck(workspace.input_dir)
original_apply = workspace._apply_prepared_file_registrations
original_register = render._register_rendered_pages
original_delete = render._delete_unregistered_outputs


def row_snapshot():
    return [
        {
            "id": str(row.file_id),
            "path": row.storage_path,
            "exists": Path(row.storage_path).is_file(),
            "status": row.storage_status,
        }
        for row in _output_rows(fixture.session_factory)
    ]


def delayed_apply(prepared, **kwargs):
    started.set()
    assert release.wait(10), "registration was never released"
    result = original_apply(prepared, **kwargs)
    committed.set()
    if fail_after_commit:
        raise OSError("registration acknowledgement failed after commit")
    return result


def observed_registration(*args, **kwargs):
    try:
        return original_register(*args, **kwargs)
    finally:
        worker_finished.set()


def observed_delete(pages, unregistered):
    observed["worker_finished_before_cleanup"] = worker_finished.is_set()
    release.set()
    assert committed.wait(10), "registration did not commit before deletion"
    assert worker_finished.wait(10), "registration thread did not finish"
    observed["before_delete"] = row_snapshot()
    observed["delete_pages"] = sorted(unregistered)
    original_delete(pages, unregistered)


patch.setattr(workspace, "_apply_prepared_file_registrations", delayed_apply)
patch.setattr(render, "_register_rendered_pages", observed_registration)
patch.setattr(render, "_delete_unregistered_outputs", observed_delete)


async def main():
    global release_timer
    task = asyncio.create_task(render.render_pptx_to_images(str(deck), workspace))
    tasks.append(task)
    for _ in range(1000):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set(), "the real registration stage was not reached"
    assert not task.done()
    release_timer = threading.Timer(1, release.set)
    release_timer.start()
    if mode == "cancel_await":
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    # In shutdown mode asyncio.run cancels every pending Task itself.


try:
    asyncio.run(main())
    observed["cancelled"] = tasks[0].cancelled()
    observed["worker_finished"] = worker_finished.is_set()
    observed["after_shutdown"] = row_snapshot()
    (root / "observations.json").write_text(json.dumps(observed, indent=2))
finally:
    release.set()
    if release_timer is not None:
        release_timer.join()
    try:
        next(fixture_generator)
    except StopIteration:
        pass
    patch.undo()
"""


@pytest.mark.parametrize("mode", ["shutdown", "cancel_await"])
@pytest.mark.parametrize("fail_after_commit", [False, True])
def test_registration_shutdown_keeps_committed_local_pngs(
    tmp_path, mode, fail_after_commit
):
    import json
    import sqlite3

    environment = dict(os.environ)
    source_root = _render_lifecycle_source_root()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source_root, os.environ.get("PYTHONPATH", "")) if part
    )
    environment.update(
        REGISTRATION_TEST_ROOT=str(tmp_path),
        REGISTRATION_TEST_MODE=mode,
        REGISTRATION_TEST_FAILURE=str(fail_after_commit).lower(),
        HOME=str(tmp_path / "home"),
        TMPDIR=str(tmp_path / "tmp"),
        XAGENT_STORAGE_ROOT=str(tmp_path / "storage"),
    )
    (tmp_path / "home").mkdir()
    (tmp_path / "tmp").mkdir()
    for key in list(environment):
        if key.lower().endswith("_proxy") or key in {
            "XAGENT_TEST_REDIS_URL",
            "MILVUS_URI",
        }:
            environment.pop(key)
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _REGISTRATION_SHUTDOWN_CHILD],
        cwd=str(pathlib.Path(source_root).parent),
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, combined
    observed = json.loads((tmp_path / "observations.json").read_text())
    assert observed["cancelled"] and observed["worker_finished"], observed
    assert len(observed["before_delete"]) == 2, observed
    assert all(
        row["exists"] and row["status"] == "available"
        for row in observed["before_delete"]
    ), observed
    with sqlite3.connect(tmp_path / "metadata.db") as database:
        rows = database.execute(
            "SELECT file_id, storage_path, storage_status FROM uploaded_files"
        ).fetchall()
    assert len(rows) == 2, observed
    assert all(status == "available" for _, _, status in rows), rows
    assert all(pathlib.Path(path).is_file() for _, path, _ in rows), (
        "normal shutdown deleted local PNGs after their records committed",
        observed,
    )
    assert observed["worker_finished_before_cleanup"], observed
    assert observed["delete_pages"] == [], observed
    for warning in (
        "Task was destroyed but it is pending",
        "Task exception was never retrieved",
        "Future exception was never retrieved",
    ):
        assert warning not in combined, combined


@pytest.mark.asyncio
async def test_blocking_operation_preserves_context_and_runs_off_loop():
    from contextvars import ContextVar

    render = _load_render_module()
    value = ContextVar("render_context", default="missing")
    token = value.set("workspace-context")
    loop_thread = threading.get_ident()
    try:
        observed = await render._run_blocking(
            lambda: (value.get(), threading.get_ident())
        )
    finally:
        value.reset(token)
    assert observed[0] == "workspace-context"
    assert observed[1] != loop_thread


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["conversion", "write", "registration"])
@pytest.mark.parametrize("existing_image", [False, True])
async def test_replaced_render_directory_never_modifies_outside_files(
    durable_workspace, monkeypatch, tmp_path, stage, existing_image
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    deck = _write_deck(workspace.input_dir)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel.txt").write_bytes(b"preserve this file")
    if existing_image:
        (outside / "slide_001.png").write_bytes(b"preserve this image")
    before = {p.name: (p.read_bytes(), p.stat().st_ino) for p in outside.iterdir()}
    moved = workspace.output_dir / "moved_render"
    replaced = []

    def replace_directory():
        directory = next(workspace.output_dir.glob("pptx_render_*"))
        directory.rename(moved)
        directory.symlink_to(outside, target_is_directory=True)
        replaced.append(directory)

    async def convert(source, output_dir, **kwargs):
        if stage == "conversion":
            replace_directory()
        pdf = output_dir / "deck.pdf"
        pdf.write_bytes(b"%PDF synthetic")
        return pdf

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert)
    monkeypatch.setattr(render, "fitz", _FakeFitz([(72, 54)], []))
    save = _FakePixelExtent.save

    def save_and_replace(pixmap, destination, *args, **kwargs):
        if stage == "write":
            replace_directory()
        return save(pixmap, destination, *args, **kwargs)

    monkeypatch.setattr(_FakePixelExtent, "save", save_and_replace)
    register = workspace.register_files

    def register_and_replace(files, **kwargs):
        ids = register(files, **kwargs)
        if stage == "registration":
            replace_directory()
        return ids

    monkeypatch.setattr(workspace, "register_files", register_and_replace)
    result = await render.render_pptx_to_images(str(deck), workspace)

    after = {p.name: (p.read_bytes(), p.stat().st_ino) for p in outside.iterdir()}
    assert after == before, "rendering changed files outside the task"
    assert result["success"] is False, result
    assert len(replaced) == 1
    assert replaced[0].is_symlink()
    if stage == "registration":
        assert (moved / "slide_001.png").read_bytes() == PNG_1X1
        assert len(_output_rows(durable_workspace.session_factory)) == 1


@pytest.mark.asyncio
async def test_render_enables_hidden_slides_for_pdf_export(
    durable_workspace, monkeypatch
):
    render = _load_render_module()
    workspace = durable_workspace.workspace
    source = _write_deck(workspace.input_dir)
    options = []

    async def convert(source, output, **kwargs):
        options.append(kwargs)
        pdf = output / "deck.pdf"
        pdf.write_bytes(b"%PDF synthetic")
        return pdf

    monkeypatch.setattr(render, "convert_pptx_to_pdf", convert)
    _install_rasterizer(monkeypatch, render, page_count=3, requested=[], rendered=[])
    result = await render.render_pptx_to_images(str(source), workspace)
    assert result["success"] is True, result
    assert options[0].get("include_hidden_slides") is True


def _write_hidden_middle_deck(path):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches

    presentation = Presentation()
    for number, color in enumerate([(220, 30, 40), (20, 180, 60), (30, 60, 220)], 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(*color)
        text = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
        text.text_frame.text = f"Source slide {number}"
        if number == 2:
            slide._element.set("show", "0")
    presentation.save(str(path))


@pytest.mark.asyncio
@pytest.mark.parametrize("page_number, color", [(2, (20, 180, 60)), (3, (30, 60, 220))])
async def test_real_soffice_preserves_hidden_source_slide_numbers(
    durable_workspace, page_number, color
):
    if shutil.which("soffice") is None:
        pytest.skip("LibreOffice is not installed")
    from PIL import Image

    render = _load_render_module()
    workspace = durable_workspace.workspace
    source = workspace.input_dir / "hidden-middle.pptx"
    _write_hidden_middle_deck(source)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    result = await render.render_pptx_to_images(
        str(source), workspace, start_page=page_number, end_page=page_number
    )
    assert result["success"] is True, result
    assert result["total_pages"] == 3
    assert result["page_numbers"] == [page_number]
    rows = _output_rows(durable_workspace.session_factory)
    assert len(rows) == 1 and rows[0].storage_status == "available"
    with Image.open(rows[0].storage_path) as image:
        assert image.convert("RGB").getpixel((10, 10)) == color
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


@pytest.mark.asyncio
async def test_real_soffice_preview_keeps_default_hidden_slide_behavior(tmp_path):
    if shutil.which("soffice") is None:
        pytest.skip("LibreOffice is not installed")
    import fitz

    conversion = _load_conversion_module()
    source = tmp_path / "hidden-middle.pptx"
    _write_hidden_middle_deck(source)
    output = tmp_path / "pdf"
    output.mkdir()
    pdf = await conversion.convert_pptx_to_pdf(source, output)
    with fitz.open(pdf) as document:
        assert document.page_count == 2
        assert "Source slide 3" in document[1].get_text()
