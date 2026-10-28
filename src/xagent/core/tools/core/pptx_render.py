"""Render the slides of a workspace ``.pptx`` into one PNG per slide.

The module is deliberately independent from the web preview path: it converts
the deck through :mod:`xagent.core.tools.core.pptx_conversion`, rasterises only
the selected PDF pages with PyMuPDF, and registers each produced PNG as a
durable workspace file so a later turn can show it or pass its ``file_id`` to a
vision model.

Bounds are part of the tool contract, not caller advice:

* at most :data:`MAX_RENDERED_PAGES` PNG pages per call;
* at most :data:`MAX_PAGE_PIXELS` pixels per PNG, decided from the PDF page
  rectangle before a pixmap is allocated;
* one LibreOffice conversion with the shared
  :data:`~xagent.core.tools.core.pptx_conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS`
  timeout.

LibreOffice converts the whole deck, including hidden slides, so PDF page
numbers match source slide numbers. Only the PNG output is page-bounded.
"""

from __future__ import annotations

import asyncio
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from ....config import get_max_upload_size_bytes
from ...file_ref import build_workspace_file_ref, parse_file_id_ref
from ...workspace import TaskWorkspace
from ..artifacts import build_inline_artifact
from .pptx_conversion import (
    DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    PptxConversionFailedError,
    PptxConversionTimeoutError,
    PptxConverterUnavailableError,
    convert_pptx_to_pdf,
)

MAX_RENDERED_PAGES = 20
MAX_PAGE_PIXELS = 4_000_000
DEFAULT_RENDER_SCALE = 2.0
PNG_MIME_TYPE = "image/png"
RENDER_OUTPUT_DIR_PREFIX = "pptx_render_"


class PptxRenderError(ValueError):
    """Base class for render failures that map to a tool error result."""

    error_code = "render_error"


class PptxRenderArgumentError(PptxRenderError):
    error_code = "invalid_argument"


class PptxRenderInputError(PptxRenderError):
    error_code = "input_not_found"


class PptxRenderNotAFileError(PptxRenderError):
    error_code = "input_not_a_file"


class PptxRenderUnsupportedInputError(PptxRenderError):
    error_code = "unsupported_format"


class PptxRenderOutsideWorkspaceError(PptxRenderError):
    error_code = "input_outside_workspace"


class PptxRenderEmptyPresentationError(PptxRenderError):
    error_code = "empty_presentation"


class PptxRenderPageRangeError(PptxRenderError):
    error_code = "page_range_invalid"


class PptxRenderPdfError(PptxRenderError):
    error_code = "pdf_unreadable"


class PptxRenderRasterError(PptxRenderError):
    error_code = "raster_failed"


class PptxRenderOutputError(PptxRenderError):
    error_code = "output_unavailable"


@dataclass(frozen=True)
class PageSelection:
    """Resolved 1-based inclusive page range for one render call."""

    page_numbers: tuple[int, ...]
    truncated: bool
    total_pages: int


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: Path
    width: int
    height: int


def page_file_name(page_number: int) -> str:
    """Return the deterministic output filename for a 1-based page number."""
    return f"slide_{page_number:03d}.png"


def resolve_page_selection(
    total_pages: int,
    start_page: int | None = None,
    end_page: int | None = None,
) -> PageSelection:
    """Resolve the page range against the deck size.

    With neither bound given the first ``min(total, MAX_RENDERED_PAGES)`` pages
    are selected and ``truncated`` says the page bound was applied. An explicit
    range is never silently rewritten: a range over the bound or outside the
    deck raises instead of selecting different pages.
    """
    if total_pages < 1:
        raise PptxRenderEmptyPresentationError(
            "The presentation contains no slides to render."
        )
    start, end = start_page, end_page
    if (start is not None and start < 1) or (end is not None and end < 1):
        raise PptxRenderArgumentError("Page numbers must be at least 1.")
    if start is not None and end is not None and start > end:
        raise PptxRenderArgumentError("start_page must not be greater than end_page.")
    if start is None and end is None:
        last_page = min(total_pages, MAX_RENDERED_PAGES)
        return PageSelection(
            page_numbers=tuple(range(1, last_page + 1)),
            truncated=total_pages > MAX_RENDERED_PAGES,
            total_pages=total_pages,
        )
    resolved_start = 1 if start is None else start
    resolved_end = total_pages if end is None else end
    if resolved_start > total_pages:
        raise PptxRenderPageRangeError(
            f"start_page ({resolved_start}) is beyond the last page "
            f"({total_pages}) of this presentation."
        )
    if resolved_end > total_pages:
        raise PptxRenderPageRangeError(
            f"end_page ({resolved_end}) is beyond the last page "
            f"({total_pages}) of this presentation."
        )
    requested = resolved_end - resolved_start + 1
    if requested > MAX_RENDERED_PAGES:
        raise PptxRenderArgumentError(
            f"The requested range covers {requested} slides, but at most "
            f"{MAX_RENDERED_PAGES} pages can be rendered per call. "
            "Narrow start_page/end_page."
        )
    return PageSelection(
        page_numbers=tuple(range(resolved_start, resolved_end + 1)),
        truncated=False,
        total_pages=total_pages,
    )


def unrendered_page_ranges(
    total_pages: int, page_numbers: list[int] | tuple[int, ...]
) -> list[list[int]]:
    """Return the 1-based page ranges that were not rendered."""
    ranges: list[list[int]] = []
    expected = 1
    for page_number in sorted(set(page_numbers)):
        if page_number > expected:
            ranges.append([expected, page_number - 1])
        expected = page_number + 1
    if expected <= total_pages:
        ranges.append([expected, total_pages])
    return ranges


def _page_scale(page: Any, scale: float, max_pixels: int) -> float:
    width, height = float(page.rect.width), float(page.rect.height)
    area = width * height
    if width <= 0 or height <= 0 or not math.isfinite(area) or area <= 0:
        raise PptxRenderRasterError("The PDF page has invalid or oversized dimensions.")
    scale = min(DEFAULT_RENDER_SCALE, scale, math.sqrt(max_pixels / area))
    while scale > 0:
        extent = (page.rect * fitz.Matrix(scale, scale)).irect
        if extent.width <= 0 or extent.height <= 0:
            break
        if extent.width * extent.height <= max_pixels:
            return scale
        scale *= 0.99
    raise PptxRenderRasterError(
        "The PDF page is too large to render within the pixel limit."
    )


def _open_pdf(pdf_path: Path) -> Any:
    try:
        return fitz.open(str(pdf_path))
    except Exception as exc:
        raise PptxRenderPdfError(
            f"The converted PDF {pdf_path.name} could not be opened: {exc}"
        ) from exc


def read_pdf_page_count(pdf_path: Path) -> int:
    """Return the number of pages of a converted PDF."""
    document = _open_pdf(Path(pdf_path))
    try:
        return int(document.page_count)
    finally:
        document.close()


def render_pdf_pages_to_pngs(
    pdf_path: Path,
    page_numbers: list[int] | tuple[int, ...],
    output_dir: Path,
    *,
    output_fd: int | None = None,
    scale: float = DEFAULT_RENDER_SCALE,
    max_pixels: int = MAX_PAGE_PIXELS,
) -> tuple[int, list[RenderedPage]]:
    """Rasterize selected pages and create PNGs through a held directory handle."""
    owned_fd = output_fd is None
    if output_fd is None:
        output_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    document = None
    try:
        document = _open_pdf(pdf_path)
        rendered = []
        for number in page_numbers:
            page = document[number - 1]
            effective_scale = _page_scale(page, scale, max_pixels)
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(effective_scale, effective_scale), alpha=False
            )
            target = output_dir / page_file_name(number)
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=output_fd,
            )
            with os.fdopen(descriptor, "wb") as image_file:
                pixmap.pil_save(image_file, format="PNG")
            rendered.append(RenderedPage(number, target, pixmap.width, pixmap.height))
        return int(document.page_count), rendered
    except PptxRenderError:
        raise
    except Exception as exc:
        raise PptxRenderRasterError(
            f"Could not render the slide images: {exc}"
        ) from exc
    finally:
        if document is not None:
            document.close()
        if owned_fd:
            os.close(output_fd)


def _looks_like_file_id(value: str) -> bool:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return str(parsed) == value.lower()


def _resolve_pptx_input(workspace: TaskWorkspace, raw_selector: str) -> Path:
    """Resolve an authorized workspace input that must be an existing .pptx."""
    selector = str(raw_selector).strip()
    file_id = parse_file_id_ref(selector)
    if file_id is None and _looks_like_file_id(selector):
        file_id = selector
    if file_id is not None:
        resolved = workspace.resolve_file_id_detached(file_id)
        if resolved is None:
            raise PptxRenderInputError(
                f"File reference {file_id!r} is not available in this task workspace."
            )
        path = Path(resolved)
    else:
        try:
            path = Path(
                workspace.resolve_path_with_search(selector, detached_file_ids=True)
            )
        except FileNotFoundError as exc:
            raise PptxRenderInputError(
                f"The presentation {selector!r} was not found in this task workspace."
            ) from exc
        except ValueError as exc:
            raise PptxRenderOutsideWorkspaceError(
                f"The path {selector!r} is outside the task workspace or its "
                "authorized upload directories."
            ) from exc
    try:
        path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PptxRenderInputError(
            f"The presentation {selector!r} could not be resolved to a file."
        ) from exc
    if not path.is_file():
        raise PptxRenderNotAFileError(
            f"The presentation {selector!r} is not a regular file."
        )
    if path.suffix.lower() != ".pptx":
        raise PptxRenderUnsupportedInputError(
            f"The presentation {selector!r} must be a .pptx file."
        )
    return path


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "success": False,
        "error_code": code,
        "error": message,
    }
    result.update(extra)
    return result


def _selection_payload(selection: PageSelection) -> dict[str, Any]:
    rendered_count = len(selection.page_numbers)
    return {
        "total_pages": selection.total_pages,
        "rendered_pages": rendered_count,
        "page_numbers": list(selection.page_numbers),
        "truncated": selection.truncated,
        "unrendered_pages": selection.total_pages - rendered_count,
        "unrendered_page_ranges": unrendered_page_ranges(
            selection.total_pages, selection.page_numbers
        ),
        "page_limit": MAX_RENDERED_PAGES,
        "max_pixels_per_page": MAX_PAGE_PIXELS,
        "conversion_timeout_seconds": DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    }


def _build_pages_payload(
    workspace: TaskWorkspace,
    rendered: list[RenderedPage],
    persisted: dict[int, str],
) -> list[dict[str, Any]]:
    """Build FileRefs and inline artifacts; runs in a worker thread."""
    pages: list[dict[str, Any]] = []
    for page in rendered:
        file_id = persisted[page.page_number]
        file_ref = build_workspace_file_ref(
            workspace=workspace,
            file_path=page.path,
            file_id=file_id,
            mime_type=PNG_MIME_TYPE,
        )
        pages.append(
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "file_id": file_id,
                "file_ref": file_ref,
                "artifacts": [build_inline_artifact(file_ref)],
            }
        )
    return pages


def _render_and_register(
    pdf_path: Path,
    workspace: TaskWorkspace,
    start_page: int | None,
    end_page: int | None,
) -> dict[str, Any]:
    selection = resolve_page_selection(
        read_pdf_page_count(pdf_path), start_page, end_page
    )
    directory_name = RENDER_OUTPUT_DIR_PREFIX + uuid.uuid4().hex
    output_dir = workspace.output_dir / directory_name
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open(workspace.output_dir, flags)
    try:
        os.mkdir(directory_name, mode=0o700, dir_fd=parent_fd)
        directory_fd = None
        registration_started = False
        try:
            directory_fd = os.open(directory_name, flags, dir_fd=parent_fd)
            _, rendered = render_pdf_pages_to_pngs(
                pdf_path, selection.page_numbers, output_dir, output_fd=directory_fd
            )
            registration_started = True
            try:
                persisted = {
                    page.page_number: workspace.register_delivery_file(str(page.path))
                    for page in rendered
                }
                pages = _build_pages_payload(workspace, rendered, persisted)
            except Exception as exc:
                return _error(
                    "registration_failed", f"Could not register slide images: {exc}"
                )
            return {
                "success": True,
                "output_dir": str(output_dir),
                **_selection_payload(selection),
                "pages": pages,
                "artifacts": [
                    artifact for page in pages for artifact in page["artifacts"]
                ],
            }
        finally:
            try:
                if not registration_started:
                    if directory_fd is not None:
                        for number in selection.page_numbers:
                            try:
                                os.unlink(page_file_name(number), dir_fd=directory_fd)
                            except FileNotFoundError:
                                pass
                    # rmdir never follows a replacement symlink and only removes an empty directory.
                    try:
                        os.rmdir(directory_name, dir_fd=parent_fd)
                    except (FileNotFoundError, NotADirectoryError):
                        pass
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)
    finally:
        os.close(parent_fd)


async def render_pptx_to_images(
    pptx_path: str,
    workspace: TaskWorkspace | None,
    *,
    start_page: int | None = None,
    end_page: int | None = None,
) -> dict[str, Any]:
    """Render selected source slides, including hidden ones, into registered PNGs.

    Registration failures preserve the images. Failures before registration
    remove this call's output; cancellation propagates without waiting for workers.
    """
    if workspace is None:
        return _error(
            "workspace_unavailable", "Rendering slides requires a task workspace."
        )
    if workspace.current_task_id is None:
        return _error(
            "durable_registration_unavailable",
            "Rendering slides requires a persistent task owner.",
        )
    staged_input = None
    try:
        source = await asyncio.to_thread(_resolve_pptx_input, workspace, pptx_path)
        staged_input = await asyncio.to_thread(
            workspace.stage_resolved_path_for_external_upload,
            source,
            pptx_path,
            max_bytes=get_max_upload_size_bytes(),
            host_private=True,
        )
        with tempfile.TemporaryDirectory(prefix="xagent-pptx-render-") as pdf_dir:
            pdf_path = await convert_pptx_to_pdf(
                staged_input,
                Path(pdf_dir),
                timeout_seconds=DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
                include_hidden_slides=True,
            )
            result = await asyncio.to_thread(
                _render_and_register, pdf_path, workspace, start_page, end_page
            )
            if result["success"]:
                result["pptx_path"] = str(source)
            return result
    except PptxRenderError as exc:
        return _error(exc.error_code, str(exc))
    except PptxConverterUnavailableError as exc:
        return _error("pdf_conversion_unavailable", str(exc))
    except PptxConversionTimeoutError as exc:
        return _error("pdf_conversion_timeout", str(exc))
    except PptxConversionFailedError as exc:
        return _error("pdf_conversion_failed", str(exc))
    except (FileNotFoundError, ValueError) as exc:
        return _error("input_not_found", str(exc))
    except OSError as exc:
        return _error("output_unavailable", str(exc))
    finally:
        if staged_input is not None:
            await asyncio.to_thread(
                workspace.discard_staged_external_upload, staged_input
            )
