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

LibreOffice converts the whole deck; only the PNG output is page-bounded.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from ...file_ref import build_workspace_file_ref, parse_file_id_ref
from ...workspace import TaskWorkspace
from ..artifacts import build_inline_artifact
from .pptx_conversion import (
    DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    PptxConversionError,
    PptxConversionFailedError,
    PptxConversionTimeoutError,
    PptxConverterUnavailableError,
    convert_pptx_to_pdf,
)

logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:  # pragma: no cover - PyMuPDF is part of the backend image
    fitz = None

MAX_RENDERED_PAGES = 20
MAX_PAGE_PIXELS = 4_000_000
DEFAULT_RENDER_SCALE = 2.0
PNG_MIME_TYPE = "image/png"
RENDER_OUTPUT_DIR_PREFIX = "pptx_render_"

_T = TypeVar("_T")


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


class PptxRenderRegistrationError(PptxRenderError):
    error_code = "registration_failed"

    def __init__(self, message: str, *, registered_pages: tuple[int, ...] = ()):
        super().__init__(message)
        self.registered_pages = registered_pages


@dataclass(frozen=True)
class PageSelection:
    """Resolved 1-based inclusive page range for one render call."""

    page_numbers: tuple[int, ...]
    truncated: bool
    total_pages: int


@dataclass(frozen=True)
class PageGeometry:
    """Pixmap geometry for one PDF page, decided before allocation."""

    scale: float
    width: int
    height: int


@dataclass(frozen=True)
class RenderedPage:
    """One PNG produced from a PDF page."""

    page_number: int
    path: Path
    width: int
    height: int


def page_file_name(page_number: int) -> str:
    """Return the deterministic output filename for a 1-based page number."""
    return f"slide_{page_number:03d}.png"


def _strict_page_number(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PptxRenderArgumentError(
            f"{name} must be a 1-based integer page number, got {type(value).__name__}."
        )
    return int(value)


def validate_page_arguments(
    start_page: Any, end_page: Any
) -> tuple[int | None, int | None]:
    """Validate the caller's page range independently of the deck size.

    Missing ends are legal; the start defaults to page 1, the end defaults to
    the last page. An explicit range that can never fit the page bound is
    rejected here, before any conversion work starts.
    """
    start = _strict_page_number("start_page", start_page)
    end = _strict_page_number("end_page", end_page)
    for name, value in (("start_page", start), ("end_page", end)):
        if value is not None and value < 1:
            raise PptxRenderArgumentError(f"{name} must be at least 1, got {value}.")
    if start is not None and end is not None and start > end:
        raise PptxRenderArgumentError(
            f"start_page ({start}) must not be greater than end_page ({end})."
        )
    if end is not None:
        requested = end - (start if start is not None else 1) + 1
        if requested > MAX_RENDERED_PAGES:
            raise PptxRenderArgumentError(
                f"The requested range covers {requested} slides, but at most "
                f"{MAX_RENDERED_PAGES} pages can be rendered per call. "
                "Narrow start_page/end_page."
            )
    return start, end


def resolve_page_selection(
    total_pages: int,
    start_page: Any = None,
    end_page: Any = None,
) -> PageSelection:
    """Resolve the page range against the deck size.

    With neither bound given the first ``min(total, MAX_RENDERED_PAGES)`` pages
    are selected and ``truncated`` says the page bound was applied. An explicit
    range is never silently rewritten: a range over the bound or outside the
    deck raises instead of selecting different pages.
    """
    if isinstance(total_pages, bool) or not isinstance(total_pages, int):
        raise PptxRenderEmptyPresentationError(
            "The converted PDF has no readable page count."
        )
    if total_pages < 1:
        raise PptxRenderEmptyPresentationError(
            "The presentation contains no slides to render."
        )
    start, end = validate_page_arguments(start_page, end_page)
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


def _pixmap_extent(length: float, scale: float) -> int:
    """Predict the pixel extent PyMuPDF allocates for one page dimension."""
    return max(1, math.ceil(length * scale - 1e-9))


def compute_page_geometry(
    page_width: float,
    page_height: float,
    *,
    scale: float = DEFAULT_RENDER_SCALE,
    max_pixels: int = MAX_PAGE_PIXELS,
) -> PageGeometry:
    """Decide the raster scale and rounded pixel size for one PDF page.

    The scale is lowered until the *rounded* width x height product fits
    ``max_pixels``; the aspect ratio is preserved because one scale is shared
    by both dimensions. Called before the pixmap exists, so an oversized page
    never allocates an oversized raster.
    """
    for value in (page_width, page_height, scale):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PptxRenderRasterError(
                "PDF page dimensions and render scale must be numbers."
            )
        if not math.isfinite(value) or value <= 0:
            raise PptxRenderRasterError(
                "PDF page dimensions and render scale must be positive and finite."
            )
    if isinstance(max_pixels, bool) or not isinstance(max_pixels, int):
        raise PptxRenderRasterError("The page pixel budget must be an integer.")
    if max_pixels < 1:
        raise PptxRenderRasterError("The page pixel budget must be positive.")
    area = float(page_width) * float(page_height)
    if not math.isfinite(area):
        raise PptxRenderRasterError(
            "The PDF page area is too large to compute a raster size."
        )
    max_scale = math.sqrt(max_pixels / area)
    effective_scale = min(float(scale), max_scale)
    width = _pixmap_extent(float(page_width), effective_scale)
    height = _pixmap_extent(float(page_height), effective_scale)
    for _attempt in range(64):
        if width * height <= max_pixels:
            return PageGeometry(scale=effective_scale, width=width, height=height)
        effective_scale *= math.sqrt(max_pixels / (width * height)) * (1 - 1e-9)
        if not math.isfinite(effective_scale) or effective_scale <= 0:
            break
        width = _pixmap_extent(float(page_width), effective_scale)
        height = _pixmap_extent(float(page_height), effective_scale)
    raise PptxRenderRasterError(
        "Could not reduce the page raster below the pixel budget."
    )


def _open_pdf(pdf_path: Path) -> Any:
    if fitz is None:
        raise PptxRenderRasterError(
            "PyMuPDF is required to render slides to images but is not "
            "installed in this environment."
        )
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
    scale: float = DEFAULT_RENDER_SCALE,
    max_pixels: int = MAX_PAGE_PIXELS,
) -> tuple[int, list[RenderedPage]]:
    """Rasterise exactly the requested 1-based pages into ``output_dir``.

    Only the selected pages reach ``get_pixmap``; the pixel budget is checked
    from the page rectangle first. Any write or raster failure propagates as
    :class:`PptxRenderRasterError` so the caller can clean up the whole call.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        raise PptxRenderRasterError(f"Output directory {output_dir} does not exist.")
    document = _open_pdf(Path(pdf_path))
    rendered: list[RenderedPage] = []
    try:
        total_pages = int(document.page_count)
        for page_number in page_numbers:
            page = document[page_number - 1]
            rectangle = page.rect
            geometry = compute_page_geometry(
                float(rectangle.width),
                float(rectangle.height),
                scale=scale,
                max_pixels=max_pixels,
            )
            if fitz is None:  # pragma: no cover - guarded by _open_pdf
                raise PptxRenderRasterError("PyMuPDF is not available.")
            matrix = fitz.Matrix(geometry.scale, geometry.scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            if int(pixmap.width) * int(pixmap.height) > max_pixels:
                raise PptxRenderRasterError(
                    f"Page {page_number} rasterised to "
                    f"{pixmap.width}x{pixmap.height} pixels, which exceeds the "
                    f"{max_pixels} pixel budget."
                )
            target = output_dir / page_file_name(page_number)
            try:
                pixmap.save(str(target))
            except OSError as exc:
                raise PptxRenderRasterError(
                    f"Could not write slide image {target.name}: {exc}"
                ) from exc
            rendered.append(
                RenderedPage(
                    page_number=page_number,
                    path=target,
                    width=int(pixmap.width),
                    height=int(pixmap.height),
                )
            )
        return total_pages, rendered
    finally:
        document.close()


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


def _persisted_page_ids(
    workspace: TaskWorkspace, rendered: list[RenderedPage]
) -> dict[int, str]:
    """Return the page numbers whose PNG path has a persisted file record."""
    persisted: dict[int, str] = {}
    for page in rendered:
        try:
            file_id = workspace.get_registered_file_id(str(page.path))
        except Exception:  # pragma: no cover - defensive: unreadable metadata
            logger.warning(
                "Could not verify registration of rendered slide %s",
                page.path,
                exc_info=True,
            )
            file_id = None
        if file_id:
            persisted[page.page_number] = str(file_id)
    return persisted


def _register_rendered_pages(
    workspace: TaskWorkspace, rendered: list[RenderedPage]
) -> dict[int, str]:
    """Batch-register every PNG and verify each registration is persisted."""
    paths = [str(page.path) for page in rendered]
    try:
        registered_ids = workspace.register_files([(path, None) for path in paths])
    except Exception as exc:
        persisted = _persisted_page_ids(workspace, rendered)
        raise PptxRenderRegistrationError(
            f"Rendered slide images could not be registered durably: {exc}",
            registered_pages=tuple(sorted(persisted)),
        ) from exc
    if len(registered_ids) != len(rendered):
        persisted = _persisted_page_ids(workspace, rendered)
        raise PptxRenderRegistrationError(
            f"Only {len(registered_ids)} of {len(rendered)} rendered slide "
            "images received a file id.",
            registered_pages=tuple(sorted(persisted)),
        )
    persisted = _persisted_page_ids(workspace, rendered)
    missing = [
        page.page_number for page in rendered if page.page_number not in persisted
    ]
    if missing:
        raise PptxRenderRegistrationError(
            "Rendered slide images were not persisted as durable files for "
            f"page(s) {', '.join(str(number) for number in missing)}; their "
            "file ids would not be reusable in a later turn.",
            registered_pages=tuple(sorted(persisted)),
        )
    for page, file_id in zip(rendered, registered_ids):
        if persisted.get(page.page_number) != str(file_id):
            raise PptxRenderRegistrationError(
                f"The persisted file id for page {page.page_number} does not "
                "match the id returned by registration.",
                registered_pages=tuple(sorted(persisted)),
            )
    return persisted


def _delete_unregistered_outputs(
    rendered: list[RenderedPage], persisted_pages: set[int]
) -> None:
    """Remove this call's PNGs that were not durably registered."""
    directories = {page.path.parent for page in rendered}
    for page in rendered:
        if page.page_number in persisted_pages:
            continue
        try:
            page.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.warning(
                "Could not remove unregistered rendered slide %s",
                page.path,
                exc_info=True,
            )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            # A registered PNG from this call still lives here.
            pass


def _remove_tree(path: Path | None) -> None:
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


async def _run_blocking(operation: Callable[[], _T]) -> _T:
    """Run blocking work in a thread and drain it if the caller is cancelled."""
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        raise


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


async def render_pptx_to_images(
    pptx_path: Any,
    workspace: Any,
    *,
    start_page: Any = None,
    end_page: Any = None,
) -> dict[str, Any]:
    """Render selected slides of a workspace ``.pptx`` into registered PNGs.

    Returns a result dictionary. Failures are reported as ``success=False``
    with a stable ``error_code`` and a human-readable ``error``; they never
    raise for expected input/conversion/registration failures.
    """
    if workspace is None or not isinstance(workspace, TaskWorkspace):
        return _error(
            "workspace_unavailable",
            "Rendering slides requires a task workspace; no workspace was "
            "bound to this tool call.",
        )
    if getattr(workspace, "current_task_id", None) is None:
        return _error(
            "durable_registration_unavailable",
            "Rendering slides requires a workspace with a persistent task "
            "owner so the PNG pages can be registered as durable files.",
        )
    if not isinstance(pptx_path, str) or not pptx_path.strip():
        return _error(
            "invalid_argument",
            "pptx_path must be a non-empty .pptx path, file reference or "
            "file_id string.",
        )
    try:
        validated_start, validated_end = validate_page_arguments(start_page, end_page)
    except PptxRenderError as exc:
        return _error(exc.error_code, str(exc))

    try:
        resolved_input = await _run_blocking(
            lambda: _resolve_pptx_input(workspace, pptx_path)
        )
    except PptxRenderError as exc:
        return _error(exc.error_code, str(exc))

    temp_root: Path | None = None
    output_dir: Path | None = None
    rendered: list[RenderedPage] = []
    persisted: dict[int, str] = {}
    succeeded = False
    try:
        try:
            temp_root = Path(
                await asyncio.to_thread(tempfile.mkdtemp, prefix="xagent-pptx-render-")
            )
            pdf_dir = temp_root / "pdf"
            await asyncio.to_thread(pdf_dir.mkdir)
            output_dir = Path(workspace.output_dir) / (
                f"{RENDER_OUTPUT_DIR_PREFIX}{uuid.uuid4().hex[:12]}"
            )
            await asyncio.to_thread(output_dir.mkdir)
        except OSError as exc:
            return _error(
                "raster_failed",
                f"Could not create a temporary render directory: {exc}",
            )

        try:
            pdf_path = await convert_pptx_to_pdf(
                resolved_input,
                pdf_dir,
                timeout_seconds=DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
            )
        except PptxConverterUnavailableError as exc:
            return _error("pdf_conversion_unavailable", str(exc))
        except PptxConversionTimeoutError as exc:
            return _error("pdf_conversion_timeout", str(exc))
        except PptxConversionFailedError as exc:
            return _error("pdf_conversion_failed", str(exc))
        except PptxConversionError as exc:  # pragma: no cover - base fallback
            return _error("pdf_conversion_failed", str(exc))

        try:
            total_pages = await _run_blocking(lambda: read_pdf_page_count(pdf_path))
            selection = resolve_page_selection(
                total_pages, validated_start, validated_end
            )
        except PptxRenderError as exc:
            return _error(exc.error_code, str(exc))

        try:
            _total, rendered = await _run_blocking(
                lambda: render_pdf_pages_to_pngs(
                    pdf_path, selection.page_numbers, output_dir
                )
            )
        except PptxRenderError as exc:
            return _error(exc.error_code, str(exc))
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Rendering slides failed unexpectedly", exc_info=True)
            return _error("raster_failed", f"Rendering the slides failed: {exc}")

        try:
            persisted = await _run_blocking(
                lambda: _register_rendered_pages(workspace, rendered)
            )
        except PptxRenderRegistrationError as exc:
            return _error(
                "registration_failed",
                str(exc),
                registered_pages=list(exc.registered_pages),
                rendered_pages=len(rendered),
            )

        pages_payload: list[dict[str, Any]] = []
        for page in rendered:
            file_id = persisted[page.page_number]
            file_ref = build_workspace_file_ref(
                workspace=workspace,
                file_path=page.path,
                file_id=file_id,
                mime_type=PNG_MIME_TYPE,
            )
            pages_payload.append(
                {
                    "page_number": page.page_number,
                    "width": page.width,
                    "height": page.height,
                    "file_id": file_id,
                    "file_ref": file_ref,
                    "artifacts": [build_inline_artifact(file_ref)],
                }
            )
        succeeded = True
        return {
            "success": True,
            "pptx_path": str(resolved_input),
            "output_dir": str(output_dir),
            **_selection_payload(selection),
            "pages": pages_payload,
            "artifacts": [
                artifact for page in pages_payload for artifact in page["artifacts"]
            ],
        }
    finally:
        try:
            await asyncio.to_thread(_remove_tree, temp_root)
        except BaseException:  # pragma: no cover - never mask the outcome
            logger.warning("Could not remove the render temp directory", exc_info=True)
        if not succeeded and rendered:
            try:
                persisted_pages = set(persisted)
                if not persisted_pages:
                    persisted_pages = set(
                        await _run_blocking(
                            lambda: _persisted_page_ids(workspace, rendered)
                        )
                    )
                await asyncio.to_thread(
                    _delete_unregistered_outputs, rendered, persisted_pages
                )
            except BaseException:  # pragma: no cover - never mask the outcome
                logger.warning(
                    "Could not clean up unregistered rendered slides",
                    exc_info=True,
                )
        elif not succeeded and output_dir is not None:
            try:
                await asyncio.to_thread(_remove_tree, output_dir)
            except BaseException:  # pragma: no cover
                logger.warning(
                    "Could not remove the empty render output directory",
                    exc_info=True,
                )
