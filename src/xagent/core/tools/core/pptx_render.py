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
import contextvars
import logging
import math
import shutil
import struct
import tempfile
import threading
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
    create_owned_directory,
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


class PptxRenderOutputError(PptxRenderError):
    error_code = "output_unavailable"


class PptxRenderRegistrationError(PptxRenderError):
    error_code = "registration_failed"

    def __init__(
        self,
        message: str,
        *,
        registered_pages: tuple[int, ...] = (),
        registered_file_ids: dict[int, str] | None = None,
        unverified_pages: tuple[int, ...] = (),
        unregistered_pages: tuple[int, ...] = (),
    ):
        super().__init__(message)
        self.registered_pages = registered_pages
        self.registered_file_ids = registered_file_ids or {}
        self.unverified_pages = unverified_pages
        self.unregistered_pages = unregistered_pages


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


def _f32(value: float) -> float:
    """Round a Python float to the single precision MuPDF uses internally."""
    packed = struct.pack("f", float(value))
    return float(struct.unpack("f", packed)[0])


def _pixmap_extent(length: float, scale: float) -> int:
    """Return a conservative pixel extent for one page dimension.

    MuPDF stores page geometry and matrices as 32-bit floats and rounds the
    rendered bounding box with a small epsilon. The returned value is an upper
    bound of what ``Page.get_pixmap`` will allocate, so a budget checked with
    it can never be exceeded by the real raster. Exact integers keep their
    value (an N-pixel page at scale 1 is allowed to use exactly N pixels).
    """
    raw = _f32(_f32(length) * _f32(scale))
    if not math.isfinite(raw):
        raise PptxRenderRasterError("The PDF page raster size is not finite.")
    nearest_integer = round(raw)
    if abs(raw - nearest_integer) <= 1e-9:
        return max(1, int(nearest_integer))
    fraction = raw - math.floor(raw)
    if fraction <= 0.002:
        # MuPDF rounds with a ~0.001 epsilon, so a value just above an
        # integer can still round down; use the next whole pixel as the
        # conservative upper bound instead of under-predicting.
        return max(1, math.floor(raw) + 1)
    return max(1, math.ceil(raw))


def compute_page_geometry(
    page_width: float,
    page_height: float,
    *,
    scale: float = DEFAULT_RENDER_SCALE,
    max_pixels: int = MAX_PAGE_PIXELS,
) -> PageGeometry:
    """Decide the raster scale and rounded pixel size for one PDF page.

    A bounded bisection finds the largest scale whose conservative rounded
    width x height product fits ``max_pixels``. The search always progresses
    (it halves the scale interval instead of nudging by a rounding-dependent
    factor), so a page that can legally be rendered is never rejected for a
    rounding plateau. The aspect ratio is preserved because one scale is
    shared by both dimensions. Called before the pixmap exists, so an
    oversized page never allocates an oversized raster.
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

    def extents(candidate_scale: float) -> tuple[int, int]:
        return (
            _pixmap_extent(float(page_width), candidate_scale),
            _pixmap_extent(float(page_height), candidate_scale),
        )

    def fits(candidate_scale: float) -> bool:
        width, height = extents(candidate_scale)
        return width * height <= max_pixels

    requested_scale = float(scale)
    if fits(requested_scale):
        width, height = extents(requested_scale)
        return PageGeometry(scale=requested_scale, width=width, height=height)
    lower_scale = 0.0
    upper_scale = requested_scale
    for _attempt in range(80):
        middle_scale = (lower_scale + upper_scale) / 2.0
        if middle_scale <= 0:
            break
        if fits(middle_scale):
            lower_scale = middle_scale
        else:
            upper_scale = middle_scale
    width, height = extents(lower_scale)
    if lower_scale <= 0 or width * height > max_pixels:
        raise PptxRenderRasterError(
            "Could not reduce the page raster below the pixel budget."
        )
    return PageGeometry(scale=lower_scale, width=width, height=height)


def _mupdf_pixel_extents(page: Any, scale: float) -> tuple[int, int]:
    """Return the exact pixel extents MuPDF will use for one page and scale."""
    if fitz is None:  # pragma: no cover - guarded by _open_pdf
        raise PptxRenderRasterError("PyMuPDF is not available.")
    rectangle = (page.rect * fitz.Matrix(scale, scale)).irect
    return max(1, int(rectangle.width)), max(1, int(rectangle.height))


def _fit_scale_to_page(page: Any, requested_scale: float, max_pixels: int) -> float:
    """Find the largest scale whose MuPDF-rounded raster fits the budget.

    ``compute_page_geometry`` only predicts the extent from the page
    rectangle; this check uses MuPDF's own rounding on the real page and is
    still run before any pixmap is allocated. A bisection guarantees progress
    even when the rounding keeps the extent on a plateau.
    """

    def fits(candidate_scale: float) -> bool:
        width, height = _mupdf_pixel_extents(page, candidate_scale)
        return width * height <= max_pixels

    if fits(requested_scale):
        return requested_scale
    lower_scale = 0.0
    upper_scale = float(requested_scale)
    for _attempt in range(80):
        middle_scale = (lower_scale + upper_scale) / 2.0
        if middle_scale <= 0:
            break
        if fits(middle_scale):
            lower_scale = middle_scale
        else:
            upper_scale = middle_scale
    if lower_scale <= 0 or not fits(lower_scale):
        raise PptxRenderRasterError(
            "Could not reduce the page raster below the pixel budget."
        )
    return lower_scale


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
            effective_scale = _fit_scale_to_page(page, geometry.scale, max_pixels)
            matrix = fitz.Matrix(effective_scale, effective_scale)
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


@dataclass(frozen=True)
class _RegistrationState:
    """Result of looking up the durable file records for rendered PNGs.

    A page is only reported as unregistered when a working metadata lookup
    definitively found no record. Lookup errors are "unverified" and must
    never authorize deleting the local image.
    """

    registered: dict[int, str]
    unregistered: set[int]
    unverified: set[int]


def _lookup_persisted_file_id(path: Path) -> str | None:
    """Return the persisted file id for one local path, or raise on failure.

    ``TaskWorkspace.get_registered_file_id`` intentionally swallows database
    faults and reports them as "no record"; that is indistinguishable from a
    really missing record. This lookup keeps the metadata fault visible so the
    render cleanup can distinguish unregistered from unverifiable.
    """
    from ....web.models.uploaded_file import UploadedFile
    from ...storage.manager import create_db_session

    database = create_db_session()
    try:
        record = (
            database.query(UploadedFile)
            .filter(UploadedFile.storage_path == str(Path(path).resolve()))
            .first()
        )
        if record is None:
            return None
        return str(record.file_id)
    finally:
        database.close()


@dataclass(frozen=True)
class _PersistedFileRecord:
    """Detached metadata snapshot for one registered file id."""

    file_id: str
    user_id: int | None
    task_id: int | None
    storage_status: str | None
    storage_path: str | None
    storage_key: str | None
    detached_reason: str | None


def _lookup_uploaded_file_record(file_id: str) -> _PersistedFileRecord | None:
    """Return the metadata snapshot for one file id, or None when absent.

    Metadata faults propagate so verification can distinguish "record absent"
    from "record unreadable".
    """
    from ....web.models.uploaded_file import UploadedFile
    from ...storage.manager import create_db_session

    database = create_db_session()
    try:
        record = (
            database.query(UploadedFile)
            .filter(UploadedFile.file_id == str(file_id))
            .first()
        )
        if record is None:
            return None
        return _PersistedFileRecord(
            file_id=str(record.file_id),
            user_id=int(record.user_id) if record.user_id is not None else None,
            task_id=int(record.task_id) if record.task_id is not None else None,
            storage_status=(
                str(record.storage_status)
                if record.storage_status is not None
                else None
            ),
            storage_path=str(record.storage_path) if record.storage_path else None,
            storage_key=str(record.storage_key) if record.storage_key else None,
            detached_reason=getattr(record, "detached_reason", None),
        )
    finally:
        database.close()


def _workspace_is_delegated(workspace: TaskWorkspace) -> bool:
    """Whether registration may materialise files under a parent task root.

    Delegated workspaces keep their own execution directory but persist files
    under the parent task's canonical output directory, so a path-based lookup
    of the original rendered image legitimately misses.
    """
    db_task_id = getattr(workspace, "db_task_id", None)
    if db_task_id is None:
        return False
    try:
        parsed_task_id = int(str(getattr(workspace, "id", "")).split("_")[-1])
    except (TypeError, ValueError):
        parsed_task_id = None
    return parsed_task_id != int(db_task_id)


def _expected_registration_owner(workspace: TaskWorkspace) -> tuple[int, int]:
    """Return (owner user id, task id) that persisted records must belong to."""
    from ....web.models.task import Task
    from ...storage.manager import create_db_session

    expected_task_id = getattr(workspace, "db_task_id", None) or getattr(
        workspace, "current_task_id", None
    )
    if expected_task_id is None:
        raise PptxRenderRegistrationError(
            "Cannot verify slide registration without a persistent task owner."
        )
    database = create_db_session()
    try:
        row = (
            database.query(Task.user_id)
            .filter(Task.id == int(expected_task_id))
            .first()
        )
    finally:
        database.close()
    if row is None:
        raise PptxRenderRegistrationError(
            "Cannot verify slide registration: the owning task record is missing."
        )
    return int(row.user_id), int(expected_task_id)


def _persisted_record_problem(
    record: _PersistedFileRecord,
    *,
    workspace: TaskWorkspace,
    page: RenderedPage,
    expected_user_id: int,
    expected_task_id: int,
) -> str | None:
    """Validate ownership and the page's local or delegated canonical path."""
    if record.detached_reason is not None:
        return "the file record is detached"
    if record.storage_status not in {"available", "legacy"}:
        return f"the file storage status is {record.storage_status!r}"
    if record.user_id != expected_user_id:
        return "the file record belongs to another owner"
    if record.task_id != expected_task_id:
        return "the file record belongs to another task"
    if not record.storage_path:
        return "the file record has no canonical local path"
    try:
        stored_path = Path(record.storage_path).resolve(strict=True)
        if not stored_path.is_file():
            return "the canonical path is not a file"
        source = page.path.resolve(strict=True)
        if not _workspace_is_delegated(workspace):
            if stored_path == source:
                return None
        else:
            # Match the existing workspace materialization layout, including
            # its numeric filename suffix when a canonical file already exists.
            parent_root = (
                workspace._user_workspace_base_dir(expected_user_id)
                / f"web_task_{expected_task_id}"
                / "output"
            ).resolve(strict=True)
            relative = source.relative_to(workspace.output_dir.resolve(strict=True))
            expected = parent_root / relative
            suffix = stored_path.stem.removeprefix(f"{source.stem}_")
            matching_name = stored_path.name == source.name or (
                stored_path.stem.startswith(f"{source.stem}_")
                and suffix.isascii()
                and suffix.isdecimal()
                and int(suffix) > 0
                and stored_path.suffix == source.suffix
            )
            if (
                stored_path.is_relative_to(parent_root)
                and stored_path.parent == expected.parent
                and matching_name
            ):
                return None
    except (OSError, RuntimeError, ValueError):
        return "the canonical path could not be verified"
    return "the file record does not point to this slide's canonical path"


def _verify_registrations_by_path(
    workspace: TaskWorkspace, rendered: list[RenderedPage]
) -> _RegistrationState:
    """Path-based fallback used when registration raised before returning ids.

    The workspace API answers the normal case. Because it reports metadata
    faults as "no record", a miss is re-checked with an exception-preserving
    lookup: only a working lookup that still finds nothing downgrades the page
    to unregistered, while a fault keeps it unverified. In a delegated
    workspace a path miss is not proof of a missing registration, because the
    persisted copy lives under the parent task directory.
    """
    registered: dict[int, str] = {}
    unregistered: set[int] = set()
    unverified: set[int] = set()
    delegated = _workspace_is_delegated(workspace)
    for page in rendered:
        try:
            known_file_id = workspace.get_registered_file_id(str(page.path))
        except Exception:  # noqa: BLE001 - explicit tri-state classification
            known_file_id = None
        if known_file_id:
            registered[page.page_number] = str(known_file_id)
            continue
        try:
            file_id = _lookup_persisted_file_id(page.path)
        except Exception:  # noqa: BLE001 - explicit tri-state classification
            logger.warning(
                "Could not verify registration of rendered slide %s",
                page.path,
                exc_info=True,
            )
            unverified.add(page.page_number)
            continue
        if file_id:
            registered[page.page_number] = file_id
        elif delegated:
            unverified.add(page.page_number)
        else:
            unregistered.add(page.page_number)
    return _RegistrationState(
        registered=registered,
        unregistered=unregistered,
        unverified=unverified,
    )


def _verify_returned_registrations(
    workspace: TaskWorkspace,
    rendered: list[RenderedPage],
    registered_ids: tuple[str, ...],
) -> tuple[_RegistrationState, dict[int, str]]:
    """Classify pages using the ids registration returned and their records.

    A returned id is checked against the persisted record's owner, task,
    storage status and canonical location. In a delegated workspace the
    canonical location is the parent task's output copy, so matching the
    original render path is neither required nor attempted.
    """
    registered: dict[int, str] = {}
    unregistered: set[int] = set()
    unverified: set[int] = set()
    problems: dict[int, str] = {}

    try:
        expected_user_id, expected_task_id = _expected_registration_owner(workspace)
    except Exception:  # noqa: BLE001 - metadata lookup fault on the owner task
        logger.warning(
            "Could not load the owning task for registration verification",
            exc_info=True,
        )
        return (
            _RegistrationState({}, set(), {page.page_number for page in rendered}),
            problems,
        )

    for page, registered_id in zip(rendered, registered_ids):
        returned_id = str(registered_id)
        try:
            record = _lookup_uploaded_file_record(returned_id)
        except Exception:  # noqa: BLE001 - explicit tri-state classification
            unverified.add(page.page_number)
            continue
        if record is None:
            # A missing returned id does not prove that registration did not
            # commit under a different id or a delegated canonical path.
            fallback = _verify_registrations_by_path(workspace, [page])
            if page.page_number in fallback.unregistered:
                unregistered.add(page.page_number)
            else:
                unverified.add(page.page_number)
            continue
        problem = _persisted_record_problem(
            record,
            workspace=workspace,
            page=page,
            expected_user_id=expected_user_id,
            expected_task_id=expected_task_id,
        )
        if problem is None:
            registered[page.page_number] = returned_id
        else:
            unverified.add(page.page_number)
            problems[page.page_number] = problem
    return (
        _RegistrationState(
            registered=registered,
            unregistered=unregistered,
            unverified=unverified,
        ),
        problems,
    )


def _register_rendered_pages(
    workspace: TaskWorkspace, rendered: list[RenderedPage]
) -> dict[int, str]:
    """Batch-register every PNG and verify each returned id against its record."""
    paths = [str(page.path) for page in rendered]
    try:
        registered_ids = tuple(
            str(file_id)
            for file_id in workspace.register_files([(path, None) for path in paths])
        )
    except Exception as exc:
        state = _verify_registrations_by_path(workspace, rendered)
        raise PptxRenderRegistrationError(
            f"Rendered slide images could not be registered durably: {exc}",
            registered_pages=tuple(sorted(state.registered)),
            registered_file_ids=dict(state.registered),
            unverified_pages=tuple(sorted(state.unverified)),
            unregistered_pages=tuple(sorted(state.unregistered)),
        ) from exc
    if len(registered_ids) != len(rendered):
        state = _verify_registrations_by_path(workspace, rendered)
        raise PptxRenderRegistrationError(
            f"Only {len(registered_ids)} of {len(rendered)} rendered slide "
            "images received a file id.",
            registered_pages=tuple(sorted(state.registered)),
            registered_file_ids=dict(state.registered),
            unverified_pages=tuple(sorted(state.unverified)),
            unregistered_pages=tuple(sorted(state.unregistered)),
        )
    state, problems = _verify_returned_registrations(
        workspace, rendered, registered_ids
    )
    returned_ids = {
        page.page_number: str(file_id)
        for page, file_id in zip(rendered, registered_ids)
    }
    known_registered = dict(state.registered)
    for page_number in state.unverified:
        known_registered.setdefault(page_number, returned_ids[page_number])
    if state.unregistered:
        raise PptxRenderRegistrationError(
            "Rendered slide images were not persisted as durable files for "
            f"page(s) {', '.join(str(number) for number in sorted(state.unregistered))}; "
            "their file ids would not be reusable in a later turn.",
            registered_pages=tuple(sorted(known_registered)),
            registered_file_ids=dict(known_registered),
            unverified_pages=tuple(sorted(state.unverified)),
            unregistered_pages=tuple(sorted(state.unregistered)),
        )
    if state.unverified:
        detail = "".join(
            f" page {number}: {reason};" for number, reason in sorted(problems.items())
        )
        raise PptxRenderRegistrationError(
            "Registration could not be verified for page(s) "
            f"{', '.join(str(number) for number in sorted(state.unverified))}; "
            "the images were kept because their registration may have "
            f"committed, but this call cannot claim delivery.{detail}",
            registered_pages=tuple(sorted(known_registered)),
            registered_file_ids=dict(known_registered),
            unverified_pages=tuple(sorted(state.unverified)),
            unregistered_pages=(),
        )
    return state.registered


def _delete_unregistered_outputs(
    rendered: list[RenderedPage], unregistered_pages: set[int]
) -> None:
    """Remove this call's PNGs that are definitively not registered."""
    for page in rendered:
        if page.page_number not in unregistered_pages:
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


def _create_output_directory(output_root: Path) -> Path:
    """Exclusively create this call's unique PNG directory under ``output_root``.

    ``mkdir`` without ``exist_ok`` makes ownership unambiguous: the directory
    either did not exist before this call or creation fails. A pre-existing
    directory is never adopted or deleted.
    """
    for _attempt in range(8):
        candidate = output_root / (f"{RENDER_OUTPUT_DIR_PREFIX}{uuid.uuid4().hex[:12]}")
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        except OSError as exc:
            raise PptxRenderOutputError(
                f"Could not create the slide image directory: {exc}"
            ) from exc
        return candidate
    raise PptxRenderOutputError(
        "Could not create a fresh slide image directory under the workspace "
        "output directory."
    )


def _prepare_output_directory(workspace: TaskWorkspace) -> Path:
    """Resolve, create and validate this call's output directory.

    This is blocking filesystem work and runs on a worker thread. The exclusive
    creation and every validation step happen while the creating thread still
    owns cleanup, so a cancellation during any of them cannot leak the
    directory.
    """
    workspace_root = Path(workspace.workspace_dir).resolve()
    output_root = Path(workspace.output_dir)
    try:
        resolved_output_root = output_root.resolve(True)
    except OSError as exc:
        raise PptxRenderOutputError(
            f"The workspace output directory is not available: {exc}"
        ) from exc
    if not resolved_output_root.is_relative_to(workspace_root):
        raise PptxRenderOutputError(
            "The workspace output directory resolves outside the workspace; "
            "refusing to write slide images there."
        )
    if not resolved_output_root.is_dir():
        raise PptxRenderOutputError(
            "The workspace output directory is not a directory."
        )
    created = _create_output_directory(resolved_output_root)
    try:
        try:
            resolved_created = created.resolve(True)
        except OSError as exc:
            raise PptxRenderOutputError(
                f"Could not verify the created slide image directory: {exc}"
            ) from exc
        if (
            resolved_created.parent != resolved_output_root
            or not resolved_created.is_relative_to(workspace_root)
        ):
            raise PptxRenderOutputError(
                "The created slide image directory is not directly inside the "
                "workspace output directory."
            )
        return resolved_created
    except BaseException:
        shutil.rmtree(created, ignore_errors=True)
        raise


async def _create_render_output_directory(workspace: TaskWorkspace) -> Path:
    """Create this call's output directory without doing filesystem I/O on the loop."""
    try:
        return await create_owned_directory(
            lambda: _prepare_output_directory(workspace)
        )
    except PptxRenderError:
        raise
    except OSError as exc:  # pragma: no cover - defensive fallback
        raise PptxRenderOutputError(
            f"Could not create the slide image directory: {exc}"
        ) from exc


def _remove_tree(path: Path | None) -> None:
    if path is None:
        return
    shutil.rmtree(path, ignore_errors=True)


async def _run_blocking(operation: Callable[[], _T]) -> _T:
    """Drain the actual worker before cancellation can trigger output cleanup."""
    completed = threading.Event()
    context = contextvars.copy_context()

    def run() -> _T:
        try:
            return context.run(operation)
        finally:
            completed.set()

    # Submit directly so normal loop shutdown cannot cancel a proxy Task before
    # the worker has finished. Only the worker signals completion.
    future = asyncio.get_running_loop().run_in_executor(None, run)
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not completed.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                continue
        # The thread can signal completion just before its result reaches the
        # loop. Consume that result too, including exceptions after a commit.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not future.cancelled():
            future.exception()
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


async def _cleanup_failed_render(
    workspace: TaskWorkspace,
    output_dir: Path,
    rendered: list[RenderedPage],
    registration_state: _RegistrationState | None,
) -> None:
    """Remove only this call's unregistered images after a failed render.

    Registered pages and pages whose registration could not be verified are
    kept: a metadata outage is not evidence that the image is unowned. When no
    page needs keeping, the call-owned output directory itself is removed.
    """
    if not rendered:
        try:
            await asyncio.to_thread(_remove_tree, output_dir)
        except BaseException:  # pragma: no cover - never mask the outcome
            logger.warning(
                "Could not remove the empty render output directory",
                exc_info=True,
            )
        return
    state = registration_state
    if state is None:
        try:
            state = await _run_blocking(
                lambda: _verify_registrations_by_path(workspace, rendered)
            )
        except BaseException:  # pragma: no cover - keep files if unclassifiable
            logger.warning(
                "Could not classify rendered slide registrations during "
                "cleanup; keeping the images",
                exc_info=True,
            )
            return
    try:
        await asyncio.to_thread(
            _delete_unregistered_outputs, rendered, set(state.unregistered)
        )
    except BaseException:  # pragma: no cover - never mask the outcome
        logger.warning("Could not clean up unregistered rendered slides", exc_info=True)
    kept_pages = len(rendered) - len(state.unregistered)
    if kept_pages == 0:
        try:
            await asyncio.to_thread(_remove_tree, output_dir)
        except BaseException:  # pragma: no cover - never mask the outcome
            logger.warning(
                "Could not remove the empty render output directory",
                exc_info=True,
            )


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
    output_owned = False
    rendered: list[RenderedPage] = []
    persisted: dict[int, str] = {}
    registration_state: _RegistrationState | None = None
    succeeded = False
    try:
        try:
            temp_root = await create_owned_directory(
                lambda: Path(tempfile.mkdtemp(prefix="xagent-pptx-render-"))
            )
            pdf_dir = temp_root / "pdf"
            await asyncio.to_thread(pdf_dir.mkdir)
        except OSError as exc:
            return _error(
                "raster_failed",
                f"Could not create a temporary render directory: {exc}",
            )

        try:
            output_dir = await _create_render_output_directory(workspace)
        except PptxRenderError as exc:
            return _error(exc.error_code, str(exc))
        except OSError as exc:  # pragma: no cover - defensive fallback
            return _error(
                "output_unavailable",
                f"Could not create the slide image directory: {exc}",
            )
        output_owned = True

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
            registration_state = _RegistrationState(
                registered=dict(exc.registered_file_ids),
                unregistered=set(exc.unregistered_pages),
                unverified=set(exc.unverified_pages),
            )
            return _error(
                "registration_failed",
                str(exc),
                registered_pages=list(exc.registered_pages),
                unverified_pages=list(exc.unverified_pages),
                unregistered_pages=list(exc.unregistered_pages),
                rendered_pages=len(rendered),
            )

        try:
            pages_payload = await _run_blocking(
                lambda: _build_pages_payload(workspace, rendered, persisted)
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "Could not build FileRefs for rendered slides", exc_info=True
            )
            registration_state = _RegistrationState(
                registered=dict(persisted),
                unregistered=set(),
                unverified=set(),
            )
            return _error(
                "registration_failed",
                f"Could not build FileRefs for the rendered slides: {exc}",
                registered_pages=sorted(persisted),
                unverified_pages=[],
                unregistered_pages=[],
                rendered_pages=len(rendered),
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
        if not succeeded and output_owned and output_dir is not None:
            try:
                await _cleanup_failed_render(
                    workspace, output_dir, rendered, registration_state
                )
            except BaseException:  # pragma: no cover - never mask the outcome
                logger.warning(
                    "Could not clean up the failed render output",
                    exc_info=True,
                )
