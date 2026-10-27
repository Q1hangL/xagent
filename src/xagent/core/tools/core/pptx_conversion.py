"""Shared LibreOffice conversion from ``.pptx`` to a one-off PDF.

Both the ``.pptx`` preview cache and the slide render tool need the same
LibreOffice invocation contract, so the process handling lives here:

* a parameter list (never a shell command) with ``--headless`` and
  ``--convert-to pdf``;
* one LibreOffice user profile per call, so two concurrent conversions cannot
  fight over the same configuration lock;
* a bounded wait whose timeout kills and actually reaps the child process;
* cleanup of the profile and of the child on success, failure and
  cancellation.

Callers own what happens to the produced PDF: the preview keeps its managed
cache and publication guard, the render tool uses a disposable directory.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Final

logger = logging.getLogger(__name__)

DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS: Final[float] = 60.0
DEFAULT_SOFFICE_BINARY: Final[str] = "soffice"


class PptxConversionError(RuntimeError):
    """Base class for a ``.pptx`` -> PDF conversion failure."""


class PptxConverterUnavailableError(PptxConversionError):
    """LibreOffice is not installed or not reachable through PATH."""


class PptxConversionTimeoutError(PptxConversionError):
    """LibreOffice did not finish inside the configured timeout."""


class PptxConversionFailedError(PptxConversionError):
    """LibreOffice ran but did not produce a usable PDF."""


def build_soffice_arguments(
    pptx_path: Path,
    output_dir: Path,
    profile_dir: Path,
    *,
    soffice_binary: str = DEFAULT_SOFFICE_BINARY,
) -> list[str]:
    """Return the argv for one headless ``.pptx`` -> PDF conversion.

    ``profile_dir`` becomes ``-env:UserInstallation=file://...``: a writable
    LibreOffice configuration directory that is separate from the PDF output
    directory. It is what keeps concurrent conversions from serialising on (or
    corrupting) a shared profile lock.
    """
    return [
        soffice_binary,
        "--headless",
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]


def expected_pdf_path(pptx_path: Path, output_dir: Path) -> Path:
    """Return the PDF path LibreOffice is expected to create for a source file."""
    return Path(output_dir) / f"{Path(pptx_path).stem}.pdf"


def _produced_pdf_exists(pdf_path: Path) -> bool:
    """Return whether the expected output PDF exists on disk.

    Kept as a small module-level function so callers can run the filesystem
    check in a worker thread instead of on the event loop.
    """
    return Path(pdf_path).is_file()


_BACKGROUND_RECLAIMS: set[asyncio.Task[None]] = set()


async def _reclaim_created_directory(
    creation: "asyncio.Task[Path]",
) -> None:
    """Wait for a directory creation that outlived its caller and remove it."""
    try:
        created = await creation
    except BaseException:  # noqa: BLE001 - nothing was created to clean up
        return
    await asyncio.to_thread(shutil.rmtree, created, ignore_errors=True)


async def create_owned_directory(create: Callable[[], Path]) -> Path:
    """Create one directory in a worker thread and own it on cancellation.

    A cancelled ``asyncio.to_thread`` call does not stop the worker, so the
    directory can still appear after the caller has returned. Cancellation
    propagates immediately, while a background reclaim task waits for the
    worker, removes the directory it created, and then exits. On success the
    returned path is the caller's to clean up.
    """
    creation = asyncio.create_task(asyncio.to_thread(create))
    try:
        return await asyncio.shield(creation)
    except asyncio.CancelledError:
        reclaim = asyncio.create_task(_reclaim_created_directory(creation))
        _BACKGROUND_RECLAIMS.add(reclaim)
        reclaim.add_done_callback(_BACKGROUND_RECLAIMS.discard)
        raise


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Kill a conversion child and wait for it to be reaped."""
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:  # pragma: no cover - already gone
            pass
    try:
        await process.wait()
    except ProcessLookupError:  # pragma: no cover - already reaped
        pass


async def convert_pptx_to_pdf(
    pptx_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    soffice_binary: str = DEFAULT_SOFFICE_BINARY,
) -> Path:
    """Convert ``pptx_path`` into ``output_dir`` and return the produced PDF.

    The caller supplies the output directory and decides whether to keep or
    publish the PDF. A unique LibreOffice profile directory is created for this
    call and removed again on every exit path.

    Raises:
        PptxConverterUnavailableError: LibreOffice cannot be started.
        PptxConversionTimeoutError: the conversion exceeded ``timeout_seconds``;
            the child was killed and reaped.
        PptxConversionFailedError: a non-zero exit or no expected PDF.
    """
    pptx_path = Path(pptx_path)
    output_dir = Path(output_dir)
    profile_dir: Path | None = None
    try:
        profile_dir = await create_owned_directory(
            lambda: Path(tempfile.mkdtemp(prefix="xagent-pptx-profile-"))
        )
        arguments = build_soffice_arguments(
            pptx_path,
            output_dir,
            profile_dir,
            soffice_binary=soffice_binary,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise PptxConverterUnavailableError(
                f"LibreOffice is not installed or not on PATH ({soffice_binary!r}); "
                f"cannot convert {pptx_path.name} to PDF."
            ) from exc
        try:
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await _terminate_process(process)
            raise PptxConversionTimeoutError(
                f"LibreOffice conversion of {pptx_path.name} timed out after "
                f"{timeout_seconds:g} seconds."
            ) from exc
        except asyncio.CancelledError:
            await _terminate_process(process)
            raise
        if process.returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()[:200]
            raise PptxConversionFailedError(
                f"LibreOffice exited {process.returncode} while converting "
                f"{pptx_path.name}: {detail or 'no stderr output'}"
            )
        pdf_path = expected_pdf_path(pptx_path, output_dir)
        if not await asyncio.to_thread(_produced_pdf_exists, pdf_path):
            raise PptxConversionFailedError(
                f"LibreOffice finished but did not produce the expected PDF "
                f"{pdf_path.name} for {pptx_path.name}."
            )
        return pdf_path
    finally:
        if profile_dir is not None:
            await asyncio.to_thread(shutil.rmtree, profile_dir, ignore_errors=True)
