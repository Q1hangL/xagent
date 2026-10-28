"""Shared LibreOffice conversion from ``.pptx`` to a one-off PDF.

Both the ``.pptx`` preview cache and the slide render tool need the same
LibreOffice invocation contract, so the process handling lives here:

* a parameter list (never a shell command) with ``--headless`` and
  ``--convert-to pdf``;
* one LibreOffice user profile per call, so two concurrent conversions cannot
  fight over the same configuration lock;
* a bounded wait whose timeout terminates the isolated process group;
* cleanup of the profile and of the child on success, failure and
  cancellation.

Callers own what happens to the produced PDF: the preview keeps its managed
cache and publication guard, the render tool uses a disposable directory.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Final

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
    include_hidden_slides: bool = False,
) -> list[str]:
    """Return the argv for one headless ``.pptx`` -> PDF conversion.

    ``profile_dir`` becomes ``-env:UserInstallation=file://...``: a writable
    LibreOffice configuration directory that is separate from the PDF output
    directory. It is what keeps concurrent conversions from serialising on (or
    corrupting) a shared profile lock.
    """
    pdf_filter = (
        'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}'
        if include_hidden_slides
        else "pdf"
    )
    return [
        soffice_binary,
        "--headless",
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--convert-to",
        pdf_filter,
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]


def expected_pdf_path(pptx_path: Path, output_dir: Path) -> Path:
    """Return the PDF path LibreOffice is expected to create for a source file."""
    return Path(output_dir) / f"{Path(pptx_path).stem}.pdf"


PROCESS_REAP_TIMEOUT_SECONDS: Final[float] = 1.0


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    # A conversion owns its session. Kill descendants too, including after the
    # session leader exits while a descendant still holds the output pipes.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _communicate_with_group(
    process: asyncio.subprocess.Process,
) -> tuple[bytes, bytes]:
    communication = asyncio.create_task(process.communicate())
    try:
        while process.returncode is None and not communication.done():
            await asyncio.sleep(0.01)
        if process.returncode is not None and not communication.done():
            _kill_process_group(process)
        return await communication
    finally:
        if not communication.done():
            communication.cancel()
        await asyncio.gather(communication, return_exceptions=True)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate the isolated group and bound reaping, even with inherited pipes."""
    _kill_process_group(process)
    try:
        await asyncio.wait_for(process.wait(), PROCESS_REAP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        # asyncio exposes no public close for subprocess pipe transports.
        transport = getattr(process, "_transport", None)
        if transport is not None:
            transport.close()


async def convert_pptx_to_pdf(
    pptx_path: Path,
    output_dir: Path,
    *,
    timeout_seconds: float = DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS,
    soffice_binary: str = DEFAULT_SOFFICE_BINARY,
    include_hidden_slides: bool = False,
) -> Path:
    """Convert ``pptx_path`` into ``output_dir`` and return the produced PDF.

    The caller supplies the output directory and decides whether to keep or
    publish the PDF. A unique LibreOffice profile directory is created for this
    call and removed again on every exit path. Hidden slides are included only
    when explicitly requested; previews retain the default export behavior.

    Raises:
        PptxConverterUnavailableError: LibreOffice cannot be started.
        PptxConversionTimeoutError: the conversion exceeded ``timeout_seconds``;
            the process group was terminated and reaping was bounded.
        PptxConversionFailedError: a non-zero exit or no expected PDF.
    """
    pptx_path = Path(pptx_path)
    output_dir = Path(output_dir)
    profile_dir = Path(tempfile.mkdtemp(prefix="xagent-pptx-profile-"))
    process: asyncio.subprocess.Process | None = None
    try:
        arguments = build_soffice_arguments(
            pptx_path,
            output_dir,
            profile_dir,
            soffice_binary=soffice_binary,
            include_hidden_slides=include_hidden_slides,
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *arguments,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise PptxConverterUnavailableError(
                f"LibreOffice is not installed or not on PATH ({soffice_binary!r}); "
                f"cannot convert {pptx_path.name} to PDF."
            ) from exc
        try:
            _stdout, stderr = await asyncio.wait_for(
                _communicate_with_group(process), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise PptxConversionTimeoutError(
                f"LibreOffice conversion of {pptx_path.name} timed out after "
                f"{timeout_seconds:g} seconds."
            ) from exc
        if process.returncode != 0:
            detail = stderr.decode("utf-8", "replace").strip()[:200]
            raise PptxConversionFailedError(
                f"LibreOffice exited {process.returncode} while converting "
                f"{pptx_path.name}: {detail or 'no stderr output'}"
            )
        pdf_path = expected_pdf_path(pptx_path, output_dir)
        if not await asyncio.to_thread(pdf_path.is_file):
            raise PptxConversionFailedError(
                f"LibreOffice finished but did not produce the expected PDF "
                f"{pdf_path.name} for {pptx_path.name}."
            )
        return pdf_path
    finally:
        try:
            if process is not None:
                await _terminate_process(process)
        finally:
            await asyncio.to_thread(shutil.rmtree, profile_dir, ignore_errors=True)
