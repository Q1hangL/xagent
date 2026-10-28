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
import logging
import os
import shutil
import signal
import tempfile
import threading
from pathlib import Path
from typing import Callable, Final, Generic, TypeVar

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


def _produced_pdf_exists(pdf_path: Path) -> bool:
    """Return whether the expected output PDF exists on disk.

    Kept as a small module-level function so callers can run the filesystem
    check in a worker thread instead of on the event loop.
    """
    return Path(pdf_path).is_file()


_Resource = TypeVar("_Resource")


class _OwnedDirectoryCreation(Generic[_Resource]):
    """Keep cleanup on the creating thread until the caller accepts the path.

    A thread event acknowledges ownership transfer without an asyncio cleanup
    task. Cancellation or loop shutdown leaves ownership with the thread, which
    removes the directory before exiting.
    """

    def __init__(
        self,
        create: Callable[[], _Resource],
        cleanup: Callable[[_Resource], None],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._create = create
        self._cleanup = cleanup
        self._loop = loop
        self._future: asyncio.Future[_Resource] = loop.create_future()
        self._handoff_done = threading.Event()
        self._accepted = False
        self._thread = threading.Thread(
            target=self._run,
            name="xagent-owned-directory",
            daemon=False,
        )

    def start(self) -> None:
        self._thread.start()

    async def wait(self) -> _Resource:
        try:
            path = await self._future
        except BaseException:
            self._handoff_done.set()
            raise
        self._accepted = True
        self._handoff_done.set()
        return path

    def _run(self) -> None:
        try:
            path = self._create()
        except BaseException as exc:  # noqa: BLE001 - report to the waiter
            self._deliver(exc=exc)
            return
        if self._deliver(path=path):
            while not self._handoff_done.wait(0.1):
                if self._loop.is_closed():
                    break
        if not self._accepted:
            self._cleanup(path)

    def _deliver(
        self, *, path: _Resource | None = None, exc: BaseException | None = None
    ) -> bool:
        try:
            self._loop.call_soon_threadsafe(self._settle, path, exc)
        except RuntimeError:
            return False
        return True

    def _settle(self, path: _Resource | None, exc: BaseException | None) -> None:
        if self._future.done():
            return
        if exc is not None:
            self._future.set_exception(exc)
        elif path is not None:
            self._future.set_result(path)


async def create_owned_directory(create: Callable[[], Path]) -> Path:
    """Create one directory on a worker thread and own it on cancellation.

    On success the returned path is the caller's to clean up. If the caller is
    cancelled, the creating thread removes the directory itself, so cleanup
    survives cancellation of the event loop that started the call.
    """
    return await create_owned_resource(
        create, lambda path: shutil.rmtree(path, ignore_errors=True)
    )


async def create_owned_resource(
    create: Callable[[], _Resource], cleanup: Callable[[_Resource], None]
) -> _Resource:
    """Transfer a worker-created resource, keeping abandoned cleanup on its worker."""
    creation = _OwnedDirectoryCreation(create, cleanup, asyncio.get_running_loop())
    creation.start()
    return await creation.wait()


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
    waiter = asyncio.create_task(process.wait())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + PROCESS_REAP_TIMEOUT_SECONDS
    try:
        while not waiter.done():
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({waiter}, timeout=remaining)
            except asyncio.CancelledError:
                # Repeated cancellation must not bypass group teardown or
                # restart the fixed reap deadline.
                continue
        if waiter.done():
            waiter.result()
    finally:
        if not waiter.done():
            waiter.cancel()
            # asyncio exposes no public close for subprocess pipe transports.
            # A descendant that escaped the group must not hold our pipes open.
            transport = getattr(process, "_transport", None)
            if transport is not None:
                transport.close()
        await asyncio.gather(waiter, return_exceptions=True)


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
    profile_dir: Path | None = None
    process: asyncio.subprocess.Process | None = None
    try:
        profile_dir = await create_owned_directory(
            lambda: Path(tempfile.mkdtemp(prefix="xagent-pptx-profile-"))
        )
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
        if not await asyncio.to_thread(_produced_pdf_exists, pdf_path):
            raise PptxConversionFailedError(
                f"LibreOffice finished but did not produce the expected PDF "
                f"{pdf_path.name} for {pptx_path.name}."
            )
        return pdf_path
    finally:
        if process is not None:
            await _terminate_process(process)
        if profile_dir is not None:
            await asyncio.to_thread(shutil.rmtree, profile_dir, ignore_errors=True)
