"""Tests for the shared LibreOffice .pptx -> PDF conversion helper.

The helper is used by both the .pptx preview cache and the slide render tool,
so these tests pin the process contract (argv list, headless conversion,
per-call LibreOffice user profile, timeout, cancellation and cleanup) rather
than any caller's caching behavior.
"""

import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile
import threading

import pytest


def _load_conversion_module():
    try:
        from xagent.core.tools.core import pptx_conversion
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"shared pptx conversion module is missing: {exc}")
    return pptx_conversion


def _profile_dir_from_arguments(arguments: list[str]) -> pathlib.Path:
    """Return the user-installation directory named by the soffice argv."""
    for argument in arguments:
        if argument.startswith("-env:UserInstallation="):
            uri = argument.split("=", 1)[1]
            assert uri.startswith("file://"), uri
            return pathlib.Path(uri[len("file://") :])
    raise AssertionError(f"no -env:UserInstallation argument in {arguments!r}")


class _RecordingProcess:
    """Fake asyncio subprocess that records kill/wait/communicate usage."""

    def __init__(self, *, returncode: int = 0, stderr: bytes = b""):
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.stderr = stderr
        self.killed = 0
        self.wait_calls = 0
        self.communicate_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        self.returncode = self.final_returncode
        return b"", self.stderr

    def kill(self) -> None:
        self.killed += 1

    async def wait(self) -> int:
        self.wait_calls += 1
        self.returncode = -9
        return -9


class _HangingProcess(_RecordingProcess):
    """Fake subprocess whose communicate() never finishes on its own."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        self.started.set()
        # Bounded stand-in: the real per-call timeout used by the tests is much
        # shorter, and a mutant that drops the timeout must not stall the suite.
        await asyncio.sleep(5)
        return b"", b""


def _install_process_factory(monkeypatch, conversion, factory):
    calls: list[dict] = []

    async def create_subprocess_exec(*arguments, **kwargs):
        calls.append({"arguments": list(arguments), "kwargs": dict(kwargs)})
        return factory(calls)

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )
    return calls


def test_default_contract_constants_are_shared():
    conversion = _load_conversion_module()

    assert conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS == 60
    assert conversion.DEFAULT_SOFFICE_BINARY == "soffice"


def test_expected_pdf_path_uses_the_exact_source_stem(tmp_path):
    conversion = _load_conversion_module()

    assert conversion.expected_pdf_path(pathlib.Path("/d/deck.pptx"), tmp_path) == (
        tmp_path / "deck.pdf"
    )
    assert conversion.expected_pdf_path(pathlib.Path("/d/a.b.pptx"), tmp_path) == (
        tmp_path / "a.b.pdf"
    )


def test_build_arguments_use_parameter_list_without_shell_quoting(tmp_path):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck with spaces.pptx"
    output_dir = tmp_path / "out"
    profile_dir = tmp_path / "profile"

    arguments = conversion.build_soffice_arguments(pptx_path, output_dir, profile_dir)

    assert arguments == [
        "soffice",
        "--headless",
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]
    assert isinstance(arguments, list)
    assert profile_dir.as_uri().startswith("file://")


@pytest.mark.asyncio
async def test_conversion_passes_headless_convert_to_and_per_call_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    observed: dict[str, object] = {}

    process = _RecordingProcess()

    async def communicate():
        observed["profile_dir"] = _profile_dir_from_arguments(calls[0]["arguments"])
        observed["profile_existed_during_run"] = observed["profile_dir"].is_dir()
        (output_dir / "deck.pdf").write_bytes(b"%PDF-1.4 conversion")
        process.returncode = 0
        return b"", b""

    process.communicate = communicate  # type: ignore[method-assign]

    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    pdf_path = await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert pdf_path == output_dir / "deck.pdf"
    assert pdf_path.read_bytes() == b"%PDF-1.4 conversion"
    arguments = calls[0]["arguments"]
    assert arguments[0] == "soffice"
    assert arguments[1] == "--headless"
    assert arguments[3] == "--convert-to"
    assert arguments[4] == "pdf"
    assert arguments[5] == "--outdir"
    assert arguments[6] == str(output_dir)
    assert arguments[7] == str(pptx_path)
    assert calls[0]["kwargs"]["stdout"] == asyncio.subprocess.PIPE
    assert calls[0]["kwargs"]["stderr"] == asyncio.subprocess.PIPE
    # The user profile is a per-call directory, not the PDF output directory.
    assert observed["profile_existed_during_run"] is True
    assert observed["profile_dir"] != output_dir
    assert not observed["profile_dir"].exists()
    assert output_dir.is_dir()


@pytest.mark.asyncio
async def test_missing_soffice_raises_a_clear_dependency_error(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()

    async def create_subprocess_exec(*_arguments, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'soffice'")

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )

    with pytest.raises(conversion.PptxConverterUnavailableError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    message = str(fault.value)
    assert "soffice" in message
    assert "PATH" in message
    assert "deck.pptx" in message
    assert list(output_dir.iterdir()) == []
    leftovers = list(tmp_path.glob("xagent-pptx-profile-*"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_non_zero_exit_reports_stderr_and_cleans_the_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    observed: dict[str, object] = {}

    process = _RecordingProcess(returncode=1, stderr=b"libreoffice exploded")

    async def communicate():
        observed["profile_dir"] = _profile_dir_from_arguments(calls[0]["arguments"])
        process.returncode = 1
        return b"", b"libreoffice exploded"

    process.communicate = communicate  # type: ignore[method-assign]
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    message = str(fault.value)
    assert "deck.pptx" in message
    assert "libreoffice exploded" in message
    assert "1" in message
    assert not observed["profile_dir"].exists()


@pytest.mark.asyncio
async def test_success_without_the_expected_pdf_is_a_failure(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _RecordingProcess()
    _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert "deck.pdf" in str(fault.value)


@pytest.mark.asyncio
async def test_a_different_pdf_is_not_silently_accepted(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _RecordingProcess()

    async def communicate():
        (output_dir / "some-other.pdf").write_bytes(b"%PDF-1.4 stray")
        process.returncode = 0
        return b"", b""

    process.communicate = communicate  # type: ignore[method-assign]
    _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert "deck.pdf" in str(fault.value)


@pytest.mark.asyncio
async def test_timeout_kills_and_awaits_the_process_and_cleans_the_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _HangingProcess()
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionTimeoutError) as fault:
        await conversion.convert_pptx_to_pdf(
            pptx_path, output_dir, timeout_seconds=0.05
        )

    assert "0.05" in str(fault.value)
    assert "deck.pptx" in str(fault.value)
    assert process.killed == 1
    assert process.wait_calls == 1
    profile_dir = _profile_dir_from_arguments(calls[0]["arguments"])
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_cancellation_kills_and_awaits_the_process_then_cleans_up(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _HangingProcess()
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    task = asyncio.create_task(conversion.convert_pptx_to_pdf(pptx_path, output_dir))
    await asyncio.wait_for(process.started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed == 1
    assert process.wait_calls == 1
    profile_dir = _profile_dir_from_arguments(calls[0]["arguments"])
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_concurrent_conversions_get_isolated_profiles(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    output_one = tmp_path / "out-one"
    output_two = tmp_path / "out-two"
    output_one.mkdir()
    output_two.mkdir()
    first = tmp_path / "first.pptx"
    second = tmp_path / "second.pptx"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[dict] = []
    release = asyncio.Event()
    arrived = asyncio.Event()

    class _Process(_RecordingProcess):
        def __init__(self, output_dir: pathlib.Path, name: str):
            super().__init__()
            self.output_dir = output_dir
            self.name = name

        async def communicate(self):
            self.returncode = 0
            arrived.set()
            await release.wait()
            self.output_dir.joinpath(self.name).write_bytes(b"%PDF-1.4")
            return b"", b""

    async def create_subprocess_exec(*arguments, **kwargs):
        calls.append({"arguments": list(arguments), "kwargs": dict(kwargs)})
        output_dir = pathlib.Path(arguments[arguments.index("--outdir") + 1])
        return _Process(output_dir, pathlib.Path(arguments[-1]).stem + ".pdf")

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )

    both = asyncio.gather(
        conversion.convert_pptx_to_pdf(first, output_one),
        conversion.convert_pptx_to_pdf(second, output_two),
    )
    await asyncio.wait_for(arrived.wait(), timeout=5)
    release.set()
    results = await asyncio.wait_for(both, timeout=10)

    assert [path.name for path in results] == ["first.pdf", "second.pdf"]
    profiles = [_profile_dir_from_arguments(call["arguments"]) for call in calls]
    outdirs = [
        pathlib.Path(call["arguments"][call["arguments"].index("--outdir") + 1])
        for call in calls
    ]
    assert len({str(path) for path in profiles}) == 2
    assert len({str(path) for path in outdirs}) == 2
    for profile in profiles:
        assert not profile.exists()
    assert (output_one / "first.pdf").is_file()
    assert (output_two / "second.pdf").is_file()


@pytest.mark.asyncio
async def test_cancel_during_profile_creation_removes_the_created_directory(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    real_mkdtemp = tempfile.mkdtemp
    started = threading.Event()
    release = threading.Event()
    created: list[pathlib.Path] = []

    def delayed_mkdtemp(*args, **kwargs):
        if kwargs.get("prefix") != "xagent-pptx-profile-":
            return real_mkdtemp(*args, **kwargs)
        started.set()
        assert release.wait(5)
        kwargs["dir"] = str(tmp_path)
        path = pathlib.Path(real_mkdtemp(*args, **kwargs))
        created.append(path)
        return str(path)

    monkeypatch.setattr(tempfile, "mkdtemp", delayed_mkdtemp)
    task = asyncio.create_task(conversion.convert_pptx_to_pdf(pptx_path, output_dir))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        release.set()
        await asyncio.sleep(0.05)
        assert created, "the controlled profile directory was never created"
        assert all(not path.exists() for path in created), (
            "cancellation during profile creation leaked the directory"
        )
    finally:
        release.set()
        if not task.done():
            task.cancel()
        for path in created:
            import shutil

            shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_expected_pdf_existence_check_runs_off_the_event_loop(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    assert hasattr(conversion, "_produced_pdf_exists"), (
        "the shared helper must expose its produced-PDF check so it can run "
        "in a worker thread"
    )
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _RecordingProcess()
    loop_thread = threading.get_ident()
    observed: list[int] = []
    original = conversion._produced_pdf_exists

    def observed_exists(pdf_path):
        observed.append(threading.get_ident())
        return original(pdf_path)

    async def communicate():
        (output_dir / "deck.pdf").write_bytes(b"%PDF-1.4 conversion")
        process.returncode = 0
        return b"", b""

    process.communicate = communicate  # type: ignore[method-assign]
    _install_process_factory(monkeypatch, conversion, lambda _calls: process)
    monkeypatch.setattr(conversion, "_produced_pdf_exists", observed_exists)

    pdf_path = await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert pdf_path == output_dir / "deck.pdf"
    assert observed, "the produced-PDF check did not run"
    assert all(thread != loop_thread for thread in observed), (
        "the produced-PDF filesystem check blocked the event loop"
    )


def _active_source_root() -> str:
    import xagent

    return str(pathlib.Path(xagent.__path__[0]).resolve().parent)


def _lifecycle_child_environment(tmp_path) -> dict[str, str]:
    environment = dict(os.environ)
    source_root = _active_source_root()
    ambient = os.environ.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = ":".join(
        part for part in (source_root, ambient) if part
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
    tmp_dir = tmp_path / "tmpdir"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    environment["LIFECYCLE_DIR"] = str(lifecycle_root)
    environment["TMPDIR"] = str(tmp_dir)
    environment["HOME"] = str(tmp_path / "home")
    environment["XAGENT_STORAGE_ROOT"] = str(tmp_path / "storage")
    (tmp_path / "home").mkdir(exist_ok=True)
    return environment


def _run_lifecycle_child(script: str, tmp_path) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=str(pathlib.Path(_active_source_root()).parent),
        env=_lifecycle_child_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed


def _assert_no_pending_task_warnings(completed: subprocess.CompletedProcess) -> None:
    combined = completed.stdout + completed.stderr
    for marker in (
        "Task was destroyed but it is pending",
        "Task exception was never retrieved",
    ):
        assert marker not in combined, combined


_PROFILE_CREATION_CHILD = r"""
import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path

from xagent.core.tools.core import pptx_conversion as conversion

root = Path(os.environ["LIFECYCLE_DIR"])
tmpdir = Path(os.environ["TMPDIR"])
deck = root / "deck.pptx"
deck.write_bytes(b"pptx bytes")
output_dir = root / "pdf-out"
output_dir.mkdir()
started = threading.Event()
real_mkdtemp = tempfile.mkdtemp


def delayed_mkdtemp(*args, **kwargs):
    if kwargs.get("prefix") != "xagent-pptx-profile-":
        return real_mkdtemp(*args, **kwargs)
    started.set()
    time.sleep(0.3)
    return real_mkdtemp(*args, dir=str(tmpdir), **kwargs)


conversion.tempfile.mkdtemp = delayed_mkdtemp


async def main():
    task = asyncio.create_task(conversion.convert_pptx_to_pdf(deck, output_dir))
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


def test_cancelled_creation_cleanup_survives_asyncio_run_exit(tmp_path):
    completed = _run_lifecycle_child(_PROFILE_CREATION_CHILD, tmp_path)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CHILD_DONE" in completed.stdout
    leftover = [
        path
        for path in (tmp_path / "tmpdir").iterdir()
        if path.name.startswith("xagent-pptx-profile-")
    ]
    assert leftover == [], (
        "asyncio.run() shutdown left a profile directory behind: "
        f"{[str(path) for path in leftover]}"
    )
    _assert_no_pending_task_warnings(completed)


@pytest.mark.asyncio
async def test_cancelled_handoff_removes_directory_off_event_loop(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    scheduled = threading.Event()
    removed = threading.Event()
    observed: list[int] = []
    path = tmp_path / "owned"
    real_schedule = loop.call_soon_threadsafe
    real_remove = conversion.shutil.rmtree
    task = None

    def schedule(callback, *args, **kwargs):
        if threading.get_ident() != loop_thread and not scheduled.is_set():
            scheduled.set()
            real_schedule(task.cancel)
        return real_schedule(callback, *args, **kwargs)

    def remove(directory, *args, **kwargs):
        if pathlib.Path(directory) == path:
            observed.append(threading.get_ident())
        try:
            return real_remove(directory, *args, **kwargs)
        finally:
            removed.set()

    def create():
        path.mkdir()
        return path

    monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
    monkeypatch.setattr(conversion.shutil, "rmtree", remove)
    task = asyncio.create_task(conversion.create_owned_directory(create))
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await asyncio.to_thread(removed.wait, 5)
    assert not path.exists()
    assert observed and all(thread != loop_thread for thread in observed)
