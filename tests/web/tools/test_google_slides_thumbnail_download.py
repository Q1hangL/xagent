"""Shared transport and publication contracts for slide thumbnails."""

import asyncio
import inspect
import io
import json
import logging
import os
import socket
from contextlib import asynccontextmanager
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from xagent.core.utils import security
from xagent.web.tools.mcp import google_slides as slides

HOST = "lh3.googleusercontent.com"
URL = f"https://{HOST}/signed-image?arbitrary=synthetic-private-query"
_buffer = io.BytesIO()
Image.new("RGB", (2, 3), (10, 20, 30)).save(_buffer, format="PNG")
PNG = _buffer.getvalue()


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "XAGENT_TRUSTED_EGRESS_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)


@pytest.fixture
def api(monkeypatch):
    service = Mock()
    presentations = service.presentations.return_value
    presentations.get.return_value.execute.return_value = {
        "slides": [{"objectId": "s1"}]
    }
    thumbnail = presentations.pages.return_value.getThumbnail
    thumbnail.return_value.execute.return_value = {"contentUrl": URL}
    monkeypatch.setattr(slides, "get_slides_service", lambda: service)
    return presentations, thumbnail


async def invoke(**kwargs):
    result = slides.google_slides_get_slide_thumbnail(presentation_id="pres1", **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return json.loads(result)


def test_thumbnail_entry_is_async():
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)


async def test_unknown_slide_is_rejected_before_thumbnail_request(api):
    result = await invoke(slide_id="missing-slide")
    assert result["status"] == "error"
    assert "slide_id" in result["message"] and "not found" in result["message"]
    api[1].assert_not_called()


async def test_locator_guidance_link_is_preserved():
    result = slides.google_slides_get_slide_thumbnail(
        presentation_id="file:synthetic", slide_id="s1"
    )
    if inspect.isawaitable(result):
        result = await result
    result = json.loads(result)
    assert result["status"] == "error"
    assert "https://docs.google.com/presentation/d/..." in result["message"]


def test_write_publishes_only_complete_png_and_cleans_failed_temp(
    monkeypatch, tmp_path
):
    _, output, canonical, fd = slides._open_thumbnail_output_directory()
    original = os.fdopen
    observed = []

    class BrokenWrite:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, data):
            self.handle.write(data[:8])
            self.handle.flush()
            observed.extend(output.glob("*.png"))
            raise OSError("write interrupted")

    monkeypatch.setattr(os, "fdopen", lambda *a, **kw: BrokenWrite(original(*a, **kw)))
    try:
        with pytest.raises(Exception, match="could not save"):
            slides._write_thumbnail_png(
                PNG,
                output_dir=output,
                canonical_output_dir=canonical,
                dir_fd=fd,
                filename="slide.png",
            )
    finally:
        os.close(fd)
    assert observed == []
    assert list(output.iterdir()) == []


@pytest.fixture
def transport(monkeypatch):
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)
    state = {
        "requests": [],
        "response": lambda request: httpx.Response(200, content=PNG),
    }

    async def resolve(url):
        return ["93.184.216.34"]

    def respond(request):
        state["requests"].append(request)
        return state["response"](request)

    @asynccontextmanager
    async def client(proxy, **kwargs):
        state["proxy"] = proxy
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as value:
            yield value

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "public_fetch_client", client)
    return state


async def test_identity_encoding_is_requested_and_compression_rejected(
    api, transport, tmp_path
):
    transport["response"] = lambda request: httpx.Response(
        200,
        headers={"content-encoding": "gzip"},
        stream=httpx.ByteStream(b"compressed"),
    )
    result = await invoke(slide_id="s1")
    assert transport["requests"][0].headers["accept-encoding"] == "identity"
    assert result["status"] == "error"
    assert "encoding" in result["message"]
    assert not list((tmp_path / "output").iterdir())


async def test_pixel_limit_rejects_before_decode(api, transport, monkeypatch, tmp_path):
    monkeypatch.setattr(slides, "_THUMBNAIL_MAX_PIXEL_COUNT", 5)
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert "unsupported size" in result["message"]
    assert not list((tmp_path / "output").iterdir())


@pytest.mark.parametrize("scheme", ["HTTPS", "hTtPs"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, httpx.RequestError])
async def test_download_errors_redact_whole_urls(
    api, transport, scheme, error_type, caplog
):
    def fail(request):
        raise error_type(
            f"cannot fetch {scheme}://{HOST}/secret-path?arbitrary=private-value"
        )

    transport["response"] = fail
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    for secret in (HOST, "secret-path", "private-value"):
        assert secret not in json.dumps(result) + caplog.text


async def test_api_guidance_is_not_redacted(api):
    guidance = "https://docs.google.com/presentation/d/..."
    api[1].return_value.execute.side_effect = ValueError(f"Open {guidance}")
    result = await invoke(slide_id="s1")
    assert guidance in result["message"]


async def test_redirect_cookies_are_removed(api, transport):
    def respond(request):
        if len(transport["requests"]) == 1:
            return httpx.Response(
                302,
                headers={
                    "location": URL + "&next=1",
                    "set-cookie": "session=synthetic; Path=/",
                },
            )
        return httpx.Response(200, content=PNG)

    transport["response"] = respond
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    assert len(transport["requests"]) == 2
    for request in transport["requests"]:
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers


@pytest.mark.parametrize("private_ip", ["127.0.0.1", "10.0.0.1", "::1"])
async def test_real_download_rejects_private_dns_before_connect(
    api, monkeypatch, private_ip
):
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)
    resolved = []

    def resolve(host, port, *args):
        resolved.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (private_ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert (
        "public" in result["message"].lower() or "private" in result["message"].lower()
    )
    assert resolved == [HOST]


@pytest.fixture
async def loopback(monkeypatch):
    assert inspect.iscoroutinefunction(slides.google_slides_get_slide_thumbnail)
    seen, pinned, tasks = [], [], set()
    closed = asyncio.Event()
    release = asyncio.Event()
    state = {
        "mode": "ok",
        "seen": seen,
        "pinned": pinned,
        "closed": closed,
        "release": release,
    }

    async def serve(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            seen.append(request)
            if state["mode"] == "stall":
                await reader.read()
            else:
                writer.write(
                    f"HTTP/1.1 200 OK\r\nContent-Length: {len(PNG)}\r\nConnection: close\r\n\r\n".encode()
                )
                if state["mode"] == "body":
                    writer.write(PNG[:8])
                    await writer.drain()
                    await reader.read()
                else:
                    await release.wait()
                    writer.write(PNG)
                    await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)
            closed.set()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    class LocalTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            pinned.append(str(request.url))
            local = httpx.Request(
                request.method,
                request.url.copy_with(scheme="http", host="127.0.0.1", port=port),
                headers=request.headers,
                extensions=request.extensions,
            )
            return await self.inner.handle_async_request(local)

        async def aclose(self):
            await self.inner.aclose()

    async def resolve(url):
        return ["93.184.216.34"]

    @asynccontextmanager
    async def client(proxy, **kwargs):
        async with httpx.AsyncClient(transport=LocalTransport()) as value:
            yield value

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "public_fetch_client", client)
    try:
        yield state
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*list(tasks), return_exceptions=True)


@pytest.mark.parametrize("log_threads", [True, False])
async def test_loopback_logs_hide_pinned_signed_url_but_preserve_other_tasks(
    api, loopback, caplog, monkeypatch, log_threads
):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="httpx")
    with monkeypatch.context() as patch:
        patch.setattr(logging, "logThreads", log_threads)
        task = asyncio.create_task(invoke(slide_id="s1"))
        while not loopback["seen"]:
            await asyncio.sleep(0)
        other = "HTTP Request: GET https://unrelated.example/visible?value=original"
        logging.getLogger("httpx").info(other)
        loopback["release"].set()
        result = await task
    assert result["status"] == "success"
    assert any(
        "93.184.216.34" in url and "synthetic-private-query" in url
        for url in loopback["pinned"]
    )
    assert b"synthetic-private-query" in loopback["seen"][0]
    assert "synthetic-private-query" not in caplog.text
    assert other in caplog.text


@pytest.mark.parametrize("phase", ["stall", "body"])
async def test_loopback_deadline_closes_connection(
    api, loopback, monkeypatch, phase, tmp_path
):
    loopback["mode"] = phase
    monkeypatch.setattr(slides, "_THUMBNAIL_TIMEOUT_SECONDS", 0.2)
    result = await asyncio.wait_for(invoke(slide_id="s1"), 5)
    assert result["status"] == "error"
    assert "timed out" in result["message"]
    await asyncio.wait_for(loopback["closed"].wait(), 5)
    assert not list((tmp_path / "output").iterdir())


@pytest.mark.parametrize("length", ["+77", "-1", "²", "٢", " 77"])
async def test_declared_length_requires_ascii_digits(api, transport, length):
    transport["response"] = lambda request: httpx.Response(
        200,
        headers=[(b"content-length", length.encode("utf-8"))],
        stream=httpx.ByteStream(PNG),
    )
    result = await invoke(slide_id="s1")
    assert result["status"] == "error"
    assert "invalid Content-Length" in result["message"]


async def test_total_deadline_includes_dns_wait(api, transport, monkeypatch):
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def resolve(url):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "_THUMBNAIL_TIMEOUT_SECONDS", 0.1)
    result = await asyncio.wait_for(invoke(slide_id="s1"), 5)
    assert entered.is_set() and exited.is_set()
    assert "timed out" in result["message"]
    assert transport["requests"] == []


async def test_caller_cancellation_closes_stream_and_restores_logs(
    api, loopback, caplog
):
    caplog.set_level(logging.INFO, logger="httpx")
    task = asyncio.create_task(invoke(slide_id="s1"))
    async with asyncio.timeout(5):
        while not loopback["seen"]:
            await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert "synthetic-private-query" not in caplog.text
    logging.getLogger("httpx").info("after cancellation")
    assert "after cancellation" in caplog.text


@pytest.mark.parametrize("first_fails", [False, True])
async def test_interleaved_downloads_keep_independent_logging_context(
    monkeypatch, caplog, first_fails
):
    started = asyncio.Event()
    release = asyncio.Event()
    caplog.set_level(logging.INFO, logger="httpx")

    async def resolve(url):
        return ["93.184.216.34"]

    async def respond(request):
        if request.url.path == "/held-private-path":
            started.set()
            await release.wait()
        elif first_fails:
            raise httpx.RequestError("request interrupted")
        return httpx.Response(200, content=PNG)

    @asynccontextmanager
    async def client(proxy, **kwargs):
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as value:
            yield value

    monkeypatch.setattr(security, "validate_public_http_url", resolve)
    monkeypatch.setattr(slides, "public_fetch_client", client)
    held = asyncio.create_task(
        slides._download_thumbnail_image(f"https://{HOST}/held-private-path")
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        if first_fails:
            with pytest.raises(slides._ThumbnailDownloadError):
                await slides._download_thumbnail_image(URL)
        else:
            assert await slides._download_thumbnail_image(URL) == PNG
        logging.getLogger("httpx").info("unrelated task stays visible")
    finally:
        release.set()
        assert await held == PNG
    assert "held-private-path" not in caplog.text
    assert "synthetic-private-query" not in caplog.text
    assert "unrelated task stays visible" in caplog.text


async def test_trusted_proxy_is_passed_to_shared_client(api, transport, monkeypatch):
    proxy = "http://127.0.0.1:9876"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "true")
    result = await invoke(slide_id="s1")
    assert result["status"] == "success"
    assert transport["proxy"] == proxy
    assert transport["requests"][0].url.host == HOST


async def test_killed_writer_never_publishes_a_partial_png(tmp_path):
    import sys

    script = """
import os
from xagent.web.tools.mcp import google_slides as slides
original = os.fdopen
class InterruptedWrite:
    def __init__(self, handle): self.handle = handle
    def __enter__(self): return self
    def __exit__(self, *args): return self.handle.__exit__(*args)
    def write(self, data):
        self.handle.write(data[:8])
        self.handle.flush()
        print("ready", flush=True)
        import signal
        signal.pause()
os.fdopen = lambda *args, **kwargs: InterruptedWrite(original(*args, **kwargs))
_, output, canonical, fd = slides._open_thumbnail_output_directory()
slides._write_thumbnail_png(b"partial image", output_dir=output,
    canonical_output_dir=canonical, dir_fd=fd, filename="slide.png")
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 30) == b"ready\n"
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.communicate(), 5)
    output = tmp_path / "output"
    assert not list(output.glob("*.png"))
    # A force-killed process cannot run finally; remove its private staging name.
    for temporary in output.glob("*.tmp"):
        temporary.unlink()
