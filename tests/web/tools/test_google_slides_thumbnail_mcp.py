"""Thumbnail download behavior of the Google Slides connector.

The tests drive the registered MCP tool.  Argument, mapping and size checks use
a fake transport, while the transport-boundary tests (slow streams, redirects,
cookies, transport logging) run through a real ``requests`` session against a
loopback HTTP server whose adapter is mounted in place of the public one.  That
keeps the real redirect, cookie and streaming code paths under test instead of
replacing ``Session.send``.
"""

import http.server
import io
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import urlsplit

import pytest
import requests
from PIL import Image

from xagent.web.tools.mcp import google_slides

# A real PNG produced by Pillow.  Decodable image bytes keep the "the saved
# file is a valid image" assertion meaningful; the signature-only and truncated
# bodies in the invalid-image tests are rejected by the integrity check.
_PNG_BYTES = io.BytesIO()
Image.new("RGB", (2, 3), (10, 20, 30)).save(_PNG_BYTES, format="PNG")
_PNG_BYTES = _PNG_BYTES.getvalue()
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_THUMB_HOST = "lh3.googleusercontent.com"
_THUMB_URL = f"https://{_THUMB_HOST}/pages/thumb?token=thumb-secret-token"
_THUMB_SECRET = "thumb-secret-token"
_REVIEW_SECRET = "synthetic-review-signed-token"
_LARGE_LIMIT = 10 * 1024 * 1024


def _thumbnail_tool():
    tool = getattr(google_slides, "google_slides_get_slide_thumbnail", None)
    assert tool is not None, (
        "google_slides_get_slide_thumbnail is not defined on the connector module"
    )
    return tool


def _validate_thumbnail_url(url):
    validator = getattr(google_slides, "_validate_thumbnail_url", None)
    assert validator is not None, "_validate_thumbnail_url is not defined"
    return validator(url)


@pytest.fixture(autouse=True)
def _google_env(monkeypatch, tmp_path):
    # The connector only needs a token to build its (mocked) API client; the
    # image download must never use it.  Every test starts from a task
    # workspace output directory and with no ambient proxy configuration.
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR", str(tmp_path))
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


class _SlidesApi:
    """A mocked Slides API service with an inspectable thumbnail call."""

    def __init__(self):
        self.service = Mock()
        self.presentations = Mock()
        self.pages = Mock()
        self.service.presentations.return_value = self.presentations
        self.presentations.pages.return_value = self.pages
        self.get_thumbnail = self.pages.getThumbnail
        self.set_thumbnail()
        self.set_slides("s1", "s2", "s3", "s4", "s5")

    def set_thumbnail(self, payload=None, error=None):
        request = self.get_thumbnail.return_value
        if error is not None:
            request.execute.side_effect = error
        else:
            request.execute.return_value = (
                {
                    "contentUrl": _THUMB_URL,
                    "width": 1920,
                    "height": 1080,
                }
                if payload is None
                else payload
            )

    def set_slides(self, *object_ids, error=None):
        request = self.presentations.get.return_value
        if error is not None:
            request.execute.side_effect = error
        else:
            request.execute.return_value = {
                "slides": [{"objectId": object_id} for object_id in object_ids]
            }


def _install_slides_api(monkeypatch):
    api = _SlidesApi()
    monkeypatch.setattr(google_slides, "get_slides_service", lambda: api.service)
    return api


def _install_http(monkeypatch, responder):
    """Replace the controlled send seam behind the tool's requests session."""
    _thumbnail_tool()
    state = {"calls": [], "sessions": []}

    def _record(session, request, kwargs):
        state["sessions"].append(session)
        state["calls"].append((request, kwargs))
        return responder(len(state["calls"]) - 1, request, state)

    seam = getattr(google_slides, "_send_thumbnail_request", None)
    if seam is not None:

        def _fake_send(session, request, *, timeout):
            return _record(session, request, {"timeout": timeout})

        monkeypatch.setattr(google_slides, "_send_thumbnail_request", _fake_send)
    else:

        def _fake_session_send(session, request, **kwargs):
            return _record(session, request, kwargs)

        monkeypatch.setattr(google_slides.requests.Session, "send", _fake_session_send)
    return state


def _http_response(body=b"", status=200, headers=None, url=_THUMB_URL):
    response = google_slides.requests.Response()
    response.status_code = status
    response.url = url
    response.headers.update(headers or {})
    response._content = body
    response._content_consumed = True
    return response


def _no_body_response(status=200, headers=None):
    response = _http_response(b"", status=status, headers=headers)

    def _must_not_read(*_args, **_kwargs):
        raise AssertionError("the response body must not be read")

    response.iter_content = _must_not_read
    return response


def _png_of_size(size):
    """A decodable PNG padded to exactly ``size`` bytes.

    Trailing bytes after IEND are ignored by decoders, so this keeps the size
    boundary meaningful without synthesizing a huge image.
    """
    assert size >= len(_PNG_BYTES)
    return _PNG_BYTES + b"\x00" * (size - len(_PNG_BYTES))


def _streaming_response(body, chunks):
    response = _http_response(b"")
    consumed = {"bytes": 0}

    def _iter_content(chunk_size=None):
        chunk_size = chunk_size or 65536
        for index in range(chunks):
            chunk = body[index * chunk_size : (index + 1) * chunk_size]
            consumed["bytes"] += len(chunk)
            yield chunk

    response.iter_content = _iter_content
    return response, consumed


def _call_tool(**kwargs):
    return json.loads(_thumbnail_tool()(**kwargs))


def _result_path(result):
    return Path(result["path"])


def test_thumbnail_uses_the_slide_object_id_and_png_with_default_large(monkeypatch):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="slide_obj_1")

    assert result["status"] == "success"
    api.get_thumbnail.assert_called_once_with(
        presentationId="pres1",
        pageObjectId="slide_obj_1",
        thumbnailProperties_mimeType="PNG",
        thumbnailProperties_thumbnailSize="LARGE",
    )
    api.presentations.get.assert_not_called()
    assert result["mime_type"] == "image/png"
    assert result["size"] == len(_PNG_BYTES)
    assert result["slide_id"] == "slide_obj_1"
    assert result["page_number"] is None
    assert result["width"] == 1920
    assert result["height"] == 1080
    assert _result_path(result).read_bytes() == _PNG_BYTES
    assert len(http["calls"]) == 1


@pytest.mark.parametrize("size", ["LARGE", "MEDIUM", "SMALL"])
def test_thumbnail_passes_the_requested_image_size(monkeypatch, size):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="slide_obj_1", size=size)

    assert result["status"] == "success"
    assert (
        api.get_thumbnail.call_args.kwargs["thumbnailProperties_thumbnailSize"] == size
    )


def test_thumbnail_normalizes_size_case_and_whitespace(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="obj1", size=" medium ")

    assert result["status"] == "success"
    assert (
        api.get_thumbnail.call_args.kwargs["thumbnailProperties_thumbnailSize"]
        == "MEDIUM"
    )


async def _call_tool_via_mcp(arguments):
    tool = _thumbnail_tool()
    content, _result = await google_slides.mcp.call_tool(tool.__name__, arguments)
    return json.loads(content[0].text)


@pytest.mark.parametrize(
    ("page_number", "expected_object_id"),
    [(1, "s1"), (3, "s3"), (5, "s5")],
)
def test_thumbnail_resolves_a_one_based_page_number(
    monkeypatch, page_number, expected_object_id
):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", page_number=page_number)

    assert result["status"] == "success"
    assert result["page_number"] == page_number
    assert result["slide_id"] == expected_object_id
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == expected_object_id
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )


async def test_thumbnail_uses_default_large_and_png_through_the_mcp_layer(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = await _call_tool_via_mcp(
        {"presentation_id": "pres1", "slide_id": "slide_obj_1"}
    )

    assert result["status"] == "success"
    api.get_thumbnail.assert_called_once_with(
        presentationId="pres1",
        pageObjectId="slide_obj_1",
        thumbnailProperties_mimeType="PNG",
        thumbnailProperties_thumbnailSize="LARGE",
    )


async def test_thumbnail_resolves_page_number_through_the_mcp_layer(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = await _call_tool_via_mcp({"presentation_id": "pres1", "page_number": 3})

    assert result["status"] == "success"
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == "s3"
    api.presentations.get.assert_called_once_with(
        presentationId="pres1", fields="slides.objectId"
    )


def test_thumbnail_treats_numeric_looking_slide_id_as_an_object_id(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="3")

    assert result["status"] == "success"
    api.presentations.get.assert_not_called()
    assert api.get_thumbnail.call_args.kwargs["pageObjectId"] == "3"


def test_thumbnail_resolves_a_full_presentation_url(monkeypatch):
    api = _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(
        presentation_id=(
            "https://docs.google.com/presentation/d/presentation-Abc_123/edit#slide=id.p"
        ),
        slide_id="obj1",
    )

    assert result["status"] == "success"
    assert result["presentation_id"] == "presentation-Abc_123"
    assert (
        api.get_thumbnail.call_args.kwargs["presentationId"] == "presentation-Abc_123"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"slide_id": "obj1", "page_number": 2},
    ],
    ids=["neither", "both"],
)
def test_thumbnail_requires_exactly_one_slide_locator(monkeypatch, kwargs):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", **kwargs)

    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("page_number", [0, -1, 6, 1000])
def test_thumbnail_rejects_out_of_range_page_numbers(monkeypatch, page_number):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=page_number)

    assert result["status"] == "error"
    assert "page_number" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


def test_thumbnail_rejects_a_page_number_for_an_empty_deck(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_slides()
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=1)

    assert result["status"] == "error"
    assert "no slides" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("slide_id", ["", "   "])
def test_thumbnail_rejects_blank_slide_ids(monkeypatch, slide_id):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id=slide_id)

    assert result["status"] == "error"
    assert "slide_id" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("slide_id", [3, True, 1.5, ["obj1"], {"objectId": "obj1"}])
def test_thumbnail_rejects_non_string_slide_ids_in_a_direct_call(monkeypatch, slide_id):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id=slide_id)

    assert result["status"] == "error"
    assert "slide_id" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("page_number", [True, 1.5, "2", [2], {"n": 2}])
def test_thumbnail_rejects_non_integer_page_numbers_in_a_direct_call(
    monkeypatch, page_number
):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=page_number)

    assert result["status"] == "error"
    assert "page_number" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("size", ["HUGE", "", "JPEG", None, 3])
def test_thumbnail_rejects_unknown_sizes_in_a_direct_call(monkeypatch, size):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="obj1", size=size)

    assert result["status"] == "error"
    assert "size" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize("presentation_id", ["", "   ", None, 3])
def test_thumbnail_rejects_invalid_presentation_ids(monkeypatch, presentation_id):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id=presentation_id, slide_id="obj1")

    assert result["status"] == "error"
    assert "presentation_id" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"presentation_id": "pres1", "page_number": True},
        {"presentation_id": "pres1", "page_number": "2"},
        {"presentation_id": "pres1", "page_number": 2.0},
        {"presentation_id": "pres1", "page_number": 2.5},
        {"presentation_id": "pres1", "page_number": [2]},
        {"presentation_id": "pres1", "page_number": {"n": 2}},
        {"presentation_id": "pres1", "slide_id": 3},
        {"presentation_id": "pres1", "slide_id": ["s1"]},
        {"presentation_id": "pres1", "slide_id": "s1", "size": "HUGE"},
        {"presentation_id": "pres1", "slide_id": "s1", "size": 3},
    ],
)
async def test_thumbnail_schema_validation_never_coerces_a_bad_locator(
    monkeypatch, arguments
):
    from mcp.server.fastmcp.exceptions import ToolError

    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    with pytest.raises(ToolError, match="validation error"):
        await google_slides.mcp.call_tool(
            "google_slides_get_slide_thumbnail", arguments
        )

    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


async def test_thumbnail_rejects_conflicting_locators_through_the_mcp_layer(
    monkeypatch,
):
    api = _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = await _call_tool_via_mcp(
        {"presentation_id": "pres1", "slide_id": "obj1", "page_number": 2}
    )

    assert result["status"] == "error"
    assert "exactly one" in result["message"]
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


def test_thumbnail_writes_a_unique_file_per_download(monkeypatch):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    first = _call_tool(presentation_id="pres1", slide_id="obj1")
    second = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert first["path"] != second["path"]
    assert _result_path(first).read_bytes() == _PNG_BYTES
    assert _result_path(second).read_bytes() == _PNG_BYTES
    assert _result_path(first).parent == _result_path(second).parent


def test_thumbnail_keeps_the_written_file_inside_the_output_directory(
    monkeypatch, tmp_path
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="../../escape")

    assert result["status"] == "success"
    path = _result_path(result)
    assert path.is_file()
    assert path.parent == tmp_path / "output"
    assert ".." not in path.parts
    assert path.read_bytes() == _PNG_BYTES


def test_thumbnail_requires_a_task_workspace(monkeypatch):
    _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )
    monkeypatch.delenv("XAGENT_GOOGLE_DRIVE_OUTPUT_DIR")

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "task workspace" in result["message"]
    assert http["calls"] == []


@pytest.mark.parametrize(
    "url",
    [
        "https://googleusercontent.com/pages/thumb",
        "https://lh3.googleusercontent.com/pages/thumb?token=t",
        "https://lh7-rt.googleusercontent.com/pages/thumb?token=t",
        "https://lh3.googleusercontent.com:443/pages/thumb?token=t",
    ],
)
def test_validate_thumbnail_url_accepts_google_content_hosts(url):
    assert _validate_thumbnail_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://lh3.googleusercontent.com/pages/thumb",
        "https://user:pass@lh3.googleusercontent.com/pages/thumb",
        "https://user@lh3.googleusercontent.com/pages/thumb",
        "https://lh3.googleusercontent.com:8443/pages/thumb",
        "https://localhost/pages/thumb",
        "https://127.0.0.1/pages/thumb",
        "https://[2001:4860:4860::8888]/pages/thumb",
        "https://googleusercontent.com.evil.example/pages/thumb",
        "https://evilgoogleusercontent.com/pages/thumb",
        "https://evil.example/pages/thumb",
        "https://google.com/pages/thumb",
        "ftp://lh3.googleusercontent.com/pages/thumb",
        "lh3.googleusercontent.com/pages/thumb",
        "",
    ],
)
def test_validate_thumbnail_url_rejects_unsafe_urls(url):
    with pytest.raises(ValueError):
        _validate_thumbnail_url(url)


@pytest.mark.parametrize(
    "content_url",
    [
        "http://lh3.googleusercontent.com/pages/thumb?token=t",
        "https://user:pass@lh3.googleusercontent.com/pages/thumb",
        "https://lh3.googleusercontent.com:8080/pages/thumb",
        "https://googleusercontent.com.evil.example/pages/thumb",
        "https://evilgoogleusercontent.com/pages/thumb",
        "https://evil.example/pages/thumb",
    ],
)
def test_thumbnail_rejects_a_disallowed_content_url_without_requesting_it(
    monkeypatch, content_url
):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail({"contentUrl": content_url, "width": 1, "height": 1})
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_follows_a_redirect_to_another_allowed_google_host(monkeypatch):
    _install_slides_api(monkeypatch)
    redirected_url = "https://lh7-rt.googleusercontent.com/pages/thumb?token=two"

    def responder(index, request, state):
        if index == 0:
            return _http_response(
                status=302,
                headers={"Location": redirected_url},
                url=request.url,
            )
        return _http_response(_PNG_BYTES, url=request.url)

    http = _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    assert len(http["calls"]) == 2
    assert http["calls"][1][0].url == redirected_url


@pytest.mark.parametrize(
    "location",
    [
        "https://evil.example/pages/thumb",
        "http://lh3.googleusercontent.com/pages/thumb",
        "https://googleusercontent.com.evil.example/pages/thumb",
        "https://user:pass@lh3.googleusercontent.com/pages/thumb",
    ],
)
def test_thumbnail_rejects_a_redirect_target_that_fails_validation(
    monkeypatch, location
):
    _install_slides_api(monkeypatch)

    def responder(index, request, state):
        if index == 0:
            return _http_response(
                status=302, headers={"Location": location}, url=request.url
            )
        raise AssertionError(
            "a redirect target that fails validation must not be requested"
        )

    http = _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert len(http["calls"]) == 1


def test_thumbnail_rejects_more_than_three_redirects(monkeypatch):
    _install_slides_api(monkeypatch)
    urls = [
        "https://lh3.googleusercontent.com/pages/0?token=t",
        "https://lh3.googleusercontent.com/pages/1?token=t",
        "https://lh3.googleusercontent.com/pages/2?token=t",
        "https://lh3.googleusercontent.com/pages/3?token=t",
        "https://lh3.googleusercontent.com/pages/4?token=t",
        "https://lh3.googleusercontent.com/pages/5?token=t",
    ]

    def responder(index, request, state):
        return _http_response(
            status=302, headers={"Location": urls[index + 1]}, url=request.url
        )

    http = _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "redirect" in result["message"]
    assert len(http["calls"]) == 4


def test_thumbnail_rejects_declared_content_length_over_the_limit(
    monkeypatch, tmp_path
):
    _install_slides_api(monkeypatch)
    response = _no_body_response(headers={"Content-Length": str(_LARGE_LIMIT + 1)})
    _install_http(monkeypatch, lambda index, request, state: response)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "too large" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_rejects_an_unlabelled_oversized_stream(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    body = _PNG_SIGNATURE + b"x" * (2 * _LARGE_LIMIT)
    response, consumed = _streaming_response(body, chunks=2 * _LARGE_LIMIT // 65536 + 2)
    _install_http(monkeypatch, lambda index, request, state: response)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "too large" in result["message"]
    assert consumed["bytes"] < len(body)
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_rejects_a_stream_that_lies_about_its_length(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    body = _PNG_SIGNATURE + b"x" * (2 * _LARGE_LIMIT)
    response, consumed = _streaming_response(body, chunks=2 * _LARGE_LIMIT // 65536 + 2)
    response.headers["Content-Length"] = "100"
    _install_http(monkeypatch, lambda index, request, state: response)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "too large" in result["message"]
    assert consumed["bytes"] < len(body)
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_rejects_a_malformed_content_length(monkeypatch):
    _install_slides_api(monkeypatch)
    response = _no_body_response(headers={"Content-Length": "12ab"})
    _install_http(monkeypatch, lambda index, request, state: response)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "Content-Length" in result["message"]


def test_thumbnail_accepts_a_body_exactly_at_the_limit(monkeypatch):
    _install_slides_api(monkeypatch)
    body = _png_of_size(_LARGE_LIMIT)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    assert result["size"] == _LARGE_LIMIT
    assert _result_path(result).stat().st_size == _LARGE_LIMIT


def test_thumbnail_rejects_a_body_one_byte_over_the_limit(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    body = _png_of_size(_LARGE_LIMIT + 1)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "too large" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_stops_reading_at_the_timeout(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    response, consumed = _streaming_response(_PNG_SIGNATURE + b"x" * 200_000, chunks=4)
    _install_http(monkeypatch, lambda index, request, state: response)
    clock = {"reads": 0}
    readings = [0.0, 0.0, 10.0, 35.0]

    def fake_monotonic():
        clock["reads"] += 1
        index = min(clock["reads"] - 1, len(readings) - 1)
        return readings[index]

    monkeypatch.setattr(google_slides.time, "monotonic", fake_monotonic)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "timed out" in result["message"]
    assert consumed["bytes"] < 200_008
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_maps_an_interrupted_stream_to_an_error(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    response = _http_response(b"")

    def _iter_content(chunk_size=None):
        yield _PNG_SIGNATURE + b"partial"
        raise google_slides.requests.exceptions.ChunkedEncodingError(
            f"connection lost while reading {_THUMB_URL}"
        )

    response.iter_content = _iter_content
    _install_http(monkeypatch, lambda index, request, state: response)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "thumb-secret-token" not in json.dumps(result)
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_maps_a_read_timeout_to_an_error(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)

    def responder(index, request, state):
        raise google_slides.requests.exceptions.ReadTimeout(
            f"read timed out for {_THUMB_URL}"
        )

    _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "thumb-secret-token" not in json.dumps(result)
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_rejects_an_empty_response_body(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(b""))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "empty" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


@pytest.mark.parametrize("body", [b"not a png at all", b"<html>error</html>"])
def test_thumbnail_rejects_a_non_png_response(monkeypatch, tmp_path, body):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "PNG" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_download_does_not_carry_credentials(monkeypatch):
    _install_slides_api(monkeypatch)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    first = _call_tool(presentation_id="pres1", slide_id="obj1")
    second = _call_tool(presentation_id="pres1", page_number=2)
    assert first["status"] == "success"
    assert second["status"] == "success"

    for request, _kwargs in http["calls"]:
        header_names = {name.lower() for name in request.headers}
        assert "authorization" not in header_names
        assert "cookie" not in header_names
        assert "access-token" not in json.dumps(dict(request.headers))
    sessions = http["sessions"]
    assert len({id(session) for session in sessions}) == len(sessions)
    for session in sessions:
        assert session.auth is None
        assert len(session.cookies) == 0


def test_thumbnail_never_returns_or_logs_the_short_lived_content_url(
    monkeypatch, caplog
):
    caplog.set_level(logging.DEBUG)
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    assert "thumb-secret-token" not in json.dumps(result)
    assert "contentUrl" not in json.dumps(result)
    assert "thumb-secret-token" not in caplog.text


def test_thumbnail_http_failure_does_not_leak_the_content_url(
    monkeypatch, caplog, tmp_path
):
    caplog.set_level(logging.DEBUG)
    _install_slides_api(monkeypatch)
    _install_http(
        monkeypatch,
        lambda index, request, state: _http_response(b"", status=403),
    )

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "403" in result["message"]
    assert "thumb-secret-token" not in json.dumps(result)
    assert _THUMB_HOST not in json.dumps(result)
    assert "thumb-secret-token" not in caplog.text
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_network_error_does_not_leak_the_signed_url(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    _install_slides_api(monkeypatch)

    def responder(index, request, state):
        raise google_slides.requests.exceptions.ConnectionError(
            f"failed to connect to {_THUMB_URL}"
        )

    _install_http(monkeypatch, responder)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "thumb-secret-token" not in json.dumps(result)
    assert "thumb-secret-token" not in caplog.text
    assert result["status"] != "success"


@pytest.mark.parametrize("failure", [RuntimeError("api boom"), Exception("api boom")])
def test_thumbnail_api_failures_are_not_reported_as_success(monkeypatch, failure):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail(error=failure)
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_missing_content_url_is_an_error(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail({"width": 1920, "height": 1080})
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert http["calls"] == []


def test_thumbnail_page_lookup_failure_is_not_a_success(monkeypatch):
    api = _install_slides_api(monkeypatch)
    api.set_slides(error=RuntimeError("presentation get failed"))
    http = _install_http(
        monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES)
    )

    result = _call_tool(presentation_id="pres1", page_number=2)

    assert result["status"] == "error"
    api.get_thumbnail.assert_not_called()
    assert http["calls"] == []


async def test_thumbnail_download_failure_returns_an_error_through_the_mcp_layer(
    monkeypatch,
):
    _install_slides_api(monkeypatch)
    body = _png_of_size(_LARGE_LIMIT + 1)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = await _call_tool_via_mcp({"presentation_id": "pres1", "slide_id": "obj1"})

    assert result["status"] == "error"
    assert result["message"]


async def test_thumbnail_tool_is_registered_with_its_limits_and_failure_contract():
    tools = {tool.name: tool for tool in await google_slides.mcp.list_tools()}
    tool = tools.get("google_slides_get_slide_thumbnail")
    assert tool is not None
    description = tool.description
    for phrase in (
        "exactly one of slide_id or page_number",
        "page_number",
        "1-based",
        "PNG",
        "LARGE",
        "task workspace",
        "file_ref",
        "error",
    ):
        assert phrase in description
    schema = tool.inputSchema
    assert set(schema["properties"]) == {
        "presentation_id",
        "slide_id",
        "page_number",
        "size",
    }
    assert schema["properties"]["size"]["default"] == "LARGE"
    assert schema["properties"]["size"]["enum"] == ["SMALL", "MEDIUM", "LARGE"]
    assert schema["required"] == ["presentation_id"]
    # Existing connector tools stay registered.
    assert "google_slides_get_presentation" in tools
    assert "google_slides_import_pptx" in tools


class _LoopbackAdapter(requests.adapters.HTTPAdapter):
    """A real HTTPAdapter that connects the allowed HTTPS URL to a local server.

    The tool still sees and validates the ordinary public hostname; only the
    socket destination is redirected, so requests' real redirect, cookie and
    streaming code paths keep running.
    """

    def __init__(self, port, recorded_responses):
        super().__init__()
        self._port = port
        self._recorded_responses = recorded_responses

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        return self.poolmanager.connection_from_host(
            "127.0.0.1", self._port, scheme="http"
        )

    def cert_verify(self, *args, **kwargs):
        return None

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        self._recorded_responses.append(response)
        return response


@pytest.fixture
def real_http(monkeypatch):
    """A loopback image host exercised through a real requests session."""
    seen = []
    responses = []
    sessions = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            return None

        def _write(self, body):
            try:
                self.wfile.write(body)
            except OSError:
                return None
            return None

        def do_GET(self):
            seen.append({"path": self.path, "headers": dict(self.headers)})
            route = urlsplit(self.path).path
            if route == "/cookie":
                body = b"redirect"
                self.send_response(302)
                self.send_header(
                    "Location", f"https://{_THUMB_HOST}/image?token={_REVIEW_SECRET}"
                )
                self.send_header(
                    "Set-Cookie", "thumbnail_session=synthetic-cookie; Path=/"
                )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self._write(body)
                return
            if route == "/large-redirect":
                body = b"x" * (10 * 1024 * 1024 + 1)
                self.send_response(302)
                self.send_header("Location", "https://evil.example/not-requested")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self._write(body)
                return
            if route == "/slow":
                body = _PNG_BYTES[:20]
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                for byte in body:
                    self._write(bytes([byte]))
                    try:
                        self.wfile.flush()
                    except OSError:
                        return
                    time.sleep(0.1)
                return
            if route == "/stall":
                time.sleep(3.0)
            body = _PNG_BYTES
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_new_session = google_slides._new_thumbnail_session

    def _new_thumbnail_session():
        session = original_new_session()
        session.mount("https://", _LoopbackAdapter(server.server_port, responses))
        sessions.append(session)
        return session

    monkeypatch.setattr(google_slides, "_new_thumbnail_session", _new_thumbnail_session)
    try:
        yield SimpleNamespace(
            server=server,
            seen=seen,
            responses=responses,
            sessions=sessions,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _install_real_api(monkeypatch, path, secret=_REVIEW_SECRET):
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail(
        {
            "contentUrl": f"https://{_THUMB_HOST}{path}?token={secret}",
            "width": 2,
            "height": 3,
        }
    )
    return api


def test_thumbnail_real_slow_stream_stops_at_the_total_deadline(
    monkeypatch, real_http, tmp_path
):
    """A byte-dripping body must be cut off at the configured total deadline."""
    monkeypatch.setattr(google_slides, "_THUMBNAIL_TIMEOUT_SECONDS", 0.8)
    _install_real_api(monkeypatch, "/slow")

    started = time.monotonic()
    result = _call_tool(presentation_id="pres1", slide_id="obj1")
    elapsed = time.monotonic() - started

    assert result["status"] == "error"
    assert "timed out" in result["message"]
    assert elapsed < 1.7
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_real_stalled_response_headers_respect_the_total_deadline(
    monkeypatch, real_http, tmp_path
):
    """Connection/headers must not outlive the total deadline either."""
    monkeypatch.setattr(google_slides, "_THUMBNAIL_TIMEOUT_SECONDS", 0.8)
    _install_real_api(monkeypatch, "/stall")

    started = time.monotonic()
    result = _call_tool(presentation_id="pres1", slide_id="obj1")
    elapsed = time.monotonic() - started

    assert result["status"] == "error"
    assert "timed out" in result["message"]
    assert elapsed < 2.5
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_real_redirect_body_is_not_read_before_validation(
    monkeypatch, real_http
):
    """A disallowed redirect body must not be buffered before the host check."""
    _install_real_api(monkeypatch, "/large-redirect")

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "host" in result["message"]
    assert len(real_http.seen) == 1
    redirects = [
        response for response in real_http.responses if response.status_code == 302
    ]
    assert redirects
    for response in redirects:
        assert not response._content


def test_thumbnail_real_redirect_does_not_replay_cookies(monkeypatch, real_http):
    """Set-Cookie on an allowed redirect must not reach the next hop."""
    _install_real_api(monkeypatch, "/cookie")

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    assert len(real_http.seen) == 2
    assert "Cookie" not in real_http.seen[1]["headers"]
    assert all("Authorization" not in request["headers"] for request in real_http.seen)
    assert all(len(session.cookies) == 0 for session in real_http.sessions)


def test_thumbnail_real_transport_debug_log_does_not_leak_the_signed_query(
    monkeypatch, real_http, caplog
):
    caplog.set_level(logging.DEBUG, logger="urllib3.connectionpool")
    _install_real_api(monkeypatch, "/image")

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    assert _REVIEW_SECRET not in caplog.text


def _signed_api_error(signed_url):
    from googleapiclient.errors import HttpError
    from httplib2 import Response

    def value_error():
        return ValueError(f"provider failure {signed_url}")

    def runtime_error():
        return RuntimeError(f"provider failure {signed_url}")

    def http_error():
        return HttpError(
            Response({"status": "500"}),
            json.dumps(
                {"error": {"message": f"provider failure {signed_url}"}}
            ).encode(),
        )

    return {"value": value_error, "runtime": runtime_error, "http": http_error}


@pytest.mark.parametrize("kind", ["value", "runtime", "http"])
def test_thumbnail_api_errors_do_not_leak_the_signed_url(monkeypatch, caplog, kind):
    caplog.set_level(logging.DEBUG)
    signed_url = f"https://{_THUMB_HOST}/image?token={_REVIEW_SECRET}"
    api = _install_slides_api(monkeypatch)
    api.set_thumbnail(error=_signed_api_error(signed_url)[kind]())
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert _REVIEW_SECRET not in json.dumps(result)
    assert _REVIEW_SECRET not in caplog.text


def test_thumbnail_does_not_follow_a_symlink_in_the_output_directory(
    monkeypatch, tmp_path
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside.png"
    (output_dir / "slide_obj1.png").symlink_to(outside)

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    path = _result_path(result)
    assert path.parent == output_dir
    assert path.name != "slide_obj1.png"
    assert not outside.exists()
    assert path.read_bytes() == _PNG_BYTES


def test_thumbnail_concurrent_calls_write_two_distinct_files(monkeypatch, tmp_path):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))
    output_dir = tmp_path / "output"
    barrier = threading.Barrier(2)
    original_exists = Path.exists

    def synchronized_exists(self):
        if self.parent == output_dir and self.suffix == ".png":
            barrier.wait(timeout=5)
            return False
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", synchronized_exists)

    def _call(_index):
        return _call_tool(presentation_id="pres1", slide_id="obj1")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(_call, (0, 1)))

    assert all(result["status"] == "success" for result in results)
    paths = {result["path"] for result in results}
    assert len(paths) == 2
    assert len(list(output_dir.glob("*.png"))) == 2


@pytest.mark.parametrize(
    "body",
    [_PNG_SIGNATURE, _PNG_SIGNATURE + b"junk", _PNG_BYTES[:33]],
    ids=["signature_only", "signature_and_junk", "truncated"],
)
def test_thumbnail_rejects_png_like_bodies_that_cannot_be_decoded(
    monkeypatch, tmp_path, body
):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(body))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "error"
    assert "PNG" in result["message"]
    assert not list((tmp_path / "output").glob("*.png"))


def test_thumbnail_saves_a_decodable_png(monkeypatch):
    _install_slides_api(monkeypatch)
    _install_http(monkeypatch, lambda index, request, state: _http_response(_PNG_BYTES))

    result = _call_tool(presentation_id="pres1", slide_id="obj1")

    assert result["status"] == "success"
    with Image.open(_result_path(result)) as image:
        image.load()
        assert image.format == "PNG"
