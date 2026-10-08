import json
from unittest.mock import Mock

import pytest
from googleapiclient.errors import HttpError

from xagent.web.tools.mcp import google_docs


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "access-token")


class _HttpResponse:
    def __init__(self, status: int, reason: str = "error"):
        self.status = status
        self.reason = reason


def _http_error(status: int, body: dict) -> HttpError:
    return HttpError(
        _HttpResponse(status),
        json.dumps(body).encode("utf-8"),
        uri="https://docs.googleapis.com/v1/documents/doc123?alt=json",
    )


def _not_found() -> HttpError:
    return _http_error(
        404,
        {
            "error": {
                "code": 404,
                "message": "Requested entity was not found.",
                "status": "NOT_FOUND",
            }
        },
    )


def _mock_docs_service(monkeypatch):
    service = Mock()
    get_service = Mock(return_value=service)
    monkeypatch.setattr(google_docs, "get_docs_service", get_service)
    return service, get_service


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("doc123", "doc123"),
        ("https://docs.google.com/document/d/doc123/edit", "doc123"),
        ("https://docs.google.com/document/u/1/d/doc123/edit?tab=t.0", "doc123"),
    ],
)
def test_get_document_accepts_bare_id_and_link_forms(monkeypatch, value, expected):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.return_value = {
        "documentId": expected,
        "title": "Plan",
        "body": {"content": []},
    }

    result = json.loads(google_docs.google_docs_get_document(value))

    assert result["status"] == "success"
    assert service.documents.return_value.get.call_args.kwargs == {
        "documentId": expected
    }


def test_get_document_rejects_a_title_without_calling_the_api(monkeypatch):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(google_docs.google_docs_get_document("Q3 planning notes"))

    assert result["status"] == "error"
    message = result["message"]
    assert "'Q3 planning notes' is not a Google Docs link" in message
    assert "cannot search for or list documents by name" in message
    assert "https://docs.google.com/document/d/" in message
    assert "google_docs_create_document" in message
    get_service.assert_not_called()


def test_get_document_maps_not_found_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _not_found()

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["status"] == "error"
    message = result["message"]
    assert message.startswith("Google Docs could not open this document")
    assert "does not exist" in message
    assert "google_docs_create_document" in message
    assert "Google Drive" not in message
    assert message.endswith(
        "Google API response: HTTP 404 Requested entity was not found."
    )


def test_get_document_maps_permission_denied_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _http_error(
        403,
        {
            "error": {
                "code": 403,
                "message": "The caller does not have permission",
                "status": "PERMISSION_DENIED",
            }
        },
    )

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["message"].startswith("Google Docs could not open this document")
    assert "HTTP 403 The caller does not have permission" in result["message"]


@pytest.mark.parametrize(
    "reason_body",
    [
        {"errors": [{"reason": "rateLimitExceeded"}]},
        {"details": [{"reason": "ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]},
        {"details": [{"reason": "SERVICE_DISABLED"}]},
    ],
)
def test_get_document_keeps_raw_error_for_non_access_403(monkeypatch, reason_body):
    service, _ = _mock_docs_service(monkeypatch)
    error = _http_error(
        403, {"error": {"code": 403, "message": "Denied", **reason_body}}
    )
    service.documents.return_value.get.return_value.execute.side_effect = error

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result["message"] == str(error)


def test_get_document_keeps_raw_error_for_other_failures(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = RuntimeError(
        "boom"
    )

    result = json.loads(google_docs.google_docs_get_document("doc123"))

    assert result == {"status": "error", "message": "boom"}


def test_append_text_maps_not_found_to_an_actionable_message(monkeypatch):
    service, _ = _mock_docs_service(monkeypatch)
    service.documents.return_value.get.return_value.execute.side_effect = _not_found()

    result = json.loads(google_docs.google_docs_append_text("doc123", "more"))

    assert result["message"].startswith("Google Docs could not open this document")
    service.documents.return_value.batchUpdate.assert_not_called()


@pytest.mark.parametrize(
    "call",
    [
        lambda: google_docs.google_docs_append_text("Weekly report", "x"),
        lambda: google_docs.google_docs_replace_text("Weekly report", "a", "b"),
        lambda: google_docs.google_docs_batch_update("Weekly report", "[]"),
    ],
)
def test_editing_tools_reject_a_title_without_calling_the_api(monkeypatch, call):
    _, get_service = _mock_docs_service(monkeypatch)

    result = json.loads(call())

    assert result["status"] == "error"
    assert "is not a Google Docs link" in result["message"]
    get_service.assert_not_called()
