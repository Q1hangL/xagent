"""Google Drive credentials and Picker config for a resource owner's connection.

The ``/api/cloud`` routes read the user's own connection. Trusted in-process
callers can read the credential stored for a resource owner key instead; the
routes' status codes and details must stay exactly as they are.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, get_args
from unittest.mock import patch

import pytest
import requests
from fastapi import HTTPException, Response
from google.auth import _exponential_backoff
from google.auth.exceptions import RefreshError, TransportError
from google.auth.transport import requests as google_auth_requests
from google.oauth2.credentials import Credentials
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from xagent.web.api import auth as auth_api
from xagent.web.api.cloud_storage import (
    GOOGLE_TOKEN_URI,
    GoogleDriveCredentialError,
    GoogleDriveCredentialReason,
    _GoogleTokenRequest,
    get_google_credentials,
    get_google_drive_picker_config,
    issue_google_drive_picker_config,
)
from xagent.web.models.database import Base
from xagent.web.models.oauth_provider import OAuthProvider
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

OWNER = "delegated:7:member-3"
OTHER_OWNER = "delegated:7:member-4"
DRIVE = "https://www.googleapis.com/auth/drive"
DRIVE_FILE = "https://www.googleapis.com/auth/drive.file"
DRIVE_READONLY = "https://www.googleapis.com/auth/drive.readonly"
USERINFO = (
    "https://www.googleapis.com/auth/userinfo.email "
    "https://www.googleapis.com/auth/userinfo.profile"
)
GMAIL = "https://www.googleapis.com/auth/gmail.modify"
DB_CLIENT_ID = "123456789012-db.apps.googleusercontent.com"
ENV_CLIENT_ID = "999999999999-env.apps.googleusercontent.com"

RECONNECT_DETAIL = "Google Drive session expired. Please reconnect."
SCOPE_DETAIL = (
    "This Google Drive connection uses an outdated permission. "
    "Reconnect it before opening Google Drive Picker."
)
PICKER_NOT_CONFIGURED_DETAIL = (
    "Google Drive Picker is not configured. Set the dedicated, "
    "referrer-restricted GOOGLE_PICKER_API_KEY and either "
    "GOOGLE_PICKER_APP_ID or a numeric Google OAuth client_id. "
    "The access token and Picker key are sent to the browser."
)


def _future(minutes: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


class _Store:
    def __init__(self, tmp_path) -> None:
        self.engine = create_engine(f"sqlite:///{tmp_path / 'picker.db'}")
        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False
        )
        self.db = self.sessions()
        user = User(username="picker-user", password_hash="hash")
        self.db.add(user)
        self.db.commit()
        self.user_id = int(user.id)

    def add_provider(self, *, client_id: str, client_secret: str) -> None:
        self.db.add(
            OAuthProvider(
                provider_name="google",
                name="Google",
                client_id=client_id,
                client_secret=client_secret,
                auth_url="https://accounts.google.com/o/oauth2/auth",
                token_url="https://oauth2.googleapis.com/token",
            )
        )
        self.db.commit()

    def add_drive(
        self,
        *,
        owner: str | None,
        token: str,
        provider_user_id: str = "google-user",
        scope: str | None = f"{USERINFO} {DRIVE_FILE}",
        refresh_token: str | None = "refresh-token",
        expires_at: datetime | None = None,
    ) -> int:
        row = UserOAuth(
            user_id=self.user_id,
            provider="google-drive",
            resource_owner_key=owner,
            provider_user_id=provider_user_id,
            access_token=token,
            refresh_token=refresh_token,
            scope=scope,
            expires_at=expires_at if expires_at is not None else _future(60),
        )
        self.db.add(row)
        self.db.commit()
        return int(row.id)

    def stored(self, row_id: int) -> tuple[Any, Any, Any]:
        fresh = self.sessions()
        try:
            row = fresh.get(UserOAuth, row_id)
            assert row is not None
            return row.access_token, row.refresh_token, row.expires_at
        finally:
            fresh.close()

    def snapshot(self, row_id: int) -> tuple[Any, ...]:
        fresh = self.sessions()
        try:
            row = fresh.get(UserOAuth, row_id)
            assert row is not None
            return (
                row.access_token,
                row.refresh_token,
                row.expires_at,
                row.scope,
                row.provider_user_id,
                row.resource_owner_key,
            )
        finally:
            fresh.close()

    def close(self) -> None:
        self.db.close()
        self.engine.dispose()


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in (
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
        "GOOGLE_API_KEY",
        "GOOGLE_PICKER_API_KEY",
        "GOOGLE_PICKER_APP_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    created = _Store(tmp_path)
    created.add_provider(client_id=DB_CLIENT_ID, client_secret="db-secret")
    try:
        yield created
    finally:
        created.close()


@pytest.fixture
def picker_key(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")


def _refreshing(
    *,
    token: str = "refreshed-token",
    refresh_token: str | None = None,
    error: Exception | None = None,
):
    calls: list[str] = []

    def _refresh(self: Credentials, request: Any) -> None:
        del request
        calls.append(str(self.token))
        if error is not None:
            raise error
        self.token = token
        self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            hours=1
        )
        if refresh_token is not None:
            self._refresh_token = refresh_token

    return patch.object(Credentials, "refresh", _refresh), calls


# --- contract ------------------------------------------------------------


def test_credential_reasons_are_stable() -> None:
    assert set(get_args(GoogleDriveCredentialReason)) == {
        "picker_not_configured",
        "account_not_connected",
        "account_not_found",
        "reauth_required",
        "refresh_unavailable",
        "oauth_unconfigured",
        "scope_full_drive",
        "scope_mismatch",
        "scope_drive_missing",
    }


def test_credential_error_is_the_routes_http_error_plus_a_reason() -> None:
    error = GoogleDriveCredentialError(
        401, RECONNECT_DETAIL, reason="reauth_required", oauth_account_id=5
    )
    assert isinstance(error, HTTPException)
    assert (error.status_code, error.detail) == (401, RECONNECT_DETAIL)
    assert (error.reason, error.oauth_account_id) == ("reauth_required", 5)
    rowless = GoogleDriveCredentialError(
        503, PICKER_NOT_CONFIGURED_DETAIL, reason="picker_not_configured"
    )
    assert rowless.oauth_account_id is None


def test_contract_functions_take_their_options_by_keyword(store, picker_key) -> None:
    row_id = store.add_drive(owner=OWNER, token="owned")

    creds = get_google_credentials(
        user_id=store.user_id,
        db=store.db,
        account_id=row_id,
        resource_owner_key=OWNER,
        min_ttl=timedelta(minutes=5),
    )
    issued = issue_google_drive_picker_config(
        db=store.db,
        user_id=store.user_id,
        resource_owner_key=OWNER,
        account_id=row_id,
        minimal_scopes=True,
        min_ttl=timedelta(minutes=5),
    )

    assert creds.token == issued["access_token"] == "owned"
    with pytest.raises(TypeError):
        get_google_credentials(store.user_id, store.db, row_id, OWNER)
    with pytest.raises(TypeError):
        issue_google_drive_picker_config(store.db, store.user_id)


def _credentials_token(store: "_Store") -> str:
    return get_google_credentials(
        store.user_id, store.db, resource_owner_key=OWNER
    ).token


def _issued_token(store: "_Store") -> str:
    return issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )["access_token"]


@pytest.mark.parametrize(
    "read_token", [_credentials_token, _issued_token], ids=["credentials", "issue"]
)
@pytest.mark.parametrize(
    ("minutes_left", "expected"), [(4, "refreshed-token"), (6, "stored")]
)
def test_default_refresh_threshold_is_five_minutes(
    store, picker_key, read_token, minutes_left, expected
) -> None:
    store.add_drive(owner=OWNER, token="stored", expires_at=_future(minutes_left))
    patcher, _calls = _refreshing()

    with patcher:
        assert read_token(store) == expected


# --- owner namespace -----------------------------------------------------


def test_credentials_read_only_the_requested_owner_namespace(store) -> None:
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")
    store.add_drive(owner=OTHER_OWNER, token="other", provider_user_id="other")

    assert get_google_credentials(store.user_id, store.db).token == "ordinary"
    owned = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)
    assert owned.token == "owned"


def test_owner_namespace_cannot_select_an_ordinary_row_by_id(store) -> None:
    ordinary_id = store.add_drive(owner=None, token="ordinary")
    store.add_drive(owner=OWNER, token="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(
            store.user_id, store.db, ordinary_id, resource_owner_key=OWNER
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Selected Google Drive account not found"
    assert exc_info.value.reason == "account_not_found"


def test_missing_owner_connection_is_not_connected(store) -> None:
    store.add_drive(owner=None, token="ordinary")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Google Drive account not connected"
    assert exc_info.value.reason == "account_not_connected"
    assert exc_info.value.oauth_account_id is None


def test_ordinary_lookup_ignores_owner_rows(store) -> None:
    store.add_drive(owner=OWNER, token="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db)

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "account_not_connected"


def test_owner_branch_prefers_the_newest_row(store) -> None:
    store.add_drive(owner=OWNER, token="older", provider_user_id="first")
    store.add_drive(owner=OWNER, token="newer", provider_user_id="second")

    creds = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert creds.token == "newer"


def test_owner_branch_rereads_a_row_already_loaded_in_the_session(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="before")
    loaded = store.db.get(UserOAuth, row_id)
    assert loaded is not None and loaded.access_token == "before"
    # Another writer replaced the token after this session loaded the row.
    store.db.execute(
        text("UPDATE user_oauth SET access_token = 'after' WHERE id = :id"),
        {"id": row_id},
    )

    creds = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert creds.token == "after"


def test_owner_token_cleared_row_requires_reauth(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "reauth_required"
    assert exc_info.value.oauth_account_id == row_id


@pytest.mark.parametrize("owner", [None, OWNER])
def test_due_refresh_without_refresh_token_requires_reauth(store, owner) -> None:
    expires_at = _future(1)
    row_id = store.add_drive(
        owner=owner, token="old", refresh_token=None, expires_at=expires_at
    )
    patcher, calls = _refreshing()

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "reauth_required"
    assert exc_info.value.oauth_account_id == row_id
    assert calls == []
    access_token, refresh_token, _expires = store.stored(row_id)
    assert (access_token, refresh_token) == ("old", None)


# --- OAuth client resolution -------------------------------------------


def test_owner_branch_resolves_the_oauth_client_per_field(store, monkeypatch) -> None:
    store.db.query(OAuthProvider).update({"client_secret": ""})
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    with patch.object(
        auth_api,
        "_resolve_oauth_client_per_field",
        wraps=auth_api._resolve_oauth_client_per_field,
    ) as resolver:
        owned = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )
    ordinary = get_google_credentials(store.user_id, store.db)

    # Per field, with the helper the connector runtime refreshes it with.
    assert (owned.client_id, owned.client_secret) == (DB_CLIENT_ID, "env-secret")
    assert [call.args[0] for call in resolver.call_args_list] == ["google"]
    # The ordinary branch keeps replacing the incomplete pair as a whole.
    assert (ordinary.client_id, ordinary.client_secret) == (
        ENV_CLIENT_ID,
        "env-secret",
    )


def test_owner_branch_needs_the_google_provider_row(store, monkeypatch) -> None:
    store.db.query(OAuthProvider).delete()
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-secret")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    owned_id = store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        500,
        "Google OAuth configuration missing",
    )
    assert exc_info.value.reason == "oauth_unconfigured"
    assert exc_info.value.oauth_account_id == owned_id
    assert get_google_credentials(store.user_id, store.db).client_id == ENV_CLIENT_ID


# --- refresh -------------------------------------------------------------


def test_refresh_honours_min_ttl(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="ten-minutes", expires_at=_future(10))
    patcher, calls = _refreshing()

    with patcher:
        kept = get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)
        refreshed = get_google_credentials(
            store.user_id,
            store.db,
            resource_owner_key=OWNER,
            min_ttl=timedelta(minutes=15),
        )

    assert kept.token == "ten-minutes"
    assert calls == ["ten-minutes"]
    assert refreshed.token == "refreshed-token"
    assert store.stored(row_id)[0] == "refreshed-token"


def test_owner_refresh_persists_only_the_owner_row(store) -> None:
    expired = _future(-5)
    ordinary_id = store.add_drive(
        owner=None, token="ordinary", provider_user_id="ordinary", expires_at=expired
    )
    owned_id = store.add_drive(
        owner=OWNER, token="owned", provider_user_id="owned", expires_at=expired
    )
    ordinary_before = store.stored(ordinary_id)
    patcher, calls = _refreshing(refresh_token="rotated-refresh-token")

    with patcher:
        creds = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )

    assert calls == ["owned"]
    assert creds.token == "refreshed-token"
    access_token, refresh_token, expires_at = store.stored(owned_id)
    assert (access_token, refresh_token) == (
        "refreshed-token",
        "rotated-refresh-token",
    )
    assert expires_at is not None
    assert expires_at.replace(tzinfo=timezone.utc) > datetime.now(timezone.utc)
    assert store.stored(ordinary_id) == ordinary_before


def test_refresh_keeps_an_unrotated_refresh_token(store) -> None:
    row_id = store.add_drive(owner=None, token="old", expires_at=_future(-5))
    patcher, _calls = _refreshing()

    with patcher:
        get_google_credentials(store.user_id, store.db)

    assert store.stored(row_id)[:2] == ("refreshed-token", "refresh-token")


_INVALID_GRANT_PAYLOAD = {
    "error": "invalid_grant",
    "error_description": "Token has been expired or revoked.",
}


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TransportError("connection reset"), "refresh_unavailable"),
        (
            RefreshError("temporarily_unavailable", retryable=True),
            "refresh_unavailable",
        ),
        (
            RefreshError(
                "invalid_grant: Token has been expired or revoked.",
                _INVALID_GRANT_PAYLOAD,
            ),
            "reauth_required",
        ),
        (
            # Older google-auth releases pass the raw response body.
            RefreshError(
                "invalid_grant: Token has been expired or revoked.",
                json.dumps(_INVALID_GRANT_PAYLOAD),
            ),
            "reauth_required",
        ),
        (
            # Without a payload the response body was not JSON.
            RefreshError("invalid_grant: Token has been expired or revoked."),
            "refresh_unavailable",
        ),
        (
            RefreshError(
                "invalid_grant: Token has been expired or revoked.",
                _INVALID_GRANT_PAYLOAD,
                retryable=True,
            ),
            "refresh_unavailable",
        ),
        (
            RefreshError(
                "invalid_client: Unauthorized",
                {"error": "invalid_client", "error_description": "Unauthorized"},
            ),
            "refresh_unavailable",
        ),
        (
            RefreshError(
                '{"error": {"code": 400}}', {"error": {"code": 400}}, retryable=False
            ),
            "refresh_unavailable",
        ),
        (RuntimeError("unexpected"), "refresh_unavailable"),
    ],
    ids=[
        "transport",
        "retryable",
        "invalid-grant",
        "invalid-grant-raw-body",
        "invalid-grant-text-only",
        "invalid-grant-retryable",
        "invalid-client",
        "structured-error",
        "unexpected",
    ],
)
def test_refresh_failures_are_classified_without_clearing_tokens(
    store, owner, error, reason
) -> None:
    expires_at = _future(-5)
    row_id = store.add_drive(owner=owner, token="old", expires_at=expires_at)
    before = store.stored(row_id)
    patcher, calls = _refreshing(error=error)

    with patcher, pytest.raises(GoogleDriveCredentialError) as exc_info:
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert calls == ["old"]
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == reason
    assert exc_info.value.oauth_account_id == row_id
    assert store.stored(row_id) == before


@pytest.mark.parametrize("owner", [None, OWNER])
def test_failed_commit_after_refresh_rolls_back(store, owner) -> None:
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.stored(row_id)
    rollbacks: list[bool] = []
    original_rollback = store.db.rollback

    def _failing_commit() -> None:
        raise OperationalError("UPDATE user_oauth", {}, Exception("locked"))

    def _recording_rollback() -> None:
        rollbacks.append(True)
        original_rollback()

    patcher, _calls = _refreshing()
    with (
        patcher,
        patch.object(store.db, "commit", _failing_commit),
        patch.object(store.db, "rollback", _recording_rollback),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    assert rollbacks == [True]
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    assert store.stored(row_id) == before


def test_row_replaced_during_refresh_is_retryable(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))

    def _replace_row_then_refresh(self: Credentials, request: Any) -> None:
        del request
        other = store.sessions()
        try:
            other.query(UserOAuth).filter(UserOAuth.id == row_id).delete()
            other.commit()
        finally:
            other.close()
        self.token = "refreshed-token"

    with (
        patch.object(Credentials, "refresh", _replace_row_then_refresh),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "refresh_unavailable"
    assert exc_info.value.oauth_account_id == row_id
    # The session was rolled back and is usable again.
    assert store.db.query(UserOAuth).count() == 0


# --- refresh through google-auth ------------------------------------------
#
# These tests run google-auth's own refresh and retry code; only the HTTP call
# under its ``requests`` transport is replaced.


class _TokenEndpoint:
    """Stands in for ``requests.Session.request`` during a token refresh."""

    def __init__(
        self,
        status: int = 200,
        body: str = "",
        *,
        content_type: str = "application/json",
        error: Exception | None = None,
    ) -> None:
        self.status = status
        self.body = body
        self.content_type = content_type
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append(
            {"method": method, "url": url, "timeout": kwargs.get("timeout")}
        )
        if self.error is not None:
            raise self.error
        response = requests.Response()
        response.status_code = self.status
        response._content = self.body.encode()
        response.headers["Content-Type"] = self.content_type
        response.url = url
        return response


@pytest.fixture
def backoff_sleeps(monkeypatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(
        _exponential_backoff, "time", SimpleNamespace(sleep=sleeps.append)
    )
    return sleeps


_GATEWAY_HTML = "<html><body><h1>502 Bad Gateway</h1></body></html>"
_UNAVAILABLE_HTML = "<html><body><h1>503 Service Unavailable</h1></body></html>"


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


# (endpoint factory, reason, token endpoint calls). google-auth retries only
# the answers it considers retryable, up to three attempts.
_REFRESH_RESPONSES = [
    pytest.param(
        lambda: _TokenEndpoint(502, _GATEWAY_HTML, content_type="text/html"),
        "refresh_unavailable",
        1,
        id="502-html",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            502, _json({"error": "bad_gateway", "error_description": "Bad Gateway"})
        ),
        "refresh_unavailable",
        1,
        id="502-json",
    ),
    pytest.param(
        lambda: _TokenEndpoint(502, _json(_INVALID_GRANT_PAYLOAD)),
        "refresh_unavailable",
        1,
        id="502-json-invalid-grant",
    ),
    pytest.param(
        lambda: _TokenEndpoint(503, _UNAVAILABLE_HTML, content_type="text/html"),
        "refresh_unavailable",
        3,
        id="503-html-after-retries",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            503, _json({"error": "backend_error", "error_description": "Unavailable"})
        ),
        "refresh_unavailable",
        3,
        id="503-json-after-retries",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            400, "<html>Bad Request</html>", content_type="text/html"
        ),
        "refresh_unavailable",
        1,
        id="400-html",
    ),
    pytest.param(
        lambda: _TokenEndpoint(400, _json(_INVALID_GRANT_PAYLOAD)),
        "reauth_required",
        1,
        id="400-invalid-grant",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            400,
            _json(
                {
                    "error": "invalid_grant",
                    "error_subtype": "invalid_rapt",
                    "error_description": "reauth related error (invalid_rapt)",
                }
            ),
        ),
        "reauth_required",
        1,
        id="400-invalid-grant-reauth-subtype",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            401,
            _json(
                {
                    "error": "invalid_client",
                    "error_description": "The OAuth client was not found.",
                }
            ),
        ),
        "refresh_unavailable",
        1,
        id="401-invalid-client",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            400,
            _json(
                {"error": "unauthorized_client", "error_description": "Unauthorized"}
            ),
        ),
        "refresh_unavailable",
        1,
        id="400-unauthorized-client",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            400,
            _json({"error": "invalid_scope", "error_description": "Bad Request"}),
        ),
        "refresh_unavailable",
        1,
        id="400-invalid-scope",
    ),
    pytest.param(
        lambda: _TokenEndpoint(
            error=requests.exceptions.ReadTimeout("Read timed out.")
        ),
        "refresh_unavailable",
        1,
        id="transport-timeout",
    ),
]


@pytest.mark.parametrize("owner", [None, OWNER])
@pytest.mark.parametrize(("make_endpoint", "reason", "attempts"), _REFRESH_RESPONSES)
def test_token_endpoint_answers_are_classified(
    store, backoff_sleeps, owner, make_endpoint, reason, attempts
) -> None:
    endpoint = make_endpoint()
    row_id = store.add_drive(owner=owner, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)

    with (
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=owner)

    # The website routes keep answering exactly as before.
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert exc_info.value.reason == reason
    assert exc_info.value.oauth_account_id == row_id
    assert isinstance(exc_info.value.__cause__, (RefreshError, TransportError))
    assert [(call["method"], call["url"]) for call in endpoint.calls] == [
        ("POST", GOOGLE_TOKEN_URI)
    ] * attempts
    assert len(backoff_sleeps) == attempts - 1
    assert store.snapshot(row_id) == before


def test_refresh_failure_log_has_no_secrets_or_response_body(
    store, backoff_sleeps, caplog
) -> None:
    row_id = store.add_drive(owner=OWNER, token="old-access", expires_at=_future(-5))
    endpoint = _TokenEndpoint(502, _GATEWAY_HTML, content_type="text/html")

    with (
        caplog.at_level(logging.ERROR, logger="xagent.web.api.cloud_storage"),
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError),
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Failed to refresh Google token (refresh_unavailable): RefreshError, "
        "token endpoint status 502, error None, retryable False"
    ]
    for secret in ("old-access", "refresh-token", "db-secret", "Bad Gateway"):
        assert all(secret not in message for message in messages)
    assert store.snapshot(row_id)[0] == "old-access"


def test_refresh_timeout_is_bounded_only_for_a_resource_owner(store) -> None:
    owned_id = store.add_drive(
        owner=OWNER, token="old", provider_user_id="owned", expires_at=_future(-5)
    )
    ordinary_id = store.add_drive(
        owner=None, token="old", provider_user_id="ordinary", expires_at=_future(-5)
    )
    endpoint = _TokenEndpoint(
        200,
        _json({"access_token": "fresh", "expires_in": 3600, "token_type": "Bearer"}),
    )

    with patch.object(requests.Session, "request", endpoint.request):
        owned = get_google_credentials(
            store.user_id, store.db, resource_owner_key=OWNER
        )
        owner_timeouts = [call["timeout"] for call in endpoint.calls]
        endpoint.calls.clear()
        ordinary = get_google_credentials(store.user_id, store.db)
        website_timeouts = [call["timeout"] for call in endpoint.calls]

    # Same bound as the connector runtime's own refresh.
    assert owner_timeouts == [10.0]
    # The website branch keeps google-auth's default transport timeout.
    assert website_timeouts == [google_auth_requests._DEFAULT_TIMEOUT]
    assert (owned.token, ordinary.token) == ("fresh", "fresh")
    assert store.stored(owned_id)[0] == "fresh"
    assert store.stored(ordinary_id)[0] == "fresh"


def test_token_request_bound_replaces_any_requested_timeout() -> None:
    endpoint = _TokenEndpoint(200, _json({"access_token": "fresh"}))

    with patch.object(requests.Session, "request", endpoint.request):
        _GoogleTokenRequest()(GOOGLE_TOKEN_URI, method="POST", timeout=3)
        _GoogleTokenRequest(timeout=10.0)(GOOGLE_TOKEN_URI, method="POST", timeout=3)
        _GoogleTokenRequest(timeout=10.0)(GOOGLE_TOKEN_URI, method="POST")

    assert [call["timeout"] for call in endpoint.calls] == [3, 10.0, 10.0]


def test_owner_refresh_timeout_is_classified_as_unavailable(store) -> None:
    row_id = store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    before = store.snapshot(row_id)
    endpoint = _TokenEndpoint(
        error=requests.exceptions.ConnectTimeout("Connection timed out.")
    )

    with (
        patch.object(requests.Session, "request", endpoint.request),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        get_google_credentials(store.user_id, store.db, resource_owner_key=OWNER)

    assert [call["timeout"] for call in endpoint.calls] == [10.0]
    assert exc_info.value.reason == "refresh_unavailable"
    assert (exc_info.value.status_code, exc_info.value.detail) == (
        401,
        RECONNECT_DETAIL,
    )
    assert store.snapshot(row_id) == before


# --- issue_google_drive_picker_config ------------------------------------


def test_issue_picker_config_for_owner_returns_three_fields(store, picker_key) -> None:
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")

    result = issue_google_drive_picker_config(
        store.db,
        user_id=store.user_id,
        resource_owner_key=OWNER,
        minimal_scopes=True,
    )

    assert result == {
        "access_token": "owned",
        "developer_key": "picker-api-key",
        "app_id": "123456789012",
    }


def test_issue_picker_config_derives_app_id_from_the_per_field_client(
    store, picker_key, monkeypatch
) -> None:
    store.db.query(OAuthProvider).update({"client_id": "", "client_secret": "s"})
    store.db.commit()
    monkeypatch.setenv("GOOGLE_CLIENT_ID", ENV_CLIENT_ID)
    store.add_drive(owner=OWNER, token="owned")

    result = issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )

    assert result["app_id"] == "999999999999"


def test_issue_picker_config_resolves_the_owner_client_once(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="old", expires_at=_future(-5))
    patcher, calls = _refreshing()

    with (
        patcher,
        patch.object(
            auth_api,
            "_resolve_oauth_client_per_field",
            wraps=auth_api._resolve_oauth_client_per_field,
        ) as resolver,
    ):
        result = issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    assert calls == ["old"]
    assert result["access_token"] == "refreshed-token"
    assert [call.args[0] for call in resolver.call_args_list] == ["google"]


def test_issue_picker_config_passes_min_ttl(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="ten-minutes", expires_at=_future(10))
    patcher, calls = _refreshing()

    with patcher:
        result = issue_google_drive_picker_config(
            store.db,
            user_id=store.user_id,
            resource_owner_key=OWNER,
            minimal_scopes=True,
            min_ttl=timedelta(minutes=15),
        )

    assert calls == ["ten-minutes"]
    assert result["access_token"] == "refreshed-token"


@pytest.mark.parametrize("owner", [None, OWNER])
def test_issue_picker_config_unconfigured_is_503_before_credentials(
    store, owner
) -> None:
    store.add_drive(owner=owner, token="token")

    with (
        patch(
            "xagent.web.api.cloud_storage.scoped_user_oauth_query",
            side_effect=AssertionError("credentials must not be read"),
        ),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=owner
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (
        503,
        PICKER_NOT_CONFIGURED_DETAIL,
    )
    assert exc_info.value.reason == "picker_not_configured"


@pytest.mark.parametrize(
    ("scope", "minimal", "reason"),
    [
        (f"{USERINFO} {DRIVE}", False, "scope_full_drive"),
        (f"{USERINFO} {DRIVE}", True, "scope_full_drive"),
        (f"{DRIVE_FILE} {DRIVE_READONLY}", True, "scope_full_drive"),
        (
            f"{DRIVE_FILE} https://www.googleapis.com/auth/drive.metadata.readonly",
            False,
            "scope_mismatch",
        ),
        (f"{USERINFO} {DRIVE_FILE} {GMAIL}", True, "scope_mismatch"),
        (USERINFO, False, "scope_drive_missing"),
        (None, True, "scope_drive_missing"),
    ],
    ids=[
        "full-drive",
        "full-drive-minimal",
        "readonly-mixed",
        "mixed-drive",
        "extra-scope-minimal",
        "drive-missing",
        "no-scope",
    ],
)
def test_issue_picker_config_rejects_scopes(
    store, picker_key, scope, minimal, reason
) -> None:
    store.add_drive(owner=OWNER, token="owned", scope=scope)

    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db,
            user_id=store.user_id,
            resource_owner_key=OWNER,
            minimal_scopes=minimal,
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (409, SCOPE_DETAIL)
    assert exc_info.value.reason == reason


def test_issue_picker_config_allows_extra_scopes_unless_minimal(
    store, picker_key
) -> None:
    store.add_drive(owner=OWNER, token="owned", scope=f"{DRIVE_FILE} {GMAIL}")

    result = issue_google_drive_picker_config(
        store.db, user_id=store.user_id, resource_owner_key=OWNER
    )

    assert result["access_token"] == "owned"


def test_issue_picker_config_reports_reasons_from_credentials(
    store, picker_key
) -> None:
    with pytest.raises(GoogleDriveCredentialError) as exc_info:
        issue_google_drive_picker_config(
            store.db, user_id=store.user_id, resource_owner_key=OWNER
        )

    assert exc_info.value.status_code == 401
    assert exc_info.value.reason == "account_not_connected"


def test_issue_picker_config_warns_about_an_app_id_from_another_project(
    store, picker_key, monkeypatch, caplog
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", "555555555555")
    store.add_drive(owner=None, token="ordinary")

    with caplog.at_level(logging.WARNING, logger="xagent.web.services.google_picker"):
        result = issue_google_drive_picker_config(store.db, user_id=store.user_id)

    assert result["app_id"] == "555555555555"
    assert any("555555555555" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_picker_route_keeps_its_detail_for_a_grant_without_drive(
    monkeypatch,
) -> None:
    monkeypatch.setenv("GOOGLE_PICKER_API_KEY", "picker-api-key")
    monkeypatch.setenv("GOOGLE_PICKER_APP_ID", "1234567890")

    with (
        patch(
            "xagent.web.api.cloud_storage.get_google_credentials",
            return_value=SimpleNamespace(token="access-token", scopes=USERINFO.split()),
        ),
        patch(
            "xagent.web.api.cloud_storage.get_google_oauth_config",
            return_value=("1234567890-client.apps.googleusercontent.com", "secret"),
        ),
        pytest.raises(GoogleDriveCredentialError) as exc_info,
    ):
        await get_google_drive_picker_config(
            db=object(),
            user=SimpleNamespace(id=1),
            response=Response(),
        )

    assert (exc_info.value.status_code, exc_info.value.detail) == (409, SCOPE_DETAIL)
    assert exc_info.value.reason == "scope_drive_missing"


@pytest.mark.asyncio
async def test_picker_route_reads_the_users_own_connection(store, picker_key) -> None:
    store.add_drive(owner=OWNER, token="owned", provider_user_id="owned")
    store.add_drive(owner=None, token="ordinary", provider_user_id="ordinary")
    response = Response()

    result = await get_google_drive_picker_config(
        account_id=None,
        db=store.db,
        user=SimpleNamespace(id=store.user_id),
        response=response,
    )

    assert result["access_token"] == "ordinary"
    assert response.headers["cache-control"] == "no-store"
