"""Actor OAuth callback pages tell the member what to do next.

Every page an actor flow's callback renders when it does not connect must be
actionable (what happened, what to do, a way to close the window), must keep
the single-use flow semantics, and must log a reason code without any value
of the flow itself.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.utils.encryption import encrypt_value
from xagent.web import mcp_apps
from xagent.web.api import auth as auth_api
from xagent.web.api.auth import create_access_token, generic_oauth_callback
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User
from xagent.web.models.user_oauth import UserOAuth

OWNER = "actor:workspace-41:member-alice"
OLD_ACTOR_ERROR = "Invalid or expired actor OAuth flow"
CLOSE_BUTTON = '<button type="button" onclick="window.close()">'
AUTH_LOGGER = "xagent.web.api.auth"
CALLBACK_LOG_PREFIX = "OAuth callback did not connect"
APPS = {
    "outlook": {
        "name": "Outlook",
        "transport": "oauth",
        "provider_name": "microsoft",
        "oauth_scopes": ["Mail.ReadWrite"],
        "launch_config": {"command": "outlook"},
    },
    "calendar": {
        "name": "Calendar",
        "transport": "oauth",
        "provider_name": "custom",
        "oauth_scopes": [],
        "launch_config": {"command": "calendar"},
    },
}
DECLINED_DESCRIPTION = (
    "AADSTS65004: User declined to consent to access the app.\r\n"
    "Trace ID: 0b8c2f3e-0000-0000-0000-000000000000\r\n"
    "Correlation ID: 5d1e9a7c-0000-0000-0000-000000000000"
)
DECLINED_URI = "https://login.microsoftonline.com/error?code=65004"


class _ProviderResponse:
    def __init__(self, data: dict[str, object], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._data


@pytest.fixture
def oauth_db(tmp_path, monkeypatch):
    registry_lookup = mcp_apps.get_builtin_execution_fields_and_optional_scopes

    def test_registry(app_id: str):
        if app_id in APPS:
            return APPS[app_id], []
        return registry_lookup(app_id)

    monkeypatch.setattr(
        mcp_apps, "get_builtin_execution_fields_and_optional_scopes", test_registry
    )

    engine = create_engine(f"sqlite:///{tmp_path / 'actor-oauth-feedback.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with factory() as db:
        user = User(username="workspace-account", password_hash="hash")
        db.add(user)
        db.flush()
        for app_id, execution in APPS.items():
            db.add(
                PublicMCPApp(
                    app_id=app_id,
                    name=execution["name"],
                    description=execution["name"],
                    transport="oauth",
                    provider_name=execution["provider_name"],
                    oauth_scopes=list(execution["oauth_scopes"]),
                    launch_config=dict(execution["launch_config"]),
                    is_visible_in_connector=True,
                )
            )
            server = MCPServer(
                name=execution["name"],
                description=execution["name"],
                managed="external",
                transport="oauth",
                auth={"app_id": app_id, "provider": execution["provider_name"]},
            )
            db.add(server)
            db.flush()
            db.add(
                UserMCPServer(
                    user_id=int(user.id),
                    mcpserver_id=int(server.id),
                    is_owner=False,
                    is_active=True,
                )
            )
        db.commit()
        db.refresh(user)
        yield db, user
    engine.dispose()


@pytest.fixture
def token_endpoint(monkeypatch) -> Mock:
    post = Mock(
        return_value=_ProviderResponse(
            {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }
        )
    )
    monkeypatch.setattr(auth_api.requests, "post", post)
    monkeypatch.setattr(
        auth_api.requests,
        "get",
        Mock(return_value=_ProviderResponse({"id": "account", "mail": "a@x.test"})),
    )
    return post


@pytest.fixture
def callback_logs(caplog):
    caplog.set_level(logging.WARNING, logger=AUTH_LOGGER)

    def messages() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == AUTH_LOGGER
            and record.getMessage().startswith(CALLBACK_LOG_PREFIX)
        ]

    return messages


def _db_provider(provider: str) -> SimpleNamespace:
    if provider == "microsoft":
        return SimpleNamespace(
            client_id=encrypt_value("microsoft-client-id"),
            client_secret=encrypt_value("microsoft-client-secret"),
            auth_url="https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            token_url="https://login.microsoftonline.com/common/oauth2/v2.0/token",
            userinfo_url="https://graph.microsoft.com/v1.0/me",
            redirect_uri="https://xagent.example/api/auth/microsoft/callback",
            default_scopes=["User.Read"],
            user_id_path="id",
            email_path="mail",
        )
    return SimpleNamespace(
        client_id=encrypt_value("client-id"),
        client_secret=encrypt_value("client-secret"),
        auth_url="https://provider.example/authorize",
        token_url="https://provider.example/token",
        userinfo_url="https://provider.example/me",
        redirect_uri="https://xagent.example/api/auth/custom/callback",
        default_scopes=["profile.read"],
        user_id_path="id",
        email_path="mail",
    )


class _Flow(NamedTuple):
    provider: str
    state: str
    cookie: tuple[str, str]
    nonce: str


def _start(db: Session, user: User, app_id: str = "outlook") -> _Flow:
    provider = APPS[app_id]["provider_name"]
    response = auth_api.start_builtin_oauth_for_resource_owner(
        provider=provider,
        app_id=app_id,
        user=user,
        resource_owner_key=OWNER,
        redirect="https://app.example/connectors",
        db=db,
        db_provider=_db_provider(provider),
    )
    db.commit()
    state = parse_qs(urlparse(response.headers["location"]).query)["state"][0]
    jar = SimpleCookie()
    jar.load(response.headers["set-cookie"])
    ((cookie_name, morsel),) = [
        (name, morsel)
        for name, morsel in jar.items()
        if name.startswith("xagent_actor_oauth_")
    ]
    payload = auth_api.verify_token(state)
    assert payload is not None
    return _Flow(
        provider=provider,
        state=state,
        cookie=(cookie_name, morsel.value),
        nonce=payload["actor_flow_nonce"],
    )


def _callback(
    db: Session,
    flow: _Flow,
    *,
    state: str | None = None,
    cookie: tuple[str, str] | None = None,
    without_cookie: bool = False,
    **query: str,
):
    params = {"state": state or flow.state}
    if "error" not in query:
        params["code"] = "authorization-code-value"
    params.update(query)
    name, value = cookie or flow.cookie
    request = SimpleNamespace(
        query_params=params,
        cookies={} if without_cookie else {name: value},
    )
    return generic_oauth_callback(
        flow.provider, request, db, _db_provider(flow.provider)
    )


def _resign(flow: _Flow, **changes: object) -> str:
    payload = auth_api.verify_token(flow.state)
    assert payload is not None
    payload.update(changes)
    return create_access_token(data=payload, expires_delta=timedelta(minutes=10))


def _open_flows(db: Session) -> int:
    return db.query(ActorOAuthFlowState).count()


def _actor_grants(db: Session) -> int:
    return db.query(UserOAuth).filter(UserOAuth.resource_owner_key == OWNER).count()


def _body(response) -> str:
    return response.body.decode()


def _assert_actionable(response) -> str:
    body = _body(response)
    assert response.status_code == 400
    assert response.headers["cache-control"] == "no-store"
    assert CLOSE_BUTTON in body
    assert "select Connect again" in body
    assert OLD_ACTOR_ERROR not in body
    return body


def _assert_no_flow_values(text: str, flow: _Flow) -> None:
    for secret in (
        flow.state,
        flow.cookie[1],
        flow.nonce,
        flow.nonce[:24],
        OWNER,
        "authorization-code-value",
        "new-access",
        "new-refresh",
    ):
        assert secret not in text


def test_declined_microsoft_consent_explains_how_to_retry(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        error="consent_required",
        error_description=DECLINED_DESCRIPTION,
        error_uri=DECLINED_URI,
    )

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "choose Accept" in body
    assert "Microsoft 365 admin" in body
    assert "<code>consent_required, AADSTS65004</code>" in body
    for raw in ("User declined", "Trace ID", "Correlation ID", DECLINED_URI):
        assert raw not in body
    _assert_no_flow_values(body, flow)
    assert _open_flows(db) == 0
    token_endpoint.assert_not_called()
    assert _actor_grants(db) == 0

    (log,) = callback_logs()
    assert "provider=microsoft app_id=outlook actor_flow=true" in log
    assert "reason=provider_error" in log
    assert "provider_error=consent_required" in log
    assert "provider_error_code=AADSTS65004" in log
    assert "User declined" not in log
    _assert_no_flow_values(log, flow)


@pytest.mark.parametrize(
    ("description", "codes"),
    [
        (None, "access_denied"),
        (
            "AADSTS90094: The grant requires admin permission.",
            "access_denied, AADSTS90094",
        ),
    ],
)
def test_cancelled_microsoft_sign_in_asks_to_accept_without_admin_link(
    oauth_db, token_endpoint, description, codes
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    query = {"error": "access_denied", "error_subcode": "cancel"}
    if description is not None:
        query["error_description"] = description

    response = _callback(db, flow, **query)

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "Microsoft 365 admin" in body
    assert f"<code>{codes}</code>" in body
    assert "href=" not in body
    assert "adminconsent" not in body
    assert _open_flows(db) == 0
    token_endpoint.assert_not_called()


def test_other_providers_get_neutral_not_granted_guidance(
    oauth_db, token_endpoint
) -> None:
    db, user = oauth_db
    flow = _start(db, user, app_id="calendar")

    response = _callback(
        db, flow, error="access_denied", error_description="AADSTS65004: declined"
    )

    body = _assert_actionable(response)
    assert "Permission was not granted" in body
    assert "approve the requested access" in body
    assert "Microsoft" not in body
    assert "<code>access_denied</code>" in body


def test_rejected_microsoft_request_shows_category_and_code(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        error="invalid_client",
        error_description=(
            "AADSTS650051: The application needs to be added to the tenant "
            "first. Trace ID: 1234"
        ),
    )

    body = _assert_actionable(response)
    assert "The sign-in request was rejected" in body
    assert "<code>invalid_client, AADSTS650051</code>" in body
    assert "contact your Microsoft 365 admin or support" in body
    assert "added to the tenant" not in body
    (log,) = callback_logs()
    assert "provider_error=invalid_client" in log
    assert "provider_error_code=AADSTS650051" in log
    assert "added to the tenant" not in log


def test_temporarily_unavailable_provider_asks_to_retry_later(
    oauth_db, token_endpoint
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(db, flow, error="temporarily_unavailable")

    body = _assert_actionable(response)
    assert "temporarily unavailable" in body
    assert "in a few minutes" in body
    assert "<code>temporarily_unavailable</code>" in body


@pytest.mark.parametrize(
    ("error", "shown"),
    [
        ("interaction_required", "interaction_required"),
        ("<script>alert(1)</script>", "unknown_error"),
        ("consent required", "unknown_error"),
        ("x" * 65, "unknown_error"),
    ],
)
def test_unrecognized_provider_error_is_generic_and_never_echoed(
    oauth_db, token_endpoint, callback_logs, error, shown
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(db, flow, error=error)

    body = _assert_actionable(response)
    assert "The sign-in did not complete" in body
    assert f"<code>{shown}</code>" in body
    (log,) = callback_logs()
    assert f"provider_error={shown} " in log
    if shown == "unknown_error":
        assert error not in body
        assert error not in log
        assert "<script" not in body


def test_reloading_a_declined_sign_in_window_says_it_was_used(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    query = {
        "error": "consent_required",
        "error_description": DECLINED_DESCRIPTION,
        "error_uri": DECLINED_URI,
    }

    first = _callback(db, flow, **query)
    reload = _callback(db, flow, **query)

    assert "Permission was not granted" in _assert_actionable(first)
    body = _assert_actionable(reload)
    assert "This sign-in window has already been used" in body
    assert "already shows as connected, there is nothing else to do" in body
    token_endpoint.assert_not_called()
    first_log, reload_log = callback_logs()
    assert "reason=provider_error" in first_log
    assert "reason=flow_used_or_expired" in reload_log
    assert "provider_error_code=AADSTS65004" in reload_log
    _assert_no_flow_values(reload_log, flow)


def test_replaying_a_completed_sign_in_keeps_the_grant(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    first = _callback(db, flow)
    replay = _callback(db, flow)

    assert first.status_code == 200
    body = _assert_actionable(replay)
    assert "This sign-in window has already been used" in body
    assert "Connected Successfully" not in body
    assert token_endpoint.call_count == 1
    assert _actor_grants(db) == 1
    (log,) = callback_logs()
    assert "reason=flow_used_or_expired provider_error=-" in log
    _assert_no_flow_values(log, flow)


def test_expired_sign_in_window_says_it_was_used(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    db.query(ActorOAuthFlowState).update(
        {
            ActorOAuthFlowState.expires_at: datetime.now(timezone.utc)
            - timedelta(seconds=1)
        }
    )
    db.commit()

    response = _callback(db, flow)

    assert "This sign-in window has already been used" in _assert_actionable(response)
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert "reason=flow_used_or_expired" in log


@pytest.mark.parametrize(
    ("cookie_mode", "reason"),
    [("missing", "cookie_missing"), ("wrong", "cookie_mismatch")],
)
def test_sign_in_from_another_browser_cannot_finish(
    oauth_db, token_endpoint, callback_logs, cookie_mode, reason
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    if cookie_mode == "missing":
        response = _callback(db, flow, without_cookie=True)
    else:
        response = _callback(db, flow, cookie=(flow.cookie[0], "other-browser"))

    body = _assert_actionable(response)
    assert "This sign-in window can't finish connecting" in body
    assert "from the same browser" in body
    assert _open_flows(db) == 1
    token_endpoint.assert_not_called()
    (log,) = callback_logs()
    assert f"reason={reason} " in log
    assert "other-browser" not in log
    _assert_no_flow_values(log, flow)


def test_provider_error_with_failed_actor_check_logs_both(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)

    response = _callback(
        db,
        flow,
        without_cookie=True,
        error="consent_required",
        error_description=DECLINED_DESCRIPTION,
    )

    assert "can't finish connecting" in _assert_actionable(response)
    assert _open_flows(db) == 1
    (log,) = callback_logs()
    assert "reason=cookie_missing" in log
    assert "provider_error=consent_required" in log
    assert "provider_error_code=AADSTS65004" in log


def _bad_claims(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, actor_flow_nonce="not-a-nonce")


def _bad_owner(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, resource_owner_key=OWNER)


def _missing_user(db: Session, user: User, flow: _Flow) -> str:
    return _resign(flow, user_id=int(user.id) + 1000)


def _inactive_link(db: Session, user: User, flow: _Flow) -> str:
    db.query(UserMCPServer).update({UserMCPServer.is_active: False})
    db.commit()
    return flow.state


def _catalog_drift(db: Session, user: User, flow: _Flow) -> str:
    db.query(PublicMCPApp).filter_by(app_id="outlook").one().provider_name = "other"
    db.commit()
    return flow.state


@pytest.mark.parametrize(
    ("break_flow", "reason"),
    [
        (_bad_claims, "claims_invalid"),
        (_bad_owner, "owner_invalid"),
        (_missing_user, "user_missing"),
        (_inactive_link, "link_invalid"),
        (_catalog_drift, "catalog_invalid"),
    ],
)
def test_each_actor_check_logs_its_reason(
    oauth_db, token_endpoint, callback_logs, break_flow, reason
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    state = break_flow(db, user, flow)

    response = _callback(db, flow, state=state)

    assert "can't finish connecting" in _assert_actionable(response)
    assert _open_flows(db) == 1
    token_endpoint.assert_not_called()
    assert _actor_grants(db) == 0
    (log,) = callback_logs()
    assert f"reason={reason} " in log
    _assert_no_flow_values(log, flow)


def test_link_removed_during_exchange_cannot_finish(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    exchange = token_endpoint.return_value

    def disconnect_during_exchange(*args, **kwargs):
        db.query(UserMCPServer).update({UserMCPServer.is_active: False})
        db.commit()
        return exchange

    token_endpoint.side_effect = disconnect_during_exchange

    response = _callback(db, flow)

    assert "can't finish connecting" in _assert_actionable(response)
    assert token_endpoint.call_count == 1
    assert _actor_grants(db) == 0
    (log,) = callback_logs()
    assert "reason=link_changed_after_exchange" in log
    _assert_no_flow_values(log, flow)


@pytest.mark.parametrize("first_attempt", ["declined", "other-browser"])
def test_selecting_connect_again_after_a_failed_attempt_connects(
    oauth_db, token_endpoint, first_attempt
) -> None:
    db, user = oauth_db
    failed = _start(db, user)
    if first_attempt == "declined":
        _assert_actionable(_callback(db, failed, error="consent_required"))
    else:
        _assert_actionable(_callback(db, failed, without_cookie=True))

    retry = _start(db, user)
    response = _callback(db, retry)

    assert response.status_code == 200
    assert "Connected Successfully" in _body(response)
    assert _actor_grants(db) == 1


def _ordinary_state(user: User) -> str:
    return create_access_token(
        data={
            "type": "oauth_state",
            "user_id": user.id,
            "provider": "microsoft",
            "app_id": "outlook",
            "redirect": None,
        },
        expires_delta=timedelta(minutes=10),
    )


def _ordinary_callback(db: Session, state: str, **query: str):
    request = SimpleNamespace(query_params={"state": state, **query}, cookies={})
    return generic_oauth_callback("microsoft", request, db, _db_provider("microsoft"))


def test_ordinary_flow_provider_error_pages_are_unchanged(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    state = _ordinary_state(user)

    denied = _ordinary_callback(db, state, error="access_denied")
    admin = _ordinary_callback(
        db,
        state,
        error="access_denied",
        error_subcode="cancel",
        error_description="AADSTS90094: The grant requires admin permission.",
    )

    assert denied.status_code == 400
    assert _body(denied) == "<h1>Error: access_denied</h1>"
    assert "cache-control" not in denied.headers
    assert admin.status_code == 400
    assert "Admin approval required" in _body(admin)
    assert "adminconsent" in _body(admin)
    token_endpoint.assert_not_called()
    denied_log, admin_log = callback_logs()
    assert "actor_flow=false reason=provider_error" in denied_log
    assert "provider_error_code=AADSTS90094" in admin_log


def test_invalid_state_keeps_its_page_and_logs_reason(
    oauth_db, token_endpoint, callback_logs
) -> None:
    db, user = oauth_db
    flow = _start(db, user)
    header, payload, signature = flow.state.split(".")
    tampered = ".".join(
        (header, payload, ("a" if signature[0] != "a" else "b") + signature[1:])
    )

    response = _callback(db, flow, state=tampered)

    assert response.status_code == 400
    assert _body(response) == "<h1>Error: Invalid or expired state</h1>"
    assert _open_flows(db) == 1
    (log,) = callback_logs()
    assert "actor_flow=unknown reason=state_invalid" in log
    assert tampered not in log
