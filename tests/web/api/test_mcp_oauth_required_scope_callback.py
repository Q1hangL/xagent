"""Connect and callback honor an MCP OAuth configuration's ``required_scope``.

The authorization still requests the whole ``scope``. The callback stores the
scope the provider granted; when the server configures ``required_scope`` and
the grant does not cover it, the callback fails with ``insufficient_scope``
instead of reporting a connection that could never be used. A server without
``required_scope`` keeps storing narrower grants as before.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from xagent.web.api import mcp as mcp_api
from xagent.web.api.mcp import (
    MCPOAuthConnectRequest,
    connect_mcp_oauth,
    mcp_oauth_callback,
)
from xagent.web.builtin_mcp_registry import (
    ATLASSIAN_MCP_OAUTH_SCOPE,
    get_builtin_public_mcp_app,
)
from xagent.web.models import MCPOAuthClient, MCPOAuthFlowState, MCPOAuthGrant
from xagent.web.models.database import Base
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User
from xagent.web.services.mcp_oauth import (
    MCPOAuthClientRegistration,
    select_mcp_oauth_grants,
)

MCP_URL = "https://mcp.atlassian.com/v2/mcp"
ISSUER = "https://auth.atlassian.com/tenant"
ACTOR_OWNER = "actor:workspace:alice"
BITBUCKET_SCOPES = ("read:bitbucket:agent-interface", "write:bitbucket:agent-interface")


def _scope_without(*left_out: str) -> str:
    return " ".join(
        scope for scope in ATLASSIAN_MCP_OAUTH_SCOPE.split() if scope not in left_out
    )


GRANTED_WITHOUT_BITBUCKET = _scope_without(*BITBUCKET_SCOPES)


def _atlassian_auth() -> dict:
    app = get_builtin_public_mcp_app("atlassian")
    assert app is not None
    return dict(app["launch_config"]["auth"])


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'required-scope-callback.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="alice", password_hash="x", is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    yield db, user
    db.close()
    engine.dispose()


def _callback_request(state: str) -> Request:
    cookie = (
        f"{mcp_api.MCP_OAUTH_STATE_COOKIE}="
        f"{mcp_api._mcp_oauth_state_cookie_value(state)}"
    )
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/mcp/oauth/callback",
            "query_string": f"code=auth-code&state={state}".encode(),
            "headers": [(b"cookie", cookie.encode())],
        }
    )


def _redirect_query(response) -> dict[str, list[str]]:
    return parse_qs(urlparse(response.headers["location"]).query)


def _add_server(db, user: User, auth: dict) -> MCPServer:
    server = MCPServer.from_config(
        {
            "name": "atlassian",
            "managed": "external",
            "transport": "streamable_http",
            "url": MCP_URL,
            "auth": auth,
        }
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    db.add(
        UserMCPServer(
            user_id=user.id,
            mcpserver_id=server.id,
            is_owner=False,
            is_active=True,
        )
    )
    db.commit()
    return server


def _add_flow(
    db,
    user: User,
    auth: dict,
    *,
    state: str,
    requested_scope: str,
) -> MCPServer:
    server = _add_server(db, user, auth)
    client = MCPOAuthClient(
        mcp_server_id=server.id,
        issuer=ISSUER,
        authorization_endpoint="https://auth.atlassian.com/authorize",
        token_endpoint="https://auth.atlassian.com/oauth/token",
        client_id="atlassian-dynamic-client",
        token_endpoint_auth_method="none",
        redirect_uri="https://api.xagent.test/api/mcp/oauth/callback",
        metadata_json={"revocation_endpoint": "https://auth.atlassian.com/revoke"},
    )
    db.add(client)
    db.flush()
    generation = (
        db.query(UserMCPServer.lifecycle_generation)
        .filter(
            UserMCPServer.user_id == user.id,
            UserMCPServer.mcpserver_id == server.id,
        )
        .scalar()
    )
    db.add(
        MCPOAuthFlowState(
            state=state,
            mcp_server_id=server.id,
            user_id=user.id,
            association_lifecycle_generation=generation,
            mcp_oauth_client_id=client.id,
            resource_owner_key=ACTOR_OWNER,
            issuer=ISSUER,
            resource=MCP_URL,
            scope=requested_scope,
            code_verifier=mcp_api.encrypt_value("verifier-123"),
            redirect_after="/tools",
            expires_at=mcp_api._utc_now() + timedelta(minutes=10),
        )
    )
    db.commit()
    return server


def _token_response(monkeypatch, token_data: dict) -> list:
    """Answer the code exchange with ``token_data``; record compensations."""
    revoked: list = []

    async def fake_exchange(**_kwargs):
        return dict(token_data)

    async def record_revoke(snapshot):
        revoked.append(snapshot)

    monkeypatch.setattr(mcp_api, "_exchange_mcp_oauth_code", fake_exchange)
    monkeypatch.setattr(
        mcp_api, "_revoke_mcp_oauth_issued_token_externally", record_revoke
    )
    return revoked


ISSUED_TOKENS = {
    "access_token": "issued-access-token",
    "refresh_token": "issued-refresh-token",
    "token_type": "Bearer",
    "expires_in": 28800,
}


@pytest.mark.asyncio
async def test_callback_stores_an_atlassian_grant_without_bitbucket_scopes(
    db_session, monkeypatch
):
    db, user = db_session
    auth = _atlassian_auth()
    server = _add_flow(
        db, user, auth, state="state-11", requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE
    )
    revoked = _token_response(
        monkeypatch, {**ISSUED_TOKENS, "scope": GRANTED_WITHOUT_BITBUCKET}
    )

    response = await mcp_oauth_callback(_callback_request("state-11"), db)

    assert response.status_code == 307
    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    assert revoked == []
    grant = db.query(MCPOAuthGrant).one()
    # The grant records what the provider granted, not what was requested.
    assert grant.scope == GRANTED_WITHOUT_BITBUCKET
    assert grant.resource_owner_key == ACTOR_OWNER
    selected = select_mcp_oauth_grants(
        db,
        server_id=server.id,
        user_id=user.id,
        auth_config=auth,
        resource_owner_key=ACTOR_OWNER,
    )
    assert [candidate.id for candidate in selected] == [grant.id]


@pytest.mark.asyncio
async def test_callback_without_a_scope_in_the_token_response_keeps_the_request(
    db_session, monkeypatch
):
    """A token response without ``scope`` was granted what was requested."""
    db, user = db_session
    _add_flow(
        db,
        user,
        _atlassian_auth(),
        state="state-no-scope",
        requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE,
    )
    _token_response(monkeypatch, ISSUED_TOKENS)

    response = await mcp_oauth_callback(_callback_request("state-no-scope"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    assert db.query(MCPOAuthGrant).one().scope == ATLASSIAN_MCP_OAUTH_SCOPE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "left_out", ("email", "offline_access", "read:account", "read:me")
)
async def test_callback_refuses_a_grant_without_a_basic_identity_scope(
    db_session, monkeypatch, caplog, left_out
):
    db, user = db_session
    _add_flow(
        db,
        user,
        _atlassian_auth(),
        state="state-missing",
        requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE,
    )
    revoked = _token_response(
        monkeypatch,
        {**ISSUED_TOKENS, "scope": _scope_without(*BITBUCKET_SCOPES, left_out)},
    )

    with caplog.at_level(logging.WARNING, logger=mcp_api.logger.name):
        response = await mcp_oauth_callback(_callback_request("state-missing"), db)

    assert response.status_code == 307
    query = _redirect_query(response)
    assert "mcp_oauth_success" not in query
    assert query["mcp_oauth_error"] == ["insufficient_scope"]
    assert db.query(MCPOAuthGrant).count() == 0
    # The tokens that will not be kept are handed to the revocation path.
    assert [snapshot.access_token for snapshot in revoked] == ["issued-access-token"]
    assert revoked[0].refresh_token == "issued-refresh-token"
    assert "reason=insufficient_scope" in caplog.text
    assert f"missing_scope={left_out}" in caplog.text
    assert "issued-access-token" not in caplog.text


@pytest.mark.asyncio
async def test_callback_without_required_scope_still_stores_a_narrower_grant(
    db_session, monkeypatch
):
    """Unchanged behavior without ``required_scope``: the narrower grant is
    stored and reported as connected, even though it does not cover the
    request and so is not selected as usable."""
    db, user = db_session
    auth = {"type": "mcp_oauth", "scope": ATLASSIAN_MCP_OAUTH_SCOPE}
    server = _add_flow(
        db, user, auth, state="state-plain", requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE
    )
    revoked = _token_response(
        monkeypatch, {**ISSUED_TOKENS, "scope": GRANTED_WITHOUT_BITBUCKET}
    )

    response = await mcp_oauth_callback(_callback_request("state-plain"), db)

    assert _redirect_query(response) == {"mcp_oauth_success": ["1"]}
    assert revoked == []
    assert db.query(MCPOAuthGrant).one().scope == GRANTED_WITHOUT_BITBUCKET
    assert (
        select_mcp_oauth_grants(
            db,
            server_id=server.id,
            user_id=user.id,
            auth_config=auth,
            resource_owner_key=ACTOR_OWNER,
        )
        == []
    )


@pytest.mark.asyncio
async def test_callback_with_an_invalid_required_scope_reports_the_configuration(
    db_session, monkeypatch
):
    db, user = db_session
    _add_flow(
        db,
        user,
        {**_atlassian_auth(), "required_scope": "read:me manage:jira-configuration"},
        state="state-invalid",
        requested_scope=ATLASSIAN_MCP_OAUTH_SCOPE,
    )
    revoked = _token_response(
        monkeypatch, {**ISSUED_TOKENS, "scope": ATLASSIAN_MCP_OAUTH_SCOPE}
    )

    response = await mcp_oauth_callback(_callback_request("state-invalid"), db)

    assert _redirect_query(response)["mcp_oauth_error"] == ["invalid_configuration"]
    assert db.query(MCPOAuthGrant).count() == 0
    assert len(revoked) == 1


@pytest.mark.asyncio
async def test_connect_rejects_an_invalid_required_scope_before_any_provider_call(
    db_session, monkeypatch
):
    db, user = db_session
    server = _add_server(
        db,
        user,
        {**_atlassian_auth(), "required_scope": "read:me manage:jira-configuration"},
    )

    async def unexpected_discovery(*_args, **_kwargs):
        raise AssertionError("discovery must not run for an invalid configuration")

    monkeypatch.setattr(mcp_api, "_discover_mcp_oauth_for_server", unexpected_discovery)

    with pytest.raises(HTTPException) as exc_info:
        await connect_mcp_oauth(
            server.id,
            MCPOAuthConnectRequest(redirect_after="/tools"),
            user,
            db,
            "application/json",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "invalid_configuration"
    assert db.query(MCPOAuthFlowState).count() == 0


@pytest.mark.asyncio
async def test_connect_requests_the_whole_scope_not_the_required_part(
    db_session, monkeypatch
):
    db, user = db_session
    server = _add_server(db, user, _atlassian_auth())
    discovery = await _stub_discovery_and_registration(monkeypatch)

    response = await connect_mcp_oauth(
        server.id,
        MCPOAuthConnectRequest(redirect_after="/tools"),
        user,
        db,
        "application/json",
    )

    query = parse_qs(urlparse(json.loads(response.body)["authorization_url"]).query)
    assert query["scope"] == [ATLASSIAN_MCP_OAUTH_SCOPE]
    assert "required_scope" not in query
    assert db.query(MCPOAuthFlowState).one().scope == ATLASSIAN_MCP_OAUTH_SCOPE
    assert discovery.registrations == 1


async def _stub_discovery_and_registration(monkeypatch):
    state = SimpleNamespace(registrations=0)
    discovery = SimpleNamespace(
        resource=MCP_URL,
        scopes=(),
        protected_resource=SimpleNamespace(authorization_servers=(ISSUER,)),
        authorization_server=SimpleNamespace(
            issuer=ISSUER,
            authorization_endpoint="https://auth.atlassian.com/authorize",
            token_endpoint="https://auth.atlassian.com/oauth/token",
            registration_endpoint=f"{ISSUER}/dcr/register",
            client_id_metadata_document_supported=False,
            raw={"issuer": ISSUER},
        ),
    )

    async def fake_discovery(*_args, **_kwargs):
        return discovery

    async def fake_registration(*_args, **_kwargs):
        state.registrations += 1
        return MCPOAuthClientRegistration(
            client_id="atlassian-dynamic-client",
            token_endpoint_auth_method="none",
        )

    monkeypatch.setattr(mcp_api, "_discover_mcp_oauth_for_server", fake_discovery)
    monkeypatch.setattr(mcp_api, "register_mcp_oauth_public_client", fake_registration)
    return state
