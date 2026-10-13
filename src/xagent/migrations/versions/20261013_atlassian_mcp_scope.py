"""request a fixed scope subset for the remote Atlassian MCP connector

Revision ID: 20261013_atlassian_mcp_scope
Revises: 20261012_zendesk_oauth
Create Date: 2026-10-13

Atlassian's protected-resource metadata lists 38 scopes (1023 characters once
normalized) and its 401 challenge carries no scope, so the connector, which
requested every discovered scope, exceeded the 1000-character MCP OAuth scope
limit and could not start authorization. The catalog row now sets
``launch_config.auth.scope`` to a fixed subset.

The shared ``atlassian`` server row the connect path created still carries the
old auth, which the canonical-definition check rejects, so it is deleted
together with its OAuth flow states, grants, client registrations and user
links, as the HubSpot and Zendesk switches did. The next connect recreates the
row from the catalog and registers a fresh client. Users who had connected
Atlassian must connect again. Rows in other tables that reference the deleted
server with ON DELETE CASCADE are removed with it on databases that enforce
foreign keys.
"""

from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

from xagent.builtin_identity import builtin_provenance_identity

revision: str = "20261013_atlassian_mcp_scope"
down_revision: Union[str, None] = "20261012_zendesk_oauth"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APP_ID = "atlassian"
SERVER_TRANSPORT = "streamable_http"
SERVER_URL = "https://mcp.atlassian.com/v2/mcp"
BUILTIN_PROVENANCE = {
    "registry": "xagent",
    "app_id": APP_ID,
    "version": 1,
}
# Frozen copy of builtin_mcp_registry.ATLASSIAN_MCP_OAUTH_SCOPE at the time this
# migration was written (tests/alembic pins the two against each other).
SCOPE = (
    "email offline_access read:account read:bitbucket:agent-interface "
    "read:confluence:agent-interface read:jira:agent-interface read:me "
    "search:confluence:agent-interface search:jira:agent-interface "
    "search:rovo:agent-interface write:bitbucket:agent-interface "
    "write:confluence:agent-interface write:jira:agent-interface"
)
LEGACY_AUTH = {"type": "mcp_oauth"}
SCOPED_AUTH = {"type": "mcp_oauth", "scope": SCOPE}
LEGACY_LAUNCH_CONFIG = {
    "url": SERVER_URL,
    "auth": LEGACY_AUTH,
    "builtin_provenance": BUILTIN_PROVENANCE,
}
SCOPED_LAUNCH_CONFIG = {
    "url": SERVER_URL,
    "auth": SCOPED_AUTH,
    "builtin_provenance": BUILTIN_PROVENANCE,
}

PUBLIC_MCP_APPS = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("launch_config", sa.JSON),
)
MCP_SERVERS = sa.table(
    "mcp_servers",
    sa.column("id", sa.Integer),
    sa.column("name", sa.String),
    sa.column("transport", sa.String),
    sa.column("url", sa.String),
    sa.column("auth", sa.JSON),
)
USER_MCPSERVERS = sa.table(
    "user_mcpservers",
    sa.column("mcpserver_id", sa.Integer),
    sa.column("is_owner", sa.Boolean),
)
# Every table whose rows belong to one mcp_servers row.
SERVER_CHILD_TABLES = (
    ("mcp_oauth_flow_states", "mcp_server_id"),
    ("mcp_oauth_grants", "mcp_server_id"),
    ("mcp_oauth_clients", "mcp_server_id"),
    ("user_mcpservers", "mcpserver_id"),
)


def _columns(bind: sa.engine.Connection, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if table_name not in set(inspector.get_table_names()):
        return set()
    return {column["name"] for column in inspector.get_columns(table_name)}


def _owned_launch(value: object) -> bool:
    return isinstance(value, dict) and builtin_provenance_identity(
        value.get("builtin_provenance")
    ) == builtin_provenance_identity(BUILTIN_PROVENANCE)


def _set_owned_launch_config(
    bind: sa.engine.Connection, launch_config: dict[str, Any]
) -> bool:
    """Write the catalog launch config if this app's row is the builtin one.

    Returns whether the row exists and carries the builtin provenance marker.
    For such a row the execution fields are read from the code registry, so
    this write only keeps the persisted copy free of drift. A row without the
    marker is an operator's own app and is left alone, as is every server row
    (the caller stops).
    """
    if not {"app_id", "launch_config"}.issubset(_columns(bind, "public_mcp_apps")):
        return False
    row = (
        bind.execute(
            sa.select(PUBLIC_MCP_APPS.c.launch_config).where(
                PUBLIC_MCP_APPS.c.app_id == APP_ID
            )
        )
        .mappings()
        .first()
    )
    if row is None or not _owned_launch(row["launch_config"]):
        return False
    if row["launch_config"] != launch_config:
        bind.execute(
            sa.update(PUBLIC_MCP_APPS)
            .where(PUBLIC_MCP_APPS.c.app_id == APP_ID)
            .values(launch_config=launch_config)
        )
    return True


def _delete_catalog_servers_with_auth(
    bind: sa.engine.Connection, auth: dict[str, Any]
) -> None:
    """Delete the shared server row the catalog connect path created.

    That row is named after the app id, uses the catalog transport and URL,
    stores exactly the catalog auth (an ``app_id`` key some rows carry is
    ignored, as the canonical-definition check ignores it) and has no owning
    user link. A user-created server under the same name has an owner, and a
    server with other auth keys (a client id, another scope) is a custom
    OAuth configuration; neither is touched. Without the columns that prove
    either, nothing is deleted.
    """
    if not {"id", "name", "transport", "url", "auth"}.issubset(
        _columns(bind, "mcp_servers")
    ) or not {"mcpserver_id", "is_owner"}.issubset(_columns(bind, "user_mcpservers")):
        return
    rows = (
        bind.execute(
            sa.select(
                MCP_SERVERS.c.id,
                MCP_SERVERS.c.transport,
                MCP_SERVERS.c.url,
                MCP_SERVERS.c.auth,
            ).where(MCP_SERVERS.c.name == APP_ID)
        )
        .mappings()
        .all()
    )
    for row in rows:
        stored_auth = row["auth"]
        if not isinstance(stored_auth, dict):
            continue
        comparable_auth = {
            key: value for key, value in stored_auth.items() if key != "app_id"
        }
        if (
            str(row["transport"] or "").lower() != SERVER_TRANSPORT
            or row["url"] != SERVER_URL
            or comparable_auth != auth
        ):
            continue
        server_id = int(row["id"])
        owned = bind.execute(
            sa.select(USER_MCPSERVERS.c.mcpserver_id).where(
                USER_MCPSERVERS.c.mcpserver_id == server_id,
                USER_MCPSERVERS.c.is_owner.is_(True),
            )
        ).first()
        if owned is not None:
            continue
        # Be explicit for SQLite deployments where migration connections may
        # not have foreign-key cascade enforcement enabled.
        for table_name, server_column in SERVER_CHILD_TABLES:
            if server_column not in _columns(bind, table_name):
                continue
            child = sa.table(table_name, sa.column(server_column, sa.Integer))
            bind.execute(sa.delete(child).where(child.c[server_column] == server_id))
        bind.execute(sa.delete(MCP_SERVERS).where(MCP_SERVERS.c.id == server_id))


def upgrade() -> None:
    bind = op.get_bind()
    if not _set_owned_launch_config(bind, SCOPED_LAUNCH_CONFIG):
        return
    _delete_catalog_servers_with_auth(bind, LEGACY_AUTH)


def downgrade() -> None:
    bind = op.get_bind()
    if not _set_owned_launch_config(bind, LEGACY_LAUNCH_CONFIG):
        return
    _delete_catalog_servers_with_auth(bind, SCOPED_AUTH)
