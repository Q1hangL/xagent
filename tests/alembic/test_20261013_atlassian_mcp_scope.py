"""Tests for the migration that pins the remote Atlassian connector's scope."""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)

VERSIONS = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"
CHILD_TABLES = ("mcp_oauth_flow_states", "mcp_oauth_grants", "mcp_oauth_clients")


def _load(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, VERSIONS / filename)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load("20261013_atlassian_mcp_scope.py", "atlassian_mcp_scope_migration")


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _run(connection, migration, *steps):
    with patch.object(migration, "op", _operations(connection)):
        for step in steps:
            getattr(migration, step)()


def _schema(metadata):
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("name", String),
        Column("transport", String),
        Column("launch_config", JSON),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
    )
    links = Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("user_id", Integer),
        Column("mcpserver_id", Integer),
        Column("is_owner", Boolean, default=False),
    )
    for table_name in CHILD_TABLES:
        Table(
            table_name,
            metadata,
            Column("id", Integer, primary_key=True),
            Column("mcp_server_id", Integer),
        )
    return apps, servers, links


def _setup(tmp_path, migration, *, launch_config=None):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, servers, links = _schema(metadata)
    metadata.create_all(engine)
    if launch_config is not False:
        with engine.begin() as connection:
            connection.execute(
                apps.insert().values(
                    app_id="atlassian",
                    name="Atlassian (Jira, Confluence, Bitbucket)",
                    transport="streamable_http",
                    launch_config=(
                        migration.LEGACY_LAUNCH_CONFIG
                        if launch_config is None
                        else launch_config
                    ),
                )
            )
    return engine, metadata, apps, servers, links


def _add_server(
    connection,
    metadata,
    servers,
    links,
    *,
    server_id,
    auth,
    name="atlassian",
    url="https://mcp.atlassian.com/v2/mcp",
    transport="streamable_http",
    owner=False,
):
    connection.execute(
        servers.insert().values(
            id=server_id, name=name, transport=transport, url=url, auth=auth
        )
    )
    connection.execute(
        links.insert().values(user_id=7, mcpserver_id=server_id, is_owner=owner)
    )
    for table_name in CHILD_TABLES:
        connection.execute(
            metadata.tables[table_name].insert().values(mcp_server_id=server_id)
        )


def _child_count(connection, metadata, server_id):
    total = connection.execute(
        select(func.count()).where(
            metadata.tables["user_mcpservers"].c.mcpserver_id == server_id
        )
    ).scalar_one()
    for table_name in CHILD_TABLES:
        table = metadata.tables[table_name]
        total += connection.execute(
            select(func.count()).where(table.c.mcp_server_id == server_id)
        ).scalar_one()
    return total


def _launch_config(connection, apps):
    return connection.execute(
        select(apps.c.launch_config).where(apps.c.app_id == "atlassian")
    ).scalar_one()


def _server_ids(connection, servers):
    return set(connection.execute(select(servers.c.id)).scalars())


def test_upgrade_pins_scope_and_removes_the_legacy_shared_server(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="unrelated",
            url="https://mcp.example.com/mcp",
            auth={"type": "mcp_oauth"},
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {2}
        assert _child_count(connection, metadata, 1) == 0
        assert _child_count(connection, metadata, 2) == 4


def test_upgrade_matches_a_shared_server_whose_auth_also_names_the_app(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            transport="STREAMABLE_HTTP",
            auth={**migration.LEGACY_AUTH, "app_id": "atlassian"},
        )

        _run(connection, migration, "upgrade")

        assert _server_ids(connection, servers) == set()
        assert _child_count(connection, metadata, 1) == 0


def test_upgrade_keeps_a_user_owned_server_under_the_name(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
            owner=True,
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}
        assert _child_count(connection, metadata, 1) == 4


def test_upgrade_keeps_servers_that_are_not_the_catalog_definition(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        # A custom OAuth configuration (its own client), a foreign URL and a
        # foreign transport are not rows the catalog connect path writes.
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth={**migration.LEGACY_AUTH, "client_id": "operator-client"},
        )
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="Atlassian",
            auth=migration.LEGACY_AUTH,
        )

        _run(connection, migration, "upgrade")

        assert _server_ids(connection, servers) == {1, 2}

    for case, server_kwargs in (
        ("url", {"url": "https://evil.example/mcp"}),
        ("transport", {"transport": "sse"}),
    ):
        case_path = tmp_path / case
        case_path.mkdir()
        engine, metadata, _apps, servers, links = _setup(case_path, migration)
        with engine.begin() as connection:
            _add_server(
                connection,
                metadata,
                servers,
                links,
                server_id=1,
                auth=migration.LEGACY_AUTH,
                **server_kwargs,
            )
            _run(connection, migration, "upgrade")
            assert _server_ids(connection, servers) == {1}, case


def test_upgrade_keeps_a_server_already_recreated_with_the_scope(tmp_path):
    """Re-running the upgrade (or running it after a connect under the new
    catalog) must not disconnect users of the current definition."""
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=None
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.SCOPED_AUTH,
        )

        _run(connection, migration, "upgrade", "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}
        assert _child_count(connection, metadata, 1) == 4


def test_upgrade_without_a_server_row_only_updates_the_catalog(tmp_path):
    migration = _load_migration()
    engine, _metadata, apps, servers, _links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == set()


def test_upgrade_accepts_an_older_provenance_version(tmp_path):
    migration = _load_migration()
    launch_config = dict(migration.LEGACY_LAUNCH_CONFIG)
    launch_config["builtin_provenance"] = {
        **migration.BUILTIN_PROVENANCE,
        "version": 0,
    }
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=launch_config
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == set()


def test_upgrade_leaves_an_operator_catalog_row_and_its_server_alone(tmp_path):
    migration = _load_migration()
    operator_launch = {
        "url": "https://mcp.atlassian.com/v2/mcp",
        "auth": {"type": "mcp_oauth"},
    }
    engine, metadata, apps, servers, links = _setup(
        tmp_path, migration, launch_config=operator_launch
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == operator_launch
        assert _server_ids(connection, servers) == {1}
        assert _child_count(connection, metadata, 1) == 4


def test_upgrade_without_a_catalog_row_touches_no_server(tmp_path):
    migration = _load_migration()
    engine, metadata, _apps, servers, links = _setup(
        tmp_path, migration, launch_config=False
    )
    with engine.begin() as connection:
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.LEGACY_AUTH,
        )

        _run(connection, migration, "upgrade")

        assert _server_ids(connection, servers) == {1}


def test_downgrade_restores_the_catalog_and_removes_the_scoped_server(tmp_path):
    migration = _load_migration()
    engine, metadata, apps, servers, links = _setup(tmp_path, migration)
    with engine.begin() as connection:
        _run(connection, migration, "upgrade")
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=1,
            auth=migration.SCOPED_AUTH,
        )
        _add_server(
            connection,
            metadata,
            servers,
            links,
            server_id=2,
            name="unrelated",
            url="https://mcp.example.com/mcp",
            auth=migration.SCOPED_AUTH,
        )

        _run(connection, migration, "downgrade")

        assert _launch_config(connection, apps) == migration.LEGACY_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {2}
        assert _child_count(connection, metadata, 1) == 0


def test_upgrade_and_downgrade_are_noops_without_the_tables(tmp_path):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _run(connection, migration, "upgrade", "downgrade")


def test_upgrade_skips_servers_when_the_server_table_lacks_identity_columns(
    tmp_path,
):
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("launch_config", JSON),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            apps.insert().values(
                app_id="atlassian", launch_config=migration.LEGACY_LAUNCH_CONFIG
            )
        )
        connection.execute(servers.insert().values(id=1, name="atlassian"))

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}


@pytest.mark.parametrize(
    "policy",
    [
        {"headers": {"X-Debug": "1"}},
        {"timeout": 120},
        {"env": {"HTTP_PROXY": "http://proxy.internal:3128"}},
        {"managed": "internal"},
    ],
    ids=("headers", "timeout", "env", "managed"),
)
def test_upgrade_deletes_the_catalog_row_even_after_an_admin_edited_it(
    tmp_path, policy
):
    """Deliberate (as in the HubSpot and Zendesk switches): an administrator's
    edits to the shared catalog row go with it. Under the old auth the actor
    projection rejects the row regardless of those edits."""
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps, _servers, links = _schema(metadata)
    metadata.remove(metadata.tables["mcp_servers"])
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String, unique=True),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
        Column("headers", JSON),
        Column("env", JSON),
        Column("timeout", Integer),
        Column("managed", String),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            apps.insert().values(
                app_id="atlassian",
                name="Atlassian (Jira, Confluence, Bitbucket)",
                transport="streamable_http",
                launch_config=migration.LEGACY_LAUNCH_CONFIG,
            )
        )
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
                **{"managed": "external", **policy},
            )
        )
        connection.execute(
            links.insert().values(user_id=7, mcpserver_id=1, is_owner=False)
        )

        _run(connection, migration, "upgrade")

        assert _server_ids(connection, servers) == set()
        assert _child_count(connection, metadata, 1) == 0


def test_upgrade_deletes_no_server_when_ownership_cannot_be_checked(tmp_path):
    """Fail closed: without user_mcpservers.is_owner a user-owned server
    cannot be told apart from the catalog row."""
    migration = _load_migration()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    metadata = MetaData()
    apps = Table(
        "public_mcp_apps",
        metadata,
        Column("app_id", String, primary_key=True),
        Column("launch_config", JSON),
    )
    servers = Table(
        "mcp_servers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("transport", String),
        Column("url", String),
        Column("auth", JSON),
    )
    Table(
        "user_mcpservers",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("mcpserver_id", Integer),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            apps.insert().values(
                app_id="atlassian", launch_config=migration.LEGACY_LAUNCH_CONFIG
            )
        )
        connection.execute(
            servers.insert().values(
                id=1,
                name="atlassian",
                transport="streamable_http",
                url=migration.SERVER_URL,
                auth=migration.LEGACY_AUTH,
            )
        )

        _run(connection, migration, "upgrade")

        assert _launch_config(connection, apps) == migration.SCOPED_LAUNCH_CONFIG
        assert _server_ids(connection, servers) == {1}


def test_scoped_launch_config_matches_the_registry():
    """The migration holds a frozen copy of the catalog value; this catches
    drift between the two."""
    from xagent.web.builtin_mcp_registry import (
        ATLASSIAN_MCP_OAUTH_SCOPE,
        get_builtin_public_mcp_app,
    )

    migration = _load_migration()
    registry_row = get_builtin_public_mcp_app("atlassian")

    assert registry_row is not None
    assert migration.SCOPE == ATLASSIAN_MCP_OAUTH_SCOPE
    assert migration.SCOPED_LAUNCH_CONFIG == registry_row["launch_config"]
    assert registry_row["transport"] == migration.SERVER_TRANSPORT


def test_legacy_launch_config_is_what_the_seed_inserted():
    migration = _load_migration()
    seed = _load("20260920_seed_atlassian_mcp_app.py", "seed_atlassian_for_scope")

    assert migration.LEGACY_LAUNCH_CONFIG == seed.ROW["launch_config"]
    assert migration.SERVER_URL == seed.ROW["launch_config"]["url"]
