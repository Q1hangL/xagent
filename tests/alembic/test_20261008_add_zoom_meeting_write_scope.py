"""Tests for adding the Zoom meeting:write:meeting OAuth scope."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

_VERSIONS_DIR = Path(__file__).parent.parent.parent / "src/xagent/migrations/versions"


def _load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _VERSIONS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_migration_module():
    return _load_module(
        "20261008_add_zoom_meeting_write_scope.py",
        "add_zoom_meeting_write_scope_migration",
    )


def _load_seed_migration_module():
    return _load_module("20260730_seed_zoom_mcp_app.py", "seed_zoom_mcp_app_migration")


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(
    connection, description: str | None, scopes: list[str], with_description=True
):
    description_column = "description TEXT," if with_description else ""
    connection.execute(
        text(
            f"""
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                {description_column}
                oauth_scopes JSON
            )
            """
        )
    )
    description_col = ", description" if with_description else ""
    description_val = ", :description" if with_description else ""
    connection.execute(
        text(
            f"INSERT INTO public_mcp_apps (app_id, oauth_scopes{description_col}) "
            f"VALUES ('zoom', :scopes{description_val}), "
            f"('hubspot', :other_scopes{description_val})"
        ),
        {
            "scopes": json.dumps(scopes),
            "other_scopes": json.dumps(["crm.objects.contacts.read"]),
            "description": description,
        },
    )


def _row(connection, app_id: str = "zoom"):
    row = connection.execute(
        text(
            "SELECT description, oauth_scopes FROM public_mcp_apps WHERE app_id=:app_id"
        ),
        {"app_id": app_id},
    ).first()
    description, scopes = row[0], row[1]
    return description, json.loads(scopes) if isinstance(scopes, str) else scopes


def _seeded_install(connection, migration):
    _create_table(
        connection,
        description=migration.PREVIOUS_DESCRIPTION,
        scopes=migration.PREVIOUS_SCOPES,
    )


def test_upgrade_adds_write_scope_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _seeded_install(connection, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES
        assert "meeting:write:meeting" in scopes
        # Other connectors' rows are untouched.
        assert _row(connection, "hubspot")[1] == ["crm.objects.contacts.read"]


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _seeded_install(connection, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == migration.CURRENT_DESCRIPTION
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _seeded_install(connection, migration)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == migration.PREVIOUS_DESCRIPTION
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_preserves_admin_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description="Our Zoom connector",
            scopes=migration.PREVIOUS_SCOPES,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            description, scopes = _row(connection)
            assert description == "Our Zoom connector"
            assert scopes == migration.CURRENT_SCOPES
            migration.downgrade()
        description, scopes = _row(connection)
        assert description == "Our Zoom connector"
        assert scopes == migration.PREVIOUS_SCOPES


def test_upgrade_without_description_column_still_updates_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection,
            description=None,
            scopes=migration.PREVIOUS_SCOPES,
            with_description=False,
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        scopes = connection.execute(
            text("SELECT oauth_scopes FROM public_mcp_apps WHERE app_id='zoom'")
        ).scalar()
        assert json.loads(scopes) == migration.CURRENT_SCOPES


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()


def test_upgrade_leaves_existing_zoom_grants_connected(tmp_path):
    """Existing grants keep the read tools working; zoom_create_meeting asks
    for a reconnect when a grant lacks the new scope. Clearing tokens here
    would disconnect every Zoom user and could not be reverted."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _seeded_install(connection, migration)
        connection.execute(
            text(
                "CREATE TABLE user_oauth (id INTEGER PRIMARY KEY, user_id INTEGER, "
                "provider VARCHAR(50), access_token VARCHAR, refresh_token VARCHAR, "
                "scope TEXT)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO user_oauth (user_id, provider, access_token, "
                "refresh_token, scope) VALUES "
                "(1, 'zoom', 'zoom-token', 'zoom-refresh', 'meeting:read:meeting')"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        row = connection.execute(
            text(
                "SELECT access_token, refresh_token, scope FROM user_oauth "
                "WHERE provider='zoom'"
            )
        ).first()
        assert tuple(row) == ("zoom-token", "zoom-refresh", "meeting:read:meeting")


def test_fresh_install_chain_upgrades_and_downgrades_cleanly(tmp_path):
    """The seed migration's frozen copy already carries the new values, so on
    a fresh database this migration is a no-op on upgrade, and the full
    downgrade chain still removes the seeded row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    seed = _load_seed_migration_module()
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE oauth_providers (id INTEGER PRIMARY KEY, "
                "provider_name VARCHAR(50) NOT NULL UNIQUE, name VARCHAR(100) NOT "
                "NULL, client_id VARCHAR(500) NOT NULL, client_secret VARCHAR(500) "
                "NOT NULL, auth_url VARCHAR(500) NOT NULL, token_url VARCHAR(500) "
                "NOT NULL, redirect_uri VARCHAR(500), userinfo_url VARCHAR(500), "
                "user_id_path VARCHAR(100), email_path VARCHAR(100), "
                "default_scopes JSON)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE public_mcp_apps (id INTEGER PRIMARY KEY, app_id "
                "VARCHAR(100) NOT NULL UNIQUE, name VARCHAR(200) NOT NULL, "
                "description TEXT, icon VARCHAR(1000), transport VARCHAR(50) NOT "
                "NULL DEFAULT 'oauth', provider_name VARCHAR(50), category "
                "VARCHAR(100), oauth_scopes JSON, is_visible_in_connector BOOLEAN "
                "NOT NULL DEFAULT 1, launch_config JSON)"
            )
        )
        operations = _operations(connection)
        with (
            patch.object(seed, "op", operations),
            patch.object(migration, "op", operations),
        ):
            seed.upgrade()
            migration.upgrade()
            assert _row(connection) == (
                migration.CURRENT_DESCRIPTION,
                migration.CURRENT_SCOPES,
            )
            migration.downgrade()
            assert _row(connection) == (
                migration.PREVIOUS_DESCRIPTION,
                migration.PREVIOUS_SCOPES,
            )
            seed.downgrade()
        remaining = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='zoom'")
        ).scalar()
        assert remaining == 0


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "zoom"
    )
    assert registry_row["oauth_scopes"] == migration.CURRENT_SCOPES
    assert registry_row["description"] == migration.CURRENT_DESCRIPTION


def test_previous_fields_are_the_current_fields_without_the_write_scope():
    """PREVIOUS_* is what the seed migration inserted before this change; a
    downgrade must restore exactly that, not an approximation of it."""
    migration = _load_migration_module()
    assert migration.PREVIOUS_SCOPES == [
        scope for scope in migration.CURRENT_SCOPES if scope != "meeting:write:meeting"
    ]
    assert migration.PREVIOUS_DESCRIPTION == (
        "Connect to Zoom to look up meetings, and read cloud recordings and transcripts."
    )
