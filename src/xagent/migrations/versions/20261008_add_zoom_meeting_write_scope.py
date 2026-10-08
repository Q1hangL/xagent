"""add Zoom meeting:write:meeting scope

Revision ID: 20261008_add_zoom_meeting_write_scope
Revises: 20261008_task_auto_recovery
Create Date: 2026-10-08

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261008_add_zoom_meeting_write_scope"
down_revision: Union[str, None] = "20261008_task_auto_recovery"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PUBLIC_MCP_APPS_TABLE = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("description", sa.Text),
    sa.column("oauth_scopes", sa.JSON),
)

APP_ID = "zoom"

PREVIOUS_SCOPES = [
    "meeting:read:meeting",
    "meeting:read:list_meetings",
    "meeting:read:past_meeting",
    "cloud_recording:read:list_recording_files",
    "cloud_recording:read:meeting_transcript",
    "user:read:user",
]
# Same order as the registry's zoom entry (test_migration_fields_match_registry
# compares the lists verbatim): the write scope sits with the meeting:* group.
CURRENT_SCOPES = [
    "meeting:read:meeting",
    "meeting:read:list_meetings",
    "meeting:read:past_meeting",
    "meeting:write:meeting",
    "cloud_recording:read:list_recording_files",
    "cloud_recording:read:meeting_transcript",
    "user:read:user",
]

PREVIOUS_DESCRIPTION = (
    "Connect to Zoom to look up meetings, and read cloud recordings and transcripts."
)
CURRENT_DESCRIPTION = (
    "Connect to Zoom to schedule meetings, look up meetings, and read cloud "
    "recordings and transcripts."
)


def _columns_present(bind: sa.engine.Connection, required_columns: set[str]) -> bool:
    """Whether public_mcp_apps exists and has all of ``required_columns``.

    This migration must be a no-op (not an error) against a database whose
    schema predates these columns, or an admin's reduced-schema table.
    """
    inspector = sa.inspect(bind)
    if "public_mcp_apps" not in set(inspector.get_table_names()):
        return False
    columns = {c["name"] for c in inspector.get_columns("public_mcp_apps")}
    return required_columns.issubset(columns)


def _set_zoom_scopes(bind: sa.engine.Connection, scopes: list[str]) -> None:
    """Keep the persisted row in sync with the code registry's canonical value.

    The registry, not this row, supplies the scopes requested at authorize
    time; the write avoids a validate_builtin_public_mcp_apps drift report.
    """
    if not _columns_present(bind, {"app_id", "oauth_scopes"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID)
        .values(oauth_scopes=scopes)
    )


def _set_zoom_description_if_unchanged(
    bind: sa.engine.Connection, expected_current: str, new_value: str
) -> None:
    """Refresh the default description without clobbering a customization.

    description can be edited through the admin PATCH endpoint, so only a
    value that still equals the last-known default is replaced, in either
    direction.
    """
    if not _columns_present(bind, {"app_id", "description"}):
        return

    bind.execute(
        sa.update(PUBLIC_MCP_APPS_TABLE)
        .where(
            PUBLIC_MCP_APPS_TABLE.c.app_id == APP_ID,
            PUBLIC_MCP_APPS_TABLE.c.description == expected_current,
        )
        .values(description=new_value)
    )


# Existing user_oauth grants are deliberately left alone. Zoom's read tools
# keep working on them, and a connection that lacks meeting:write:meeting
# gets an explicit error from zoom_create_meeting, asking the user to
# disconnect Zoom and connect it again, when it is first used to create a
# meeting. Clearing the grants here would disconnect every Zoom user, readers
# included, and could not be undone by downgrade().


def upgrade() -> None:
    bind = op.get_bind()
    _set_zoom_scopes(bind, CURRENT_SCOPES)
    _set_zoom_description_if_unchanged(bind, PREVIOUS_DESCRIPTION, CURRENT_DESCRIPTION)


def downgrade() -> None:
    bind = op.get_bind()
    _set_zoom_scopes(bind, PREVIOUS_SCOPES)
    _set_zoom_description_if_unchanged(bind, CURRENT_DESCRIPTION, PREVIOUS_DESCRIPTION)
