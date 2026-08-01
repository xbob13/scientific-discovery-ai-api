"""Add autonomous commerce lifecycle and research institution network."""

import sqlalchemy as sa

from alembic import op
from research_lab.models import CommerceEvent, ResearchInstitution

revision = "0005_autonomous_network"
down_revision = "0004_enterprise_engine"
branch_labels = None
depends_on = None


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    bind = op.get_bind()
    existing = {item["name"] for item in sa.inspect(bind).get_columns(table_name)}
    if column.name not in existing:
        op.add_column(table_name, column)


def upgrade():
    bind = op.get_bind()
    CommerceEvent.__table__.create(bind, checkfirst=True)
    ResearchInstitution.__table__.create(bind, checkfirst=True)
    _add_column_if_missing("subscription_activations", sa.Column("external_customer_id", sa.String(255)))
    _add_column_if_missing("subscription_activations", sa.Column("external_subscription_id", sa.String(255)))
    _add_column_if_missing(
        "subscription_activations",
        sa.Column("status", sa.String(40), nullable=False, server_default="active"),
    )
    _add_column_if_missing(
        "subscription_activations", sa.Column("current_period_end", sa.DateTime(timezone=True))
    )
    _add_column_if_missing("subscription_activations", sa.Column("last_event_at", sa.DateTime(timezone=True)))


def downgrade():
    bind = op.get_bind()
    ResearchInstitution.__table__.drop(bind, checkfirst=True)
    CommerceEvent.__table__.drop(bind, checkfirst=True)
