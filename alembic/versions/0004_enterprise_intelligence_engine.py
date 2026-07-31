"""Add self-service entitlements, client mandates, patents, and governed outreach."""

from alembic import op
from research_lab.models import (
    OutreachMessage,
    PatentDocument,
    ProspectAccount,
    ResearchMandate,
    SubscriptionActivation,
    WorkspaceEntitlement,
)

revision = "0004_enterprise_engine"
down_revision = "0003_materials_intelligence_portal"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    WorkspaceEntitlement.__table__.create(bind, checkfirst=True)
    ResearchMandate.__table__.create(bind, checkfirst=True)
    SubscriptionActivation.__table__.create(bind, checkfirst=True)
    PatentDocument.__table__.create(bind, checkfirst=True)
    ProspectAccount.__table__.create(bind, checkfirst=True)
    OutreachMessage.__table__.create(bind, checkfirst=True)


def downgrade():
    bind = op.get_bind()
    OutreachMessage.__table__.drop(bind, checkfirst=True)
    ProspectAccount.__table__.drop(bind, checkfirst=True)
    PatentDocument.__table__.drop(bind, checkfirst=True)
    SubscriptionActivation.__table__.drop(bind, checkfirst=True)
    ResearchMandate.__table__.drop(bind, checkfirst=True)
    WorkspaceEntitlement.__table__.drop(bind, checkfirst=True)
