"""Add governed dataset registry and isolated client intelligence portals."""

from alembic import op
from research_lab.models import (
    ClientTopic,
    ClientWorkspace,
    DatasetDefinition,
    IntelligenceBrief,
)

revision = "0003_materials_intelligence_portal"
down_revision = "0002_candidate_assessments"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    DatasetDefinition.__table__.create(bind, checkfirst=True)
    ClientWorkspace.__table__.create(bind, checkfirst=True)
    ClientTopic.__table__.create(bind, checkfirst=True)
    IntelligenceBrief.__table__.create(bind, checkfirst=True)


def downgrade():
    bind = op.get_bind()
    IntelligenceBrief.__table__.drop(bind, checkfirst=True)
    ClientTopic.__table__.drop(bind, checkfirst=True)
    ClientWorkspace.__table__.drop(bind, checkfirst=True)
    DatasetDefinition.__table__.drop(bind, checkfirst=True)
