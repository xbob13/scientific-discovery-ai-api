"""Add evidence-backed candidate assessments."""

from alembic import op
from research_lab.models import CandidateAssessment

revision = "0002_candidate_assessments"
down_revision = "0001_truth_layer"
branch_labels = None
depends_on = None


def upgrade():
    CandidateAssessment.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    CandidateAssessment.__table__.drop(op.get_bind(), checkfirst=True)
