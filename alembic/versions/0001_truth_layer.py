"""Initial canonical truth layer."""

from alembic import op
from research_lab import models  # noqa: F401
from research_lab.db import Base

revision = "0001_truth_layer"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    Base.metadata.create_all(bind)


def downgrade():
    bind = op.get_bind()
    Base.metadata.drop_all(bind)
