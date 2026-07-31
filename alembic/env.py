from logging.config import fileConfig

from sqlalchemy import Column, MetaData, String, Table, engine_from_config, inspect, pool, text

from alembic import context
from research_lab import models  # noqa: F401
from research_lab.config import get_settings
from research_lab.db import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)
target_metadata = Base.metadata


def ensure_version_table_capacity(connection):
    """Existing revision identifiers exceed Alembic's default 32-character column."""
    inspector = inspect(connection)
    if not inspector.has_table("alembic_version"):
        Table(
            "alembic_version",
            MetaData(),
            Column("version_num", String(64), primary_key=True, nullable=False),
        ).create(connection)
        return
    columns = {column["name"]: column for column in inspector.get_columns("alembic_version")}
    version_type = columns.get("version_num", {}).get("type")
    if (
        connection.dialect.name == "postgresql"
        and getattr(version_type, "length", 64) < 64
    ):
        connection.execute(
            text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(64)")
        )


def run_migrations_offline():
    context.configure(
        url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        ensure_version_table_capacity(connection)
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_offline() if context.is_offline_mode() else run_migrations_online()
