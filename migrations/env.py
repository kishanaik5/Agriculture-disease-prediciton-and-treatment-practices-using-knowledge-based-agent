import sys
from pathlib import Path
import os

# Get the absolute path to the project root
project_root = Path(__file__).parents[1].resolve()
shared_backend_path = project_root / "SharedBackend" / "src"

# Add paths to sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(shared_backend_path) not in sys.path:
    sys.path.insert(0, str(shared_backend_path))

# Also add /app path for Docker environment
if os.path.exists("/app") and "/app" not in sys.path:
    sys.path.insert(0, "/app")
if os.path.exists("/app/SharedBackend/src") and "/app/SharedBackend/src" not in sys.path:
    sys.path.insert(0, "/app/SharedBackend/src")

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from SharedBackend.managers import BaseSchema
from app.config import init_settings
from app.models.scan import AnalysisReport  # Import models to register them

config = context.config

if config.config_file_name is not None: fileConfig(config.config_file_name)

# Initialize settings
settings = init_settings()

# for 'autogenerate' support
target_metadata = BaseSchema.metadata

config.set_main_option("sqlalchemy.url", settings.SQLALCHEMY_DATABASE_URI.replace("+asyncpg", ""))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=settings.DB_SCHEMA,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
            version_table_schema=settings.DB_SCHEMA
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
