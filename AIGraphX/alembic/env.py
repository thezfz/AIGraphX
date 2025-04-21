import os
import sys
from logging.config import fileConfig
from typing import Optional, Union, cast

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add project root to sys.path
# This allows Alembic to find the 'aigraphx' package if needed for models later
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Prioritize DATABASE_URL environment variable ---
DB_URL: Optional[str] = os.getenv("DATABASE_URL")
USING_ENV_URL = False

if DB_URL:
    print(f"Using DATABASE_URL from environment variable.")
    USING_ENV_URL = True
else:
    # If environment variable is NOT set, THEN load from config
    print("DATABASE_URL environment variable not found, loading from config...")
    try:
        # Import application configuration AFTER adding root to path
        from aigraphx.core import config

        # Check configuration structure (Pydantic Settings or direct attribute)
        if hasattr(config, "settings") and hasattr(config.settings, "DATABASE_URL"):
            DB_URL = str(config.settings.DATABASE_URL)  # Ensure string type
            print(f"Loaded DB_URL from config.settings.")
        elif hasattr(config, "DATABASE_URL"):
            DB_URL = str(getattr(config, "DATABASE_URL", ""))  # Ensure string type
            print(f"Loaded DB_URL from config module attribute.")
        else:
            # If neither structure is found, set DB_URL to None or empty string
            # to trigger the final check below.
            DB_URL = None
            print("DATABASE_URL not found in expected config structures.")

        if not DB_URL:  # Check if loading from config resulted in an empty URL
            print("Warning: Loaded DATABASE_URL from config is empty or not found.")

    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load DATABASE_URL from config: {e}")
        DB_URL = None  # Ensure DB_URL is None if config loading fails

# --- Final check and adjustment ---
if not DB_URL:
    # If DB_URL is still None or empty after checking env and config
    raise ImportError(
        "DATABASE_URL is not set in environment variables and could not be loaded from config."
    )

# Ensure DB_URL is a string before proceeding
DB_URL = str(DB_URL)

# --- Adjust DB_URL scheme for psycopg (v3) ---
if DB_URL.startswith("postgresql://"):
    adjusted_url = DB_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    # Only print if adjustment happened
    if adjusted_url != DB_URL:
        print(f"Adjusted DB_URL scheme to: {adjusted_url}")
    DB_URL = adjusted_url
elif DB_URL.startswith("postgresql+psycopg://"):
    # Already correct, no adjustment needed, but print for consistency if needed
    # print(f"DB_URL already uses correct scheme: {DB_URL}")
    pass
else:
    print(
        f"Warning: DB_URL scheme ('{DB_URL.split('://')[0]}://') might not be compatible with psycopg v3."
    )


# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
alembic_config = context.config

# Set the database URL programmatically
# This overrides the sqlalchemy.url from alembic.ini
alembic_config.set_main_option("sqlalchemy.url", DB_URL)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if alembic_config.config_file_name is not None:
    fileConfig(alembic_config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = None  # type: ignore # Keep as None for now, using raw SQL

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=DB_URL,  # Use the DB_URL directly
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Create engine using the DB_URL from config
    connectable = engine_from_config(
        {"sqlalchemy.url": DB_URL},  # Pass config dict directly
        prefix="sqlalchemy.",  # Keep prefix for other potential options
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
