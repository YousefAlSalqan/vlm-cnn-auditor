from alembic import context
from sqlalchemy import create_engine

from auditor.config import Settings
from auditor.db import Base

config = context.config


def run(connection):
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url.get_secret_value(),
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
elif config.attributes.get("connection") is not None:
    run(config.attributes["connection"])
else:
    engine = create_engine(Settings().database_url.get_secret_value())
    with engine.connect() as connection:
        run(connection)
    engine.dispose()
