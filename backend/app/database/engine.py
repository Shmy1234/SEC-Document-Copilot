from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(url: str) -> Engine:
    # One pool for the process. Supabase caps connections, so the pool is bounded
    # and pre-pinged to survive the pooler closing idle sockets.
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
