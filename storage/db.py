import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from storage.models import Base

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create every ORM-defined table in PostgreSQL. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


def apply_pending_migrations():
    """Apply idempotent schema fixes that older DB dumps may be missing."""
    statements = [
        "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS cleaned_transcript TEXT",
        "ALTER TABLE predictions ALTER COLUMN call_id DROP NOT NULL",
    ]
    try:
        with engine.connect() as conn:
            for stmt in statements:
                try:
                    conn.execute(text(stmt))
                except Exception as e:
                    logger.warning(f"Migration step skipped ({stmt!r}): {e}")
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not apply pending migrations: {e}")


@contextmanager
def get_session():
    """Yield a SQLAlchemy session that commits on success and rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
