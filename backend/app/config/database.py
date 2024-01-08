from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy import create_engine

from .settings import settings

engine = create_engine(settings.DATABASE_URI)

# Creating an engine that handles the DB connection
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
)

# The Base model which all model declarations will extend
Base = declarative_base()
Base.query = SessionLocal.query_property()
