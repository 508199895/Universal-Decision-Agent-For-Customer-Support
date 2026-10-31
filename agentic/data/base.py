"""SQLAlchemy declarative base for Agent persistence models."""

from __future__ import annotations

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    def to_dict(self, *, exclude: set[str] | None = None) -> dict:
        excluded = exclude or set()
        return {
            column.key: getattr(self, column.key)
            for column in sa_inspect(type(self)).mapper.column_attrs
            if column.key not in excluded
        }

    def __repr__(self) -> str:
        columns = ", ".join(
            f"{column.key}={getattr(self, column.key)!r}"
            for column in sa_inspect(type(self)).mapper.column_attrs
        )
        return f"{type(self).__name__}({columns})"
