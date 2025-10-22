"""Foreign key resolver for mapping human-readable fields to database IDs."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from tracker.config.logger import get_logger
from tracker.data.orm.base import Base

logger = get_logger(__name__)


class FKResolver:
    """Foreign key resolver that maps human-readable fields to database IDs.

    Efficiently caches lookups to avoid repeated database queries
    when processing many rows (e.g., CSV ingestion).
    """

    def __init__(self, session: Session):
        """Initialize the FKResolver with a SQLAlchemy session."""
        self.session = session
        # {(Model, ("field1", "field2")): {(v1, v2) -> id}}
        self._cache: dict[
            tuple[type[Base], tuple[str, ...]], dict[tuple[str, ...], int]
        ] = defaultdict(dict)

    def resolve(
        self,
        model: type[Base],
        fields: str | Sequence[str],
        values: Any | Sequence[Any],
        required: bool = False,
    ) -> int | None:
        """Resolve a value (or tuple of values) to a database ID.

        Args:
            model: SQLAlchemy model.
            fields: Column name or list/tuple of column names for lookup.
            values: Single value or tuple of values corresponding to the fields.
            required: If True, raises if no match is found.

        Returns:
            ID if found, else None.
        """
        if not values or (isinstance(values, str) and values.strip() == ""):
            return None

        if isinstance(fields, str):
            fields = (fields,)
            values = (values,)

        if len(fields) != len(values):
            raise ValueError("Number of fields and values must match.")

        key = (model, fields)
        if key not in self._cache:
            self._preload_cache(model, fields)

        fk_id = self._cache[key].get(values)
        if fk_id is None and required:
            raise ValueError(
                f"Could not resolve {model.__name__} with {dict(zip(fields, values, strict=False))}"
            )
        return fk_id

    def _preload_cache(self, model: type[Base], fields: tuple[str, ...]):
        try:
            results = self.session.query(model).all()
            mapping = {
                tuple(getattr(obj, f) for f in fields): obj.id for obj in results
            }
            self._cache[(model, fields)] = mapping
            logger.debug(
                f"Preloaded {len(mapping)} entries for {model.__name__} by {fields}"
            )
        except Exception as e:
            logger.error(
                f"Failed to preload composite FK cache for {model.__name__} {fields}: {e}"
            )
            self._cache[(model, fields)] = {}

    def clear_cache(self):
        """Clear the FKResolver cache."""
        self._cache.clear()
        logger.debug("Cleared FKResolver cache.")
