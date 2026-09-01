from .base import BAR_COLUMNS, BaseStrategy


def build_strategy(*args, **kwargs):
    """Lazy-import build_strategy to avoid circular imports with pipeline."""
    from .factory import build_strategy as _build
    return _build(*args, **kwargs)


__all__ = [
    "BaseStrategy",
    "BAR_COLUMNS",
    "build_strategy",
]
