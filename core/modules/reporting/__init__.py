from .export import export_backtest_report


def export_trade_review_html(*args, **kwargs):
    """Load the large trade-review module only when the report is requested."""
    from .trade_review import export_trade_review_html as _export_trade_review_html

    return _export_trade_review_html(*args, **kwargs)

__all__ = [
    "export_backtest_report",
    "export_trade_review_html",
]
