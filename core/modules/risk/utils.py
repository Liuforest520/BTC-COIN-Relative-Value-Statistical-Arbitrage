from core.modules.models import OrderSide


def group_by_id(items):
    groups = {}
    for item in items:
        group_id = getattr(item, "group_id", None)
        groups.setdefault(group_id, []).append(item)
    return groups


def value_of(value):
    if hasattr(value, "value"):
        return value.value
    return value


def bar_price(bars, exchange, symbol, field="open"):
    exchange_bars = bars.get(exchange, {})
    bar = exchange_bars.get(symbol)
    if bar is None:
        return None
    if isinstance(bar, dict):
        return bar[field]

    columns = ["ts", "open", "high", "close", "low", "volume"]
    return dict(zip(columns, bar))[field]


def order_notional(order, bars):
    price = order.price
    if price is None:
        price = bar_price(bars, order.exchange, order.symbol, "open")
    if price is None or order.quantity is None:
        return None
    return abs(float(price) * float(order.quantity))


def trade_notional(trade):
    return abs(float(trade.price) * float(trade.quantity))


def split_long_short(items, notional_func):
    long_value = 0.0
    short_value = 0.0

    for item in items:
        side = value_of(item.side)
        notional = notional_func(item)
        if notional is None:
            return None, None
        if side == OrderSide.BUY.value:
            long_value += notional
        elif side == OrderSide.SELL.value:
            short_value += notional

    return long_value, short_value


def deviation(long_value, short_value):
    base = max(long_value, short_value)
    if base == 0:
        return 0.0
    return abs(long_value - short_value) / base
