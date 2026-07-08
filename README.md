# BTC-COIN Relative Value Statistical Arbitrage

这个项目是一个 BTC Perp 和 COIN Perp 相对价值统计套利的研究框架。当前主策略是滚动 `log price` 回归残差 z-score，加 `beta_neutral` 仓位控制，并在回测里计入手续费、滑点和资金费率。

## 1. 启动项目

建议使用 Python 3.10 以上版本。

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

如果不想安装成 editable package，也可以只执行：

```bash
pip install -r requirements.txt
```

然后在项目根目录运行脚本。

## 2. 放置数据

原始数据需要手动放到 `data/` 目录。当前 `config/config.yaml` 默认读取下面四个文件：

```text
data/
  BTCUSDT_kline_1m_20260201_20260706_20260706_150041.csv
  COINUSDT_kline_1m_20260209_20260706_20260706_150018.csv
  binance_funding_rate_BTCUSDT_20260706_133637.csv
  binance_funding_rate_COINUSDT_20260706_134744.csv
```

K 线文件需要包含时间戳和 OHLCV 字段。资金费率文件用于按结算时间扣除 funding cost。

## 3. 跑单次回测

```bash
python scripts/run_backtest.py config/config.yaml
```

回测结果会输出到：

```text
results/backtests/
```

里面会包含 metrics、orders、trades、position curve、signal curve 和图表。`results/` 是运行产物。

## 4. 项目结构

```text
core/
  backtest/              回测主流程
  modules/
    config/              配置读取
    data/                K 线和资金费率数据导入
    exchange/            模拟交易所、订单撮合、账户权益
    logger/              loguru 日志配置
    metrics/             绩效、交易、成本和市场中性指标
    models/              Order、Trade、Funding 等数据结构
    position_sizing/     fixed_notional、beta_neutral、volatility_neutral 等仓位分配
    reporting/           回测报告和图表导出
    risk/                下单前和成交后的风控检查
    signals/             z-score、协整残差、ratio、momentum、funding filter 等信号
    strategy/            策略父类和 pair trading 策略封装

scripts/
  run_backtest.py        跑单次回测
  generate_configs.py    根据 sweep 配置生成多组 config
  run_config_sweep.py    批量运行多组 config

config/
  config.yaml            默认回测配置
```

## 5. 当前主配置

当前默认配置是：

```text
active_setup: cointegration_beta
pair: BTCUSDT / COINUSDT
signal: cointegration_zscore
regression_method: log_price
position_sizing: beta_neutral
fee_rate: 0.0005
slippage_bps: 1
funding_enabled: true
```

也就是说，策略用 `log(COIN)` 对 `log(BTC)` 做滚动回归，用残差 z-score 判断相对偏离，再按 beta-neutral 的方式配置两条腿。

## 6. 参数搜索

如果需要批量跑参数，可以使用：

```bash
python scripts/generate_configs.py
python scripts/run_config_sweep.py
```

注意：`config/sweep.yaml`、`config/generated/` 和 `results/sweeps/` 都属于参数搜索和运行产物。单次回测只需要 `config/config.yaml` 这一份主配置。
