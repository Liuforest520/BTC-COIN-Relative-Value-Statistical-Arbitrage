# Relative Value Statistical Arbitrage

面向分钟级多资产、多交易对的相对价值统计套利研究与回测框架。项目最初用于 BTC 与 COIN 的相对价值研究，目前已经扩展为统一的 Multi-Pair Pipeline，可同时处理加密资产以及以 USDT 计价的 TradFi 映射合约。

> 本项目仅用于研究、回测和模型评估，不包含真实账户下单或生产级交易执行能力，也不构成投资建议。

## 核心能力

- 按分钟流式读取多个标的的 K 线，避免一次性载入全部行情；
- 对每个 Pair 独立估计 Alpha、Beta、残差及残差 ADF 指标；
- 支持 Price、Log Price、Log Return 三类回归输入；
- 支持 11 类滚动或在线 Beta 估计器；
- 通过残差 Z-score、两阶段入场和均值回归过滤生成信号；
- 支持固定名义、Beta-neutral 和 Volatility-neutral 仓位计算；
- 在组合层统一完成 Pair 选择、资金分配和敞口控制；
- 模拟手续费、滑点、资金费率、订单撮合、拒单和未平仓估值；
- 支持单次详细回测、并行参数 Sweep 和快速汇总回测；
- 导出 CSV、JSON、组合总览页和单 Pair HTML 交易复盘页面。

## 运行流程

```text
分钟 K 线与资金费率
        ↓
Pair 数据对齐
        ↓
Estimator：Alpha / Beta / Residual / ADF
        ↓
Signal：Z-score 与均值回归判断
        ↓
Sizing：两条腿的目标仓位
        ↓
Portfolio：多 Pair 资金分配
        ↓
Risk / Execution：风控与订单规划
        ↓
Exchange Simulator：撮合、成本与权益
        ↓
Metrics / Reports：指标和复盘报告
```

## 环境安装

项目要求 Python 3.10 及以上版本。以下命令以 Windows PowerShell 为例：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

如需完整研究依赖，可使用：

```powershell
pip install -e ".[research]"
```

运行测试还需要安装 `pytest`：

```powershell
pip install pytest
```

## 数据准备

### 本地文件结构

默认配置按以下结构读取分钟 K 线和资金费率：

```text
data/
  BTCUSDT/
    BTCUSDT-1m.csv
    BTCUSDT-funding_rate_daily.csv
  ETHUSDT/
    ETHUSDT-1m.csv
```

每个标的的实际文件路径和交易所名称均在 `config/config.yaml` 的 `data.symbols` 中配置。K 线需要包含可识别的时间戳、OHLC 和成交量字段；加载器会统一字段名称、时间顺序和数据类型。

资金费率文件不是按文件名推断结算周期，而是使用文件中的真实时间戳，因此可以处理 1 小时、4 小时或 8 小时等不同结算间隔。只有在 `cost.funding_enabled: true` 时，资金费率才会进入回测。

### 从 MySQL 下载

项目提供 MySQL 行情下载器：

```powershell
python scripts/download_data.py --list-datasets
python scripts/download_data.py --dataset binance_spot_kline --symbols BTCUSDT,ETHUSDT --timeframe 1m --start 20250101 --end 20251231
```

下载器默认读取 `config/mysql.config`，也支持 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_USER`、`MYSQL_PASSWORD`、`MYSQL_DATABASE` 等环境变量。`config/mysql.config` 包含本地连接信息，已被 `.gitignore` 排除，禁止提交到仓库。

完整参数：

```powershell
python scripts/download_data.py --help
```

## 配置说明

日常单次回测使用 `config/config.yaml`。主要结构如下：

```yaml
data:
  symbols: {}

backtest:
  initial_cash: 100000
  start_time: "2025-01-01"

active_setup: multi_pair_beta
setups:
  multi_pair_beta:
    strategy: multi_pair
    pairs: []
    pipeline:
      estimator: {}
      signal: {}
      sizing: {}
      portfolio: {}
      execution: {}

cost:
  fee_rate: 0.0005
  slippage_bps: 1
  funding_enabled: false

risk:
  enabled: false
  mode: pass_through
```

当前默认配置包含 15 个 Pair，其中 10 个加密资产 Pair、5 个 TradFi 相关 Pair。默认核心参数为：

| 配置项 | 当前值 |
| --- | --- |
| Estimator | `rolling_ols` |
| Regression | `log_price` |
| Model lookback | `28800` 分钟，即 20 天 |
| Model update | `1440` 分钟，即 1 天 |
| Signal | `zscore_reversion` |
| Entry / Exit Z-score | `1.75 / 0.5` |
| Sizing | `beta_neutral` |
| Portfolio | `equal_weight` + `equal_cap` |
| Fee rate | `0.0005` |
| Slippage | `1 bps` |
| Funding | 默认关闭 |
| Risk | 默认 `pass_through` |

修改配置时，`pairs` 中使用的标的必须已经存在于 `data.symbols`。回测只加载当前启用 Pair 所需的数据，不会无条件读取配置中的全部标的。

## 模型组件

### Estimator

- `rolling_ols`
- `periodic_ols`
- `tls`
- `dols`
- `ewls`
- `huber`
- `age_weighted_wls`
- `residual_weighted_wls`
- `winsorized_ols`
- `rls`
- `kalman`

### Signal

- `zscore`：固定阈值、分位数或自适应阈值；
- `zscore_reversion`：支持两阶段触发、均值回归过滤和多种持仓退出 Z-score 算法。

### Sizing 与 Portfolio

- Sizing：`fixed_notional`、`beta_neutral`、`volatility_neutral`；
- Portfolio：`equal_weight`、`risk_parity`、`min_variance`、`constrained_qp`、`max_sharpe`。

## 单次回测

在项目根目录执行：

```powershell
python scripts/run_backtest.py config/config.yaml
```

如配置中设置了固定运行名称：

```yaml
report:
  run_name: tls_price_L60D_U1D_E2p0_X0p5
```

结果会写入：

```text
results/backtests/<run_name>/
```

主要产物包括：

- `metrics.json`、`metrics.csv`、`key_metrics.csv`：回测指标；
- `equity_curve.csv`：组合权益曲线；
- `orders.csv`、`trades.csv`：订单与成交；
- `position_curve.csv`：分钟级组合持仓；
- `signal_curve.csv`：分钟级信号和模型状态；
- `pair_curve.csv`：各 Pair 的分钟级状态；
- `funding_payments.csv`：资金费率结算记录；
- `risk_history.csv`：下单和成交后的风控检查记录；
- `final_position_valuation.json`：结束时未平仓仓位估值；
- `pair_metrics.csv`、`return_attribution.csv`、`risk_attribution.csv`：Pair 指标与归因；
- `summary.md`：本次回测的 Markdown 摘要；
- `portfolio_summary.html`：组合资金曲线总览；
- `trade_review.html`：组合与单 Pair 交易复盘入口。

## 参数 Sweep

生成参数组合：

```powershell
python scripts/generate_configs.py <sweep配置.yaml>
```

运行 Sweep：

```powershell
python scripts/run_config_sweep.py <sweep配置.yaml> --workers 8
```

Sweep 支持多进程并行、快速汇总模式、结果排序，以及对排名靠前的配置补跑完整报告。已完成的 Sweep 定义保存在 `config/sweeps/archive/`，自动展开的配置保存在其 `generated/` 子目录，但不会纳入版本控制。

`config/tls_huber_selected_24/` 保存 24 个手工筛选的 TLS/Huber 单次回测配置，这些配置应逐个传给 `scripts/run_backtest.py`，不通过 Sweep 展开。

## 查看 HTML 报告

启动本地报告中心：

```powershell
python scripts/serve_trade_review.py --results-root results/backtests --port 8767
```

浏览器访问：

```text
http://127.0.0.1:8767/
```

服务仅监听本机地址 `127.0.0.1`。也可以通过 `--run-dir` 指定启动后默认打开的回测目录，通过 `--no-browser` 禁止自动打开浏览器。

## 项目结构

```text
core/
  backtest/                 详细回测与 Sweep 快速回测
  modules/
    config/                 YAML 配置加载与校验
    data/                   K 线、资金费率、滚动窗口与 MySQL 下载
    estimator/              Alpha/Beta/残差估计器
    exchange/               模拟交易所、撮合、现金和持仓
    execution/              Pair 订单规划
    logger/                 日志配置
    metrics/                收益、成本和市场中性指标
    models/                 订单、成交、持仓和 Pipeline 数据结构
    portfolio/              多 Pair 组合分配
    reporting/              CSV、JSON 与 HTML 报告
    risk/                   下单前和成交后的风控
    signals/                Z-score 信号与退出逻辑
    sizing/                 Pair 级仓位计算
    strategy/               Multi-Pair 策略与 Pair Pipeline

scripts/                    日常运行入口
test/                       自动化测试
maintenance/                历史迁移和数据检查工具
research/scripts/           阶段性模型研究脚本
config/archive/             历史主配置
config/sweeps/archive/      已完成的 Sweep 定义
config/tls_huber_selected_24/ 手工精选的单次回测配置
data/                       本地行情数据，不上传
results/                    回测和 Sweep 结果，不上传
documents/                  本地研究资料，不上传
```

## 测试

运行完整测试：

```powershell
python -m pytest -q
```

只检查 Python 文件能否正确导入和编译：

```powershell
python -m compileall -q core scripts test
```

## 版本控制范围

仓库采用安全白名单策略。以下内容会上传：

- `core/`、`scripts/`、`test/` 中的代码；
- 主配置、历史手工配置、Sweep 定义和 24 个精选配置；
- `maintenance/` 与 `research/scripts/` 中的源代码和说明；
- README、依赖声明和项目元数据。

以下内容不会上传：

- `data/`、`results/`、`logs/`、`outputs/`、`tmp/`；
- `documents/` 中的本地研究资料和表格；
- `config/mysql.config` 等连接凭据；
- 自动生成的 Sweep 配置、研究结果和 HTML 报告；
- Python 虚拟环境、缓存、IDE 配置和 `node_modules/`。

提交前建议执行：

```powershell
git status --short
git diff --check
python -m pytest -q
```

## 风险提示

回测结果高度依赖数据质量、成交假设、手续费、滑点、资金费率、样本区间和参数选择。历史表现不代表未来收益；将研究策略用于真实资金前，需要额外完成前向测试、容量评估、异常行情测试、实盘风控和交易所接口验证。
