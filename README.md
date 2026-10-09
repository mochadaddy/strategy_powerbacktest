[![Ask DeepWiki](https://img.shields.io/badge/Ask-DeepWiki-2563eb?style=flat-square)](https://deepwiki.com/mochadaddy/strategy_powerbacktest)
# Strategy Power Backtester 📈

A professional-grade algorithmic trading backtesting framework with seamless integration with Futu OpenAPI, designed for quantitative traders and researchers.

<!-- ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)
![Futu API](https://img.shields.io/badge/Futu%20API-8.7-brightgreen) -->

## 🚀 Features


- **Data Integration**
  - Seamless integration with Futu OpenAPI for real-time and historical data
  - Support for multiple timeframes (1m to 1d)
  - Efficient data caching and storage options (SQLite, CSV, in-memory)

- **Strategy Development**
  - Flexible strategy implementation framework
  - Built-in technical indicators and analysis tools
  - Easy-to-extend base strategy class
  - Strategy parameter optimization capabilities

- **Backtesting Engine**
  - High-performance event-driven architecture
  - Realistic simulation with slippage and commission modeling
  - Comprehensive position and portfolio management
  - Detailed performance metrics and analysis

- **Risk Management**
  - Position size control
  - Maximum position limits
  - Customizable risk parameters

## 🛠 Prerequisites

- Python 3.8 or higher
- [Futu OpenD](https://www.futunn.com/download/openAPI) installed and running
- Futu trading account (demo account available)

## ⚡️ Quick Start

1. **Installation**
```bash
git clone https://github.com/billpwchan/strategy-powerbacktest.git
cd strategy-powerbacktest
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Futu OpenD**
- Install and launch Futu OpenD
- Log in to your Futu trading account
- Ensure OpenD is running on localhost:11111 (default)

3. **Run a Sample Backtest**
```bash
python main.py --strategy macd --symbol HK.00700 HK.09988 --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 100000 --commission 0.001
```
## 📊 Example Strategy

```python
from src.strategy.base_strategy import BaseStrategy
class MovingAverageCrossStrategy(BaseStrategy):
def init(self, parameters):
super().init(parameters)
self.short_window = parameters.get('short_window', 20)
self.long_window = parameters.get('long_window', 50)
```


## ⚙️ Configuration

The system is highly configurable through `config.yaml`. Key configuration sections:

- Futu API connection settings
- Backtesting parameters
- Data storage options
- Strategy-specific parameters
- Logging preferences

See `config.yaml` for detailed configuration options.

## 📈 Performance Metrics

The backtester provides comprehensive performance analytics:
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor
- Position Analysis

## 🔧 Development

### Creating a New Strategy

1. Inherit from BaseStrategy
2. Implement generate_signals method
3. Register your strategy
4. Configure parameters in config.yaml

### Project Structure
```bash
strategy-powerbacktest/
├── src/
│   ├── data/          # Data handling
│   ├── strategy/      # Trading strategies
│   ├── engine/        # Backtesting engine
│   ├── templates/     # HTML report templates
│   └── utils/         # Helper functions
├── tests/             # Test suite
├── config.yaml        # Configuration
└── main.py           # Entry point
```

## 📝 License

Apache License 2.0 - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Documentation

For detailed documentation:
- [Futu OpenAPI Documentation](https://openapi.futunn.com/)
- [API Reference](./docs/api.md)
- [Strategy Development Guide](./docs/strategies.md)

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
