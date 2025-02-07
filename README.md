# Project Structure

```
project/
├── db/
│     ├── cache/
│     │   ├── market_data/ # market data (historical of 50 stocks)
│     │   ├── fundamental_data/ # fundamental data (e.g. balance sheet, income statement, cash flow)
│     │   └── news_data/ # news data (e.g. press releases, analyst reports, company announcements)
│     ├── figures/
│     │   ├── rsi/
│     │   ├── fibonacci/
│     │   ├── pbv/
│     │   ├── macd/
│     │   ├── rsi/
│     │   ├── fibo_bollingbands/
│     │   ├── macd_ml/
│     │   └── kelter_channel/
│     └── processed/
│          └── preprocessed_data/
├── docs/
│   ├── ml/
│   │   └── rl.md
│   └── strategies/
│       ├── strategy_components.md
│       ├── backtesting.md
│       └── signal_generation.md
├── src/
│   ├── init.py
│   ├── data/
│   │   ├── init.py
│   │   ├── data_loader.py         # functions to fetch and ingest market data (historical, fundamentals, news)
│   │   ├── clean_data.py          # data cleaning and error handling routines
│   │   ├── fundamental_data.py    # load and process fundamentals (could later include news)
│   │   └── news_data.py           # load and process news data
│   ├── indicators/
│   │   ├── init.py
│   │   ├── moving_average.py      # moving average; configurable window lengths
│   │   ├── macd.py                # MACD with MACD, signal, histogram
│   │   ├── rsi.py                 # RSI indicator
│   │   ├── variance.py            # Rolling variance computation
│   │   ├── fibonacci.py           # Fibonacci levels calculation
│   │   ├── pbv.py                 # Price-to-book or Price-to-BV ratio calculations
│   │   ├── macd_ml.py             # MACD with MACD, signal, histogram
│   │   ├── fibo_bollingbands.py   # Fibonacci Bollinger Bands indicator
│   │   └── kelter_channel.py      # Kelter Channel indicator (and other channel based tools)
│   ├── strategies/
│   │   ├── init.py
│   │   ├── base_strategy.py       # Abstract or base class for strategies
│   │   ├── long_term_strategy.py  # Longer-horizon, trend following strategies
│   │   ├── day_trading_strategy.py# Intraday or short-term mean-reversion strategies
│   │   └── arbitrage_strategy.py  # Strategies to capture mispricings or statistical arbitrage
│   ├── models/
│   │   ├── init.py
│   │   ├── ml_model.py            # Supervised ML models for signal generation or regression models
│   │   └── reinforcement_learning.py  # RL models (DQL, actor-critic, etc.) – infrastructure for RL training
│   ├── env/
│   │   ├── init.py
│   │   └── env.py                 # Custom Gym environment for trading
│   ├── risk/
│   │   ├── init.py
│   │   └── risk_management.py     # Risk management strategies (e.g., position sizing, stop-loss, take-profit)
│   ├── news/
│   │   ├── init.py
│   │   ├── parsing.py     # Process and analyze news data
│   │   ├── feature_extraction.py     # Sentiment analysis of news
│   │   └── sentiment_analysis.py     # Sentiment analysis of news
│   ├── execution/
│   │   ├── init.py
│   │   └── executor.py            # Integration with APIs and/or WebSockets for live execution
│   ├── backtesting/
│   │   ├── init.py
│   │   └── backtester.py          # Vectorized and event-driven backtesting framework
│   └── utils/
│       ├── init.py
│       ├── config.py            # central configuration (e.g., file paths, parameters)
│       ├── logger.py            # logging configuration and helper functions
│       └── helper.py            # various utility functions (e.g., date/time handling, performance metrics)
├── tests/
│   ├── test_data_loader.py
│   ├── test_indicators.py
│   └── test_strategies.py
├── notebooks/
│   ├── exploratory_data_analysis.ipynb
│   ├── strategy_backtest.ipynb
│   └── rl_training.ipynb
├── requirements.txt             # dependencies (e.g., numpy, pandas, matplotlib, tensorflow, scikit-learn, etc.)
├── setup.py                     # for installation as a package if desired
└── README.md                    # project overview, usage instructions, and design rationale
```

Roadmap
data -> indicators -> strategies -> models (ML & RL) -> backtesting -> execution





