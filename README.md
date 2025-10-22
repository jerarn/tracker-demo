# Portfolio Tracker - Quantitative Analytics Engine

> **A portfolio management and risk analytics system**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![SQLAlchemy](https://img.shields.io/badge/ORM-SQLAlchemy%202.0-green.svg)](https://www.sqlalchemy.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](https://pytest.org/)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff-purple.svg)](https://github.com/charliermarsh/ruff)

## 🎯 Overview

A comprehensive portfolio analytics engine with interactive dashboard for tracking investments, analyzing performance, and managing risk.

### Key Features

- 📊 **Dashboard** - Portfolio analytics visualization with Streamlit
- 💹 **Performance Tracking** - Returns, drawdowns, and benchmark comparisons
- ⚠️ **Risk Analytics** - VaR, Sharpe ratios, and comprehensive risk metrics
- 🔄 **Multiple Accounting Methods** - FIFO, LIFO, and Average Cost
- 🖼️ **Visualizations** - Performance charts and benchmark comparisons
- 🗃️ **Data Management** - SQLAlchemy ORM with sample data generation

### Core Components
- **Data Ingestion**: API fetchers and CSV loaders
- **Analytics Engine**: Portfolio performance and risk calculations
- **Data Layer**: SQLAlchemy ORM with normalized schema
- **Dashboard**: Interactive Streamlit web interface

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and install
git clone https://github.com/jarn/tracker-demo.git
cd tracker-demo

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -e .
```

### 2. Generate Sample Data
```bash
# Create sample portfolio data
python scripts/generate_sample_data.py

# Initialize database and load data
python scripts/migrate.py init --direct
python scripts/loader.py all
```

## 📊 Analytics Data Showcase

### Run Analytics Script
```bash
python scripts/show_analytics.py
```

### Portfolio Performance Example
```
============================================================
📊 PORTFOLIO TRACKER ANALYTICS RESULTS
============================================================

📋 Data Overview
------------------------------
   Transactions:     6 records
   Market Data:      365 price points
   Cash Flows:       18 entries
   Market Data Columns: ['AAPL', 'BTC-USD', 'QQQ', 'SPY', 'CASH']
   Period:           2024-01-01 to 2024-12-30
   Duration:         364 days

💰 Portfolio Performance
------------------------------
   Current Value:    $34,657.24
   Total Return:     37.92%
   Annualized:       37.95%
   Volatility:       16.13%
   Sharpe Ratio:     2.352

⚠️ Risk Metrics
------------------------------
   Max Drawdown:     -10.98%
   Drawdown Days:    90 days
   VaR (95%):        $454.75
   CVaR (95%):       $703.28
   Downside Dev:     10.62%
   Sortino Ratio:    2.963
   Calmar Ratio:     3.457

🥧 Current Allocation
------------------------------
   SPY         :    36.33%
   AAPL        :    28.36%
   QQQ         :    18.57%
   BTC-USD     :     16.5%

📈 Benchmark Analysis
------------------------------
   Portfolio Return: 37.95%
   Benchmark (ACWI): 20.28%
   Alpha:            11.73%
   Beta:             1.139
   Outperformance:   17.67%
```

### Launch Dashboard
```bash
# Start interactive dashboard
streamlit run dashboard.py --server.port 8506
```

Open http://localhost:8506 to view your portfolio analytics dashboard!

### Dashboard Features
- **Performance Charts**: Portfolio value over time with markers
- **Risk Analysis**: Comprehensive risk metrics
- **Allocation View**: Current portfolio allocation pie charts
- **Position Details**: Holdings with P&L breakdown
- **Benchmark Comparison**: Performance vs market indices

## 🛠️ Project Structure

```
tracker/
├── src/tracker/           # Main package
│   ├── analytics/         # Performance and risk analysis
│   │   ├── accounting/    # Accounting methods for cost analysis
│   │   └── core/          # Core analytics modules (Performance, Risk)
│   ├── api/               # REST API
│   ├── config/            # Configuration management
│   ├── data/              # Data management and models
│   │   ├── access/        # External data sources
│   │   ├── managers/      # Data managers (CSV, DB)
│   │   ├── orm/           # SQLAlchemy ORM models
│   │   └── repos/         # Models repositories
│   └── services/          # Business logic and services
├── scripts/               # Entry point scripts
├── sql/                   # Database schema and queries
├── alembic/               # Database model versioning
├── data/                  # Data files
├── notebooks/             # Dashboards examples
├── examples/              # Usage & Dev examples
├── docs/                  # Documentation
├── logs/                  # Application logs
├── dl/                    # Data loading files
└── tests/                 # Unit and integration tests
```

## License

MIT License - see LICENSE file for details.
