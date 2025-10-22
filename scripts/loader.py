"""Script to load data from CSV files into the database."""

import argparse
from pathlib import Path

from tracker import (
    Account,
    CashFlow,
    ImportService,
    Instrument,
    MarketData,
    Portfolio,
    Transaction,
    TransactionMethod,
    get_logger,
    setup_logging,
)

setup_logging()
logger = get_logger(__name__)


DATA_MAP = {
    "account": {
        "csv": "data/accounts.csv",
        "model": Account,
    },
    "portfolio": {
        "csv": "data/portfolios.csv",
        "model": Portfolio,
    },
    "instrument": {
        "csv": "data/instruments.csv",
        "model": Instrument,
    },
    "cash_flow": {
        "csv": "data/cash_flows.csv",
        "model": CashFlow,
    },
    "transaction": {
        "csv": "data/transactions.csv",
        "model": Transaction,
        "method": TransactionMethod.BASIC,
    },
    "trade": {
        "csv": "data/trades.csv",
        "model": Transaction,
        "method": TransactionMethod.TRADE,
    },
    "transfer": {
        "csv": "data/transfers.csv",
        "model": Transaction,
        "method": TransactionMethod.TRANSFER,
    },
    "market_data": {
        "csv": "data/market_data_daily.csv",
        "model": MarketData,
    },
}


def main():
    """Main function to load data from CSV files into the database."""
    data_type_list = ", ".join(DATA_MAP.keys())

    parser = argparse.ArgumentParser(
        description="Load data from CSV into the database."
    )
    parser.add_argument(
        "data_types",
        help=(
            f"Comma-separated list of data types to load, or 'all' to load all. "
            f"Possible values: {data_type_list}"
        ),
    )
    parser.add_argument("--csv", help="Path to CSV file (only for single type)")
    args = parser.parse_args()

    if args.data_types.lower() == "all":
        types_to_load = list(DATA_MAP.keys())
    else:
        types_to_load = [
            t.strip() for t in args.data_types.split(",") if t.strip() in DATA_MAP
        ]
        if not types_to_load:
            logger.error("No valid data types specified.")
            return

    service = ImportService()

    for data_type in types_to_load:
        logger.info("Loading data for type: %s", data_type)
        csv_path = (
            args.csv
            if args.csv and len(types_to_load) == 1
            else DATA_MAP[data_type]["csv"]
        )
        model_cls = DATA_MAP[data_type]["model"]
        method = DATA_MAP[data_type].get("method")

        if not Path(csv_path).exists():
            logger.error("CSV file not found: %s", csv_path)
            continue

        service.load_from_csv(model_cls, csv_path, method=method)
        logger.info("Finished loading %s data.", data_type)


if __name__ == "__main__":
    main()
