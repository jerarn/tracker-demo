"""Test CSV Manager functionality."""

from pathlib import Path
import tempfile

import pytest

from tracker import CSVManager


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".csv"
    ) as temp_file:
        yield temp_file.name
    # Cleanup
    if Path(temp_file.name).exists():
        Path(temp_file.name).unlink()


@pytest.fixture
def sample_account_csv_data(temp_csv_file):
    """Create sample CSV data for accounts."""
    csv_content = """name,currency_code
Account 1,EUR
Account 2,USD
Test Account,GBP"""

    with Path(temp_csv_file).open("w", encoding="utf-8") as f:
        f.write(csv_content)

    return temp_csv_file


def test_read_csv(sample_account_csv_data):
    """Test reading CSV file."""
    data = CSVManager.read_csv(sample_account_csv_data)
    assert len(data) == 3
    assert data[0]["name"] == "Account 1"
    assert data[1]["currency_code"] == "USD"


def test_write_csv_static_method(temp_csv_file):
    """Test the static write method."""
    # Test data
    test_data = [
        {"name": "Account 1", "currency_code": "EUR", "balance": 1000.50},
        {"name": "Account 2", "currency_code": "USD", "balance": 2500.75},
        {"name": "Account 3", "currency_code": "GBP", "balance": 750.25},
    ]

    # Execute
    CSVManager.write_csv(test_data, temp_csv_file)

    # Verify file creation and content
    assert Path(temp_csv_file).exists()

    with Path(temp_csv_file).open(encoding="utf-8") as f:
        lines = f.readlines()

    # Check header
    assert "name,currency_code,balance" in lines[0]

    # Check data rows
    assert "Account 1,EUR,1000.5" in lines[1]
    assert "Account 2,USD,2500.75" in lines[2]
    assert "Account 3,GBP,750.25" in lines[3]


def test_write_csv_empty_list(temp_csv_file):
    """Test write method with empty list."""
    # Execute
    CSVManager.write_csv([], temp_csv_file)

    # Should not create file when list is empty
    # Implementation depends on your CSVManager.write method


if __name__ == "__main__":
    pytest.main([__file__])
