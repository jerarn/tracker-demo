"""Cash accounting strategy implementation."""

from .strategy import AccountingStrategy


class Cash(AccountingStrategy):
    """Cash accounting strategy (simplified for cash flows)."""

    def process_transaction(
        self,
        state: dict,
        tr_type: str,
        qty: float,
        price: float,  # noqa: ARG002
    ) -> tuple[dict, float]:
        """Process a transaction and update the state accordingly."""
        q = state.get("qty", 0.0)

        if tr_type in {"INTEREST", "DIVIDEND", "TRANSFER_IN", "DEPOSIT"}:
            q += qty
        elif tr_type in {"FEE", "TAX", "TRANSFER_OUT", "WITHDRAWAL"}:
            q -= qty

        return {"qty": q, "cost": q}, 0.0  # For cash, cost_basis = balance

    def get_current_metrics(self, state: dict) -> dict:
        """Get current metrics from the state."""
        q = state.get("qty", 0.0)
        return {"qty": q, "cost_basis": q, "avg_cost": 1.0}
