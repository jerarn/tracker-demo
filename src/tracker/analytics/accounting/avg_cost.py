"""Average Cost Accounting Strategy Implementation."""

from .strategy import AccountingStrategy


class AverageCost(AccountingStrategy):
    """Average cost accounting strategy."""

    def process_transaction(
        self, state: dict, tr_type: str, qty: float, price: float
    ) -> tuple[dict, float]:
        """Process a transaction and update the state accordingly."""
        q = state.get("qty", 0.0)
        c = state.get("cost", 0.0)
        realized = 0.0

        if tr_type in ["BUY", "TRANSFER_IN", "INTEREST", "DIVIDEND"]:
            c += price * qty
            q += qty
        elif tr_type == "SELL":
            removed = min(qty, q) if q > 0 else 0.0
            if removed > 0:
                avg_before = (c / q) if q != 0 else 0.0
                cost_removed = avg_before * removed
                proceeds_removed = removed * price
                realized = proceeds_removed - cost_removed
                c -= cost_removed
                q -= removed
            # Handle shorts
            remaining = qty - removed
            if remaining > 0:
                c -= price * remaining
                q -= remaining
        elif tr_type in {"FEE", "TAX", "TRANSFER_OUT"}:
            removed = min(qty, q) if q > 0 else 0.0
            if removed > 0:
                avg_before = (c / q) if q != 0 else 0.0
                cost_removed = avg_before * removed
                c -= cost_removed
                q -= removed
                realized = -cost_removed

        return {"qty": q, "cost": c}, realized

    def get_current_metrics(self, state: dict) -> dict:
        """Get current metrics from the state."""
        q = state.get("qty", 0.0)
        c = state.get("cost", 0.0)
        avg_cost = (c / q) if q != 0 else 0.0
        return {"qty": q, "cost_basis": c, "avg_cost": avg_cost}
