"""FIFO accounting strategy implementation."""

from .strategy import AccountingStrategy, Lot


class FIFO(AccountingStrategy):
    """FIFO (First In, First Out) accounting strategy."""

    def process_transaction(
        self, state: dict, tr_type: str, qty: float, price: float
    ) -> tuple[dict, float]:
        """Process a transaction and update the state accordingly."""
        lots = state.get("lots", [])
        realized = 0.0

        if tr_type in ["BUY", "TRANSFER_IN", "INTEREST", "DIVIDEND"]:
            if qty > 0:
                lots.append(Lot(qty, price))
        elif tr_type == "SELL":
            qty_to_sell = qty
            proceeds = 0.0
            cost_removed = 0.0
            while qty_to_sell > 0 and lots:
                lot = lots[0]
                take = min(lot.qty, qty_to_sell)
                proceeds += take * price
                cost_removed += take * lot.price
                lot.qty -= take
                qty_to_sell -= take
                if lot.qty == 0:
                    lots.pop(0)
            realized = proceeds - cost_removed
        elif tr_type in {"FEE", "TAX", "TRANSFER_OUT"}:
            qty_to_remove = qty
            removed_cost = 0.0
            while qty_to_remove > 0 and lots:
                lot = lots[0]
                take = min(lot.qty, qty_to_remove)
                removed_cost += take * lot.price
                lot.qty -= take
                qty_to_remove -= take
                if lot.qty == 0:
                    lots.pop(0)
            realized = -removed_cost

        return {"lots": lots}, realized

    def get_current_metrics(self, state: dict) -> dict:
        """Get current metrics from the state."""
        lots = state.get("lots", [])
        current_qty = sum(lot.qty for lot in lots)
        current_cost = sum(lot.qty * lot.price for lot in lots)
        avg_cost = (current_cost / current_qty) if current_qty != 0 else 0.0
        return {"qty": current_qty, "cost_basis": current_cost, "avg_cost": avg_cost}
