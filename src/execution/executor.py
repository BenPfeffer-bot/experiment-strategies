import logging


class SimulatedBroker:
    def __init__(self):
        self.orders = []
        self.position = 0
        self.pnl = 0.0
        self.trade_log = []
        # Track cumulative cost and quantity for average cost calculations
        self.total_cost = 0.0
        self.total_qty = 0

    def send_order(self, ticker, side, quantity, price, time_stamp):
        """
        Simulate sending an order; update the current position, cost basis, and PnL.
        """
        order = {
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "time": time_stamp,
        }
        self.orders.append(order)
        logging.info(f"Order executed: {order}")
        self.__update_position(order)
        return order

    def __update_position(self, order):
        side = order["side"]
        qty = order["quantity"]
        price = order["price"]
        # Update position and average cost tracking
        if side == "buy":
            self.position += qty
            self.total_cost += price * qty
            self.total_qty += qty
            self.trade_log.append(
                {"side": "buy", "quantity": qty, "price": price, "pnl": None}
            )
        elif side == "sell":
            self.position -= qty
            # For a sell order not triggered by closing, one may implement partial updates.
            self.trade_log.append(
                {"side": "sell", "quantity": qty, "price": price, "pnl": None}
            )

    def average_cost(self):
        """
        Returns the weighted average cost of the current open position.
        """
        if self.total_qty > 0:
            return self.total_cost / self.total_qty
        else:
            return 0.0

    def close_position(self, ticker, exit_price, time_stamp):
        """
        Close the open position at exit_price and update PnL.
        """
        if self.position == 0:
            return
        qty = abs(self.position)
        side = "sell" if self.position > 0 else "buy"
        order = {
            "ticker": ticker,
            "side": side,
            "quantity": qty,
            "price": exit_price,
            "time": time_stamp,
        }
        self.orders.append(order)
        avg_cost = self.average_cost()
        if side == "sell":
            pnl = (exit_price - avg_cost) * qty
        else:
            pnl = (avg_cost - exit_price) * qty
        self.pnl += pnl
        logging.info(f"Closing order executed: {order}, pnl: {pnl}")
        self.trade_log.append(
            {"side": side, "quantity": qty, "price": exit_price, "pnl": pnl}
        )
        # Reset open position and cost tracking after closing
        self.position = 0
        self.total_cost = 0.0
        self.total_qty = 0


# For real-world simulation, you could expand this broker simulation to include fee structures,
# slippage, partial fills, and more advanced risk management.

if __name__ == "__main__":
    # Sample usage
    import datetime

    broker = SimulatedBroker()
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    broker.send_order("AAPL", "buy", 10, 150.0, now)
    broker.send_order("AAPL", "buy", 5, 152.0, now)
    broker.close_position("AAPL", 155.0, now)
    print("Trade log:", broker.trade_log)
    print("Final PnL:", broker.pnl)
