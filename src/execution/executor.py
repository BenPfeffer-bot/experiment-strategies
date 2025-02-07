# File: src/execution/executor.py
import logging

class SimulatedBroker:
    def __init__(self):
        self.orders = []
        self.position = 0
        self.pnl = 0.0
        self.trade_log = []
    
    def send_order(self, ticker, side, quantity, price, time_stamp):
        """
        Simulate sending an order; update the current position and PnL.
        """
        order = {
            'ticker': ticker,
            'side': side,
            'quantity': quantity,
            'price': price,
            'time': time_stamp
        }
        self.orders.append(order)
        logging.info(f"Order executed: {order}")
        self.__update_position(order)
        return order
    
    def __update_position(self, order):
        side = order['side']
        qty = order['quantity']
        price = order['price']
        # Update internal position and record trade (for simplicity, price impact is immediate)
        if side == 'buy':
            self.position += qty
            self.trade_log.append({
                "side": "buy",
                "quantity": qty,
                "price": price,
                "pnl": None
            })
        elif side == 'sell':
            self.position -= qty
            self.trade_log.append({
                "side": "sell",
                "quantity": qty,
                "price": price,
                "pnl": None
            })

    def close_position(self, ticker, exit_price, time_stamp):
        """
        Close any open position at exit_price and update pnl.
        This function assumes that position is long (if negative then similar logic applies).
        """
        if self.position == 0:
            return
        side = 'sell' if self.position > 0 else 'buy'
        qty = abs(self.position)
        order = {
            'ticker': ticker,
            'side': side,
            'quantity': qty,
            'price': exit_price,
            'time': time_stamp
        }
        self.orders.append(order)
        # Compute PnL for closing trade:
        if side == 'sell':
            pnl = (exit_price - self.average_cost()) * qty
        else:
            pnl = (self.average_cost() - exit_price) * qty
        self.pnl += pnl
        logging.info(f"Closing order executed: {order}, pnl: {pnl}")
        self.trade_log.append({
            "side": side,
            "quantity": qty,
            "price": exit_price,
            "pnl": pnl
        })
        self.position = 0  # reset position after closing

    def average_cost(self):
        """
        Placeholder for average cost calculation; In real trading, you need to track entry costs.
        For demo, assume a fixed cost if needed.
        """
        # For a more robust implementation, track each entry's cost.
        return 0.0  # Replace with actual computation if needed.
