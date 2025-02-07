class SimulatedBroker:
    def __init__(self, initial_balance: float = 10000):
        self.balance = initial_balance
        self.positions = {}

    def open_position(
        self, ticker: str, size: float, entry_price: float, stop_loss: float, date
    ) -> dict:
        """
        Open a new position.

        Args:
            ticker (str): The ticker symbol
            size (float): Position size
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            date: Entry date

        Returns:
            dict: Position details
        """
        cost = size * entry_price
        if cost > self.balance:
            print(f"Insufficient funds to open position in {ticker}")
            return None

        self.balance -= cost

        position = {
            "ticker": ticker,
            "size": size,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "entry_date": date,
            "cost": cost,
            "side": "buy",
            "balance": self.balance,
        }

        self.positions[ticker] = position
        return position

    def close_position(self, position: dict, exit_price: float, date) -> dict:
        """
        Close an existing position.

        Args:
            position (dict): Position to close
            exit_price (float): Exit price
            date: Exit date

        Returns:
            dict: Trade details
        """
        ticker = position["ticker"]
        if ticker not in self.positions:
            print(f"No open position found for {ticker}")
            return None

        proceeds = position["size"] * exit_price
        pnl = proceeds - position["cost"]
        self.balance += proceeds

        trade = {
            "entry_date": position["entry_date"],
            "entry_price": position["entry_price"],
            "position": 1,  # Assuming long positions only
            "size": position["size"],
            "exit_date": date,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": (pnl / position["cost"]) * 100,
            "balance": self.balance,
            "cost": position["cost"],
            "ticker": ticker,
            "side": "sell",
        }

        del self.positions[ticker]
        return trade
