import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Backtest:
    def __init__(self, dates, prices, predicted_values, states_map):
        self.dates = dates
        self.prices = np.asarray(prices).flatten()  # Ensure 1D
        self.predicted_values = predicted_values
        self.states_map = states_map
        
        # Initial balance and stock count
        self.initial_balance = 10000  # Starting with $10,000
        self.balance = self.initial_balance
        self.stock_count = 0
        
        # Mapping states to holding percentages
        self.holdings_map = {
            'green': 1,   # Hold 100%
            'yellow': 0.5,  # Hold 60%
            'red': -0.5      # Hold 30%
        }
        
        self.balance_history = [self.balance]
        self.strategy_3x_leveraged_history = [self.initial_balance]
        
    def get_action(self, prediction):
        """Get action based on the prediction."""
        try:
            state = self.states_map[prediction]
        except:
            state = prediction
        return self.holdings_map[state]
    
    def calculate_max_drawdown(self, values):
        """Calculate and return the maximum drawdown."""
        # Calculate cumulative returns
        cumulative_returns = np.array(values) / self.initial_balance

        # Calculate running max
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = 1 - cumulative_returns / running_max

        return np.max(drawdowns)

    def calculate_sharpe(self, values):
        """Calculate and return the Sharpe ratio."""
        # Assuming a risk-free rate of 0 for simplicity
        daily_returns = np.diff(values) / values[:-1]
        
        # Calculate expected return and standard deviation of returns
        expected_return = np.mean(daily_returns)
        std_dev = np.std(daily_returns)
        
        # Sharpe ratio
        if std_dev == 0:
            return np.inf  # To handle the case when std_dev is 0
        return expected_return / std_dev

    def calculate_return(self, values):
        """Calculate and return the return."""
        return (values[-1] - values[0]) / values[0]
    
    def run_backtest(self):
        """Run the backtest."""
        self.buy_and_hold_history = [self.initial_balance]  # Start the buy&hold with the same initial balance
        for i in range(1, len(self.dates)):  # Starting from the second day
            # Calculate daily return
            daily_return = (self.prices[i] - self.prices[i-1]) / self.prices[i-1]
            
            # Calculate buy & hold value change for the day
            buy_and_hold_value_change = self.buy_and_hold_history[-1] * daily_return
            
            # Update buy & hold balance
            self.buy_and_hold_history.append(self.buy_and_hold_history[-1] + buy_and_hold_value_change)
            
            # Get holding percentage
            holding_percent = self.get_action(self.predicted_values[i-1])
            
            # Calculate value change for the day based on strategy
            value_change = self.balance * daily_return * holding_percent
            value_change_3x = self.strategy_3x_leveraged_history[-1] * daily_return * holding_percent * 3
            
            # Update balance
            self.balance += value_change
            
            # Store balance history
            self.balance_history.append(self.balance)
            self.strategy_3x_leveraged_history.append(self.strategy_3x_leveraged_history[-1] + value_change_3x)


        print("***------------------------------------------------***")
        print("***               BACKTEST STATISTICS              ***")
        print("*")
        print("Max Drawdown (Buy & Hold): {:.2%}".format(self.calculate_max_drawdown(self.buy_and_hold_history)))
        print("Sharpe Ratio (Buy & Hold): {:.2f}".format(self.calculate_sharpe(self.buy_and_hold_history)))
        print("Return (Buy & Hold): {:.2%}".format(self.calculate_return(self.buy_and_hold_history)))
        print("*")
        print("Max Drawdown (Strategy): {:.2%}".format(self.calculate_max_drawdown(self.balance_history)))
        print("Sharpe Ratio (Strategy): {:.2f}".format(self.calculate_sharpe(self.balance_history)))
        print("Return (Strategy): {:.2%}".format(self.calculate_return(self.balance_history)))
        print("*")
        print("Max Drawdown (Strategy 3X Leveraged): {:.2%}".format(self.calculate_max_drawdown(self.strategy_3x_leveraged_history)))
        print("Sharpe Ratio (Strategy 3X Leveraged): {:.2f}".format(self.calculate_sharpe(self.strategy_3x_leveraged_history)))
        print("Return (Strategy 3X Leveraged): {:.2%}".format(self.calculate_return(self.strategy_3x_leveraged_history)))
        print("*")
        print("***------------------------------------------------***")


    def plot_balance(self):
        """Plot the balance over time."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, self.balance_history, label='Portfolio Value', color='blue')
        plt.plot(self.dates, self.buy_and_hold_history, label='Buy & Hold Value', color='green')
        #plt.plot(self.dates, self.strategy_3x_leveraged_history, label='Strategy 3X Leveraged Value', color='red')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True)
        plt.legend()
        plt.show()

"""
# Sample usage
dates = pd.date_range(start="2022-01-01", end="2022-01-10")
prices = np.array([100, 98, 96, 94, 98, 104, 108, 112, 117, 120])
predicted_values = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 1])
states_map = {0: 'green', 1: 'green', 2: 'red', 3: 'yellow', 4: 'yellow', 5: 'green'}

bt = Backtest(dates, prices, predicted_values, states_map)
bt.run_backtest()
bt.plot_balance()
"""
