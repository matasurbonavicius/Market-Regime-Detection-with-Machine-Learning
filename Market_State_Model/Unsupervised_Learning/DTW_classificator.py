import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def classify_states(prices: np.array, states: np.array, n_states: int, to_csv: bool = False, manual_confirmation: bool = True) -> dict:
    """
    Classify states based on DTW distances to standard patterns
    and plot their cumulative returns.

    Parameters:
    - prices: A numpy array of prices.
    - states: A numpy array of hidden states.
    - n_states: Number of unique states.

    Returns:
    - A dictionary with state classifications.
    """
    # Ensure prices is 1D
    prices = np.asarray(prices).flatten()

    def save_classification(state_classifications: dict) -> None:

        """
        Save the classified states in a dataframe format so that
        predictions could access it without needing to do the whole
        calculation over again

        Parameters:
        - state_classifications: a dictionary containing states map

        Returns:
        - None. File is saved in the specified directory.
        """

        df = pd.DataFrame([state_classifications])
        df.to_csv('Models/state_classification.csv', index=False)
    
    def compute_dtw(sequence, pattern):
        """Compute DTW distance between two sequences."""
        n, m = len(sequence), len(pattern)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[0, 1:] = float('inf')
        dtw_matrix[1:, 0] = float('inf')

        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(sequence[i-1] - pattern[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

        return dtw_matrix[n, m]
    
    def on_exit():
        root.quit() 
        root.destroy()
    
    # Calculate returns and cumulative returns for each state
    # Prepend a zero to keep same length as prices
    returns = np.concatenate([[0], np.diff(prices)])
    cumulative_returns = np.zeros_like(prices)
    for state in range(n_states):
        mask = states == state
        state_returns = np.cumsum(returns[mask])
        try:
            cumulative_returns[mask] = state_returns - state_returns[0]
        except:
            cumulative_returns[mask] = state_returns
    
    patterns = {
        #'yellow': np.zeros(len(prices)),
        'red': np.linspace(0, -1, len(prices)),
        'green': np.linspace(0, 1, len(prices))
    }

    state_classifications = {}
    
    if manual_confirmation:
        nrows = (n_states + 1) // 2
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 4 * nrows))
        axes = axes.flatten()

    for state in range(n_states):
        state_data = cumulative_returns[states == state]
        
        # Compute DTW distances for the state to each pattern
        distances = {name: compute_dtw(state_data, pattern) for name, pattern in patterns.items()}
        
        # Classify state based on the pattern with minimum DTW distance
        classification = min(distances, key=distances.get)
        state_classifications[state] = classification

        if manual_confirmation:
            
            # Plotting the cumulative returns for the state
            ax = axes[state]
            ax.plot(state_data, label=f'State {state} (Classified as {classification})')
            ax.legend()
            ax.set_title(f'State {state}')
            ax.set_ylabel('Cumulative Returns')
    
    if not manual_confirmation:
        return state_classifications
    
    else:
        root = tk.Tk()
        root.title("Manual Adjustment of Classifications")

        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = tk.Frame(root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        adjusted_classifications = state_classifications.copy()

        def make_update_func(var, state):
            return lambda event: adjusted_classifications.update({state: var.get()})

        for state in range(n_states):
            
            classification = state_classifications[state]
            
            lbl = tk.Label(left_frame, text=f"State {state} (DTW classified as {classification}):")
            lbl.grid(row=state, column=0, padx=10, sticky=tk.W)
            
            var = tk.StringVar(value=classification)
            dropdown = ttk.Combobox(left_frame, textvariable=var, values=["yellow", "red", "green"])

            dropdown.bind("<<ComboboxSelected>>", make_update_func(var, state))
            dropdown.grid(row=state, column=1, padx=10)

        btn_exit = tk.Button(left_frame, text="Save & Exit", command=on_exit)
        btn_exit.grid(row=n_states, column=0, columnspan=2, pady=20)
            
        # Plot cumulative returns
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=right_frame)
        canvas.draw()
        plt.close(fig)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        root.mainloop()

        if to_csv:
            save_classification(adjusted_classifications)

        return adjusted_classifications

# Sample usage:
# np.random.seed(0)
# sample_prices = np.random.rand(100)
# sample_states = np.random.randint(0, 8, size=100)

# classify_states(sample_prices, sample_states, 8)