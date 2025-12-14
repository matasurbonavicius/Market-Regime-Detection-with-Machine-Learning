from typing import List
import numpy as np


class Oracle_Labeler:
    def __init__(self, states: List[int], prices: List[float]):

        """
        Initialize the StateModifier class.

        :param states: A list of states (0, 1, 2) representing bear, neutral, and bull respectively.
        :param prices: A list of price values corresponding to each state.

        The class is initialized with a copy of the states to ensure original data isn't modified directly.
        """

        self.states = states.copy()
        self.prices = np.asarray(prices).flatten()  # Ensure 1D

    def get_continuous_state_indices(self, current_state: int, start_idx: int) -> List[int]:
        
        """
        Get the indices of a continuous state streak.

        :param current_state: The state (0, 1, 2) to look for.
        :param start_idx: The index from which to start looking.
        :return: A list of indices for the continuous state streak.

        Logic:
        From the start index, this function continues to check for the 
        same state until a different state is encountered.
        It aggregates the indices of the similar state into a list and returns it.
        """

        state_indices = [start_idx]
        for j in range(start_idx + 1, len(self.states)):
            if self.states[j] == current_state:
                state_indices.append(j)
            else:
                break
        return state_indices

    def handle_bull_state(self, state_indices: List[int]) -> None:
        
        """
        Handle logic for a continuous bull state streak.

        :param state_indices: Indices representing a continuous bull state streak.

        Logic:
        For a bull state, it determines the maximum price within the state streak. Once the maximum price is identified,
        it turns all the states after this price to neutral within the streak.
        """

        # Find the index within `state_indices` that corresponds to the maximum price in `self.prices`. 
        max_price_idx = max(state_indices, key=lambda idx: self.prices[idx])

        for idx in range(max_price_idx + 2, state_indices[-1] + 1):
            self.states[idx] = 1


    def handle_bear_state(self, state_indices: List[int]) -> None:
        
        """
        Handle logic for a continuous bear state streak.

        :param state_indices: Indices representing a continuous bear state streak.

        Logic:
        For a bear state, it determines the minimum price within the state streak. Once the minimum price is identified,
        it turns all the states after this price to neutral within the streak.
        """

        # Find the index within `state_indices` that corresponds to the minimum price in `self.prices`. 
        min_price_idx = min(state_indices, key=lambda idx: self.prices[idx])

        for idx in range(min_price_idx + 2, state_indices[-1] + 1):
            self.states[idx] = 1

    def modify_states(self) -> List[int]:
        
        """
        Modify the states list based on price trends.

        :return: Modified list of states.

        Logic:
        Iterates over the entire states list. For each unique state, 
        it identifies the streak of continuous occurrences
        and calls the respective handler (bull or bear). 
        Once a streak is processed, the loop jumps to the next unique state.
        """

        i = 0
        while i < len(self.states):
            current_state = self.states[i]
            state_indices = self.get_continuous_state_indices(current_state, i)
            
            if current_state == 0:  # bull state
                self.handle_bull_state(state_indices)
            elif current_state == 2:  # bear state
                self.handle_bear_state(state_indices)
            
            i = state_indices[-1] + 1
        
        return self.states

# # Example usage:
# states = [2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2]
# prices = [11, 10, 12, 11, 9, 5, 8, 13, 15, 16, 10, 20, 15, 19, 17]

# modifier = Oracle_Labeler(states, prices)
# new_states = modifier.modify_states()
# print(new_states)  # Expected: [0, 2, 2, 1, 0, 0, 1, 2, 2, 2, 1, 1, 1, 1, 0]


