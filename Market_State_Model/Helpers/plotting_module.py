import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List


class Plotting_Module:
    
    """
    Advanced Plotter for visualizing data.

    Allows for plotting multiple value lists against a common date axis.
    """

    def __init__(self, 
                 dates: Union[np.ndarray, pd.Series, pd.DataFrame],
                 values: Union[np.ndarray, pd.Series, pd.DataFrame, List[Union[np.ndarray, pd.Series, pd.DataFrame]]],
                 labels: List[str],
                 colors: Optional[List[str]] = None,
                 line_style: str = '-',
                 title: Optional[str] = None) -> None:

        self.dates = self._convert_to_numpy_array(dates)
        self.values = [self._convert_to_numpy_array(val) for val in self._ensure_list(values)]
        self.labels = labels
        self.colors = colors
        self.line_style = line_style
        self.title = title

        if len(self.values) != len(self.labels):
            logging.error("Mismatch between number of value lists and labels.")
            raise ValueError("Values and labels must have the same length.")

    @staticmethod
    def _convert_to_numpy_array(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        
        """
        Converts data to a numpy array if it isn't one already.
        """

        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.values
        
        if isinstance(data, np.ndarray):
            return data

        raise TypeError("Expected data to be of type pd.Series, pd.DataFrame, or np.ndarray.")

    @staticmethod
    def _ensure_list(data: Union[np.ndarray, pd.Series, pd.DataFrame, List]) -> List:
        
        """
        Ensures the input is a list.
        """

        return data if isinstance(data, list) else [data]

    def _plot_data_scatter(self, ax, dates, values, label, color_mapping=None):
        
        """
        Utility method to plot data on an axis.
        """

        if color_mapping is not None and len(color_mapping) > 0:
            ax.scatter(dates, values, color=color_mapping, s=10, label=label)
        else:
            ax.plot(dates, values, label=label, linestyle=self.line_style)
        ax.set_ylabel(label, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc="upper left")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    def _plot_data_lines_vertical(self, ax, dates, values, label, color_mapping=None):
        """Utility method to plot data on an axis."""

        # Plot the main line (navy color)
        ax.plot(dates, values, label=label, linestyle=self.line_style, color='black')

        # If color mapping is provided, plot colored vertical lines
        if color_mapping is not None and len(color_mapping) > 0:
            for i, date in enumerate(dates): 
                ax.axvline(date, color=color_mapping[i], alpha=0.5)  # 50% transparent

        ax.set_ylabel(label, color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc="upper left")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def plot(self, scatter: bool = True) -> None:
        """
        Plot the provided data.
        """

        # Simplified color assignment
        if self.colors is None:
            color_values = [None] * len(self.labels) # Default behavior
        else:
            color_values = self.colors

        plt.style.use('ggplot')
        fig, axes = plt.subplots(nrows=len(self.values), figsize=(20, 10), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, values, label, color in zip(axes, self.values, self.labels, color_values):
            if scatter:
                self._plot_data_scatter(ax, self.dates, values, label, color)
            else:
                self._plot_data_lines_vertical(ax, self.dates, values, label, self.colors)

            # Reduce the number of x-axis ticks for the last subplot
            if ax is axes[-1]:
                ax.set_xticks(ax.get_xticks()[::200])

        fig.patch.set_facecolor('white')
        if self.title:
            plt.suptitle(self.title)

        plt.tight_layout()
        plt.show()



