import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


class Triple_Barrier_Labeling:
    
    """
    Class that implements the Triple Barrier method and visualization for financial data.

    Attributes
    df : pd.DataFrame
        The DataFrame containing the financial data.
    column_name : str
        The column name in the DataFrame that will be used.
    y : pd.Series
        The Series extracted from the DataFrame based on column_name.

    Methods
    label_data():
        Assigns labels based on the Triple Barrier method.
    plot_data():
        Visualizes the data and the generated labels.
    """
    
    def __init__(self, df: pd.DataFrame, column_name: str) -> None:
        
        """
        Parameters
        df : pd.DataFrame
            The DataFrame containing the financial data.
        column_name : str
            The column name to be processed.
        plot : bool
            if True, will display column & labels. Defaults to False
        """

        self.df = df
        self.column_name = column_name
        self.y = self.df[column_name]

    def label_data(self) -> pd.DataFrame:
        
        logging.info("Labeling the data using the Triple Barrier method.")
        
        upper_barrier = self.y * 1.02
        lower_barrier = self.y * 0.98

        labels = []

        for idx, price in enumerate(self.y):
            if np.isnan(price):
                labels.append(np.nan)
                continue

            upper_hit, lower_hit = False, False
            
            for j in range(1, 2):  # period to label ---
                if idx + j >= len(self.y):
                    break

                next_price = self.y.iloc[idx + j]
                
                if np.isnan(next_price):
                    continue
                
                if next_price >= upper_barrier.iloc[idx]:
                    upper_hit = True
                    break
                elif next_price <= lower_barrier.iloc[idx]:
                    lower_hit = True
                    break

            if upper_hit:
                labels.append(1)
            elif lower_hit:
                labels.append(-1)
            else:
                labels.append(0)

        self.df['Labels'] = labels

        logging.info("Labeling completed.")

        return self.df

    def plot_data(self) -> None:
        
        """
        Visualizes the given column and its corresponding labels.
        """
        
        logging.info(f"Plotting {self.column_name} and labels.")

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6), sharex=True)

        # Plot the chosen column data
        axes[0].plot(self.df[self.column_name], color='blue', label=self.column_name)
        axes[0].set_title(self.column_name)
        axes[0].legend()

        # Plot Labels as a line plot
        axes[1].plot(self.df.index, self.df['Targets'], color='red', label='Label', alpha=0.6)
        axes[1].set_title('Targets')
        axes[1].legend()
        axes[1].set_xlabel('Index')

        plt.tight_layout()
        plt.show()

        logging.info(f"{self.column_name} and labels plotted successfully.")


# Usage example:
# df = pd.read_csv('Data/training_prediction_data.csv')
# TBF = Triple_Barrier_Labeling(df, 'Combined Prices')
# df = TBF.label_data()
# TBF.plot_data()
# print(df)
