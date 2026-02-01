import matplotlib.pyplot as plt
import os
import pandas as pd

class Visualizer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def plot_predictions(self, y_true, y_pred, title='Predictions', filename='predictions.png'):
        """
        Plots actual vs predicted values.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(y_true, label='Rzeczywista cena', color='blue', alpha=0.7)
        plt.plot(y_pred, label='Przewidywana cena', color='red', alpha=0.7)
        plt.title(title)
        plt.xlabel('Czas (dni)')
        plt.ylabel('Cena (USD)')
        plt.legend()
        plt.grid(True)
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")

    def plot_experiment_results(self, results_df, filename='mae_vs_time.png'):
        """
        Plots MAE vs Training Time for different experiments.
        Args:
            results_df (pd.DataFrame): DataFrame with columns 'mae', 'czas_treningu', 'konfiguracja'.
        """
        if results_df.empty:
            print("No results to plot.")
            return

        plt.figure(figsize=(14, 10))
        
        # Determine colors (naive logic: if 'color' column exists use it, else default)
        if 'color' in results_df.columns:
            colors = results_df['color']
        else:
            colors = 'blue'

        plt.scatter(
            results_df['czas_treningu'], 
            results_df['mae'], 
            c=colors, 
            s=100, 
            alpha=0.8
        )

        # Annotate points
        for i, row in results_df.iterrows():
            plt.annotate(
                row['konfiguracja'], 
                (row['czas_treningu'], row['mae']),
                textcoords="offset points", 
                xytext=(5, 5), 
                ha='left', 
                fontsize=9
            )

        plt.title('Porównanie konfiguracji LSTM: MAE vs Czas Treningu')
        plt.xlabel('Czas treningu (s)')
        plt.ylabel('MAE')
        plt.grid(True)
        
        # Invert Y axis because lower MAE is better
        plt.gca().invert_yaxis()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved results plot to {save_path}")
