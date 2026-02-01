import pandas as pd
from sklearn.metrics import mean_absolute_error

def calculate_mae(y_true, y_pred):
    """Calculates Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)

def create_results_dataframe(results_list):
    """
    Creates a DataFrame from a list of experiment results.
    Args:
        results_list (list): List of dictionaries containing experiment results.
                             Each dict should have keys like 'konfiguracja', 'mae', 'czas_treningu'.
    Returns:
        pd.DataFrame: DataFrame with results.
    """
    df = pd.DataFrame(results_list)
    return df
