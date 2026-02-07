# NVIDIA Stock Price Prediction with LSTM

## Project Overview
This project serves as a comprehensive framework for predicting NVIDIA (NVDA) stock prices using Long Short-Term Memory (LSTM) neural networks. Originally developed as a Jupyter Notebook, this codebase has been refactored into a modular, production-ready Python package. It allows for flexible experimentation with different model architectures, hyperparameters, and training configurations.

## Key Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules for data processing (`data_processor.py`), model construction (`model_builder.py`), training (`trainer.py`), and visualization (`visualizer.py`).
- **Live Data Integration**: Automatically fetches the latest stock data from Yahoo Finance using `yfinance`.
- **Experiment Tracking**: Capable of running multiple experiments in a single execution to compare different LSTM configurations (e.g., varying depth, units, dropout).
- **Automated Visualization**: Generates insightful plots including "MAE vs. Training Time" and "Actual vs. Predicted Prices".

## Project Structure
```text
LSTM_Project/
├── main.py                 # Primary entry point for running the pipeline
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── output/                 # Generated plots and results
└── src/
    ├── data_processor.py   # Handles data fetching (yfinance) and preprocessing
    ├── model_builder.py    # Factory for creating flexible Keras LSTM models
    ├── trainer.py          # Manages training loops and EarlyStopping
    ├── metrics.py          # Metrics calculation (MAE) and result aggregation
    └── visualizer.py       # Plotting utilities using Matplotlib
```

## Setup and Installation

1. **Clone the repository** (if applicable).
2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Run the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the full pipeline, simply execute the `main.py` script:

```bash
python main.py
```

### What happens when you run it?
1. The script downloads NVDA stock data from 2020 to 2026.
2. Data is normalized and split into training and testing sets.
3. Three predefined experiments are executed:
   - **Simple LSTM**: 1 layer, 50 units
   - **Deep LSTM**: 2 layers, 50 units
   - **LSTM + Dropout**: 2 layers, 100 units, 0.2 dropout
4. Training progress is displayed in the console.
5. After completion, a summary of results (MAE, Time) is printed.
6. The best performing model is identified.
7. Plots are saved to the `output/` directory:
   - `mae_vs_time.png`: Scatter plot comparing experiments.
   - `best_model_predictions.png`: Price prediction chart for the best model.

## Configuration
You can modify the experiments list in `main.py` to test different architectures. You can also adjust the `epochs` parameter in the training loop for longer training runs.


## Dependencies
- tensorflow
- pandas
- numpy
- scikit-learn
- matplotlib
- yfinance