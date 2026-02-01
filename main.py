import pandas as pd
from src.data_processor import DataProcessor
from src.model_builder import ModelBuilder
from src.trainer import ModelTrainer
from src.metrics import calculate_mae, create_results_dataframe
from src.visualizer import Visualizer

def main():
    print("=== Starting LSTM Refactoring Project Pipeline ===")

    # 1. Data Processing
    # Using 'NVDA' as per user request and notebook context
    processor = DataProcessor(ticker='NVDA', start_date='2020-01-01', end_date='2026-01-01', lookback=60)
    data = processor.fetch_data()
    
    if data is None or data.empty:
        print("Failed to fetch data or data is empty. Exiting.")
        return

    print("Preprocessing data...")
    X_train, y_train, X_test, y_test, scaler = processor.preprocess_data()
    input_shape = (X_train.shape[1], 1)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    # 2. Define Experiments
    # Simulating the experiments from the original notebook
    experiments = [
        {'name': 'Simple LSTM (1 layer, 50 units)', 'units': 50, 'layers': 1, 'dropout': 0.0},
        {'name': 'Deep LSTM (2 layers, 50 units)', 'units': 50, 'layers': 2, 'dropout': 0.0},
        {'name': 'LSTM + Dropout (2 layers, 100 units, 0.2)', 'units': 100, 'layers': 2, 'dropout': 0.2},
    ]

    results = []
    trained_models = {}

    trainer = ModelTrainer(patience=3) 

    # 3. Running Experiments
    for exp in experiments:
        print(f"\n--- Running Experiment: {exp['name']} ---")
        model = ModelBuilder.build_lstm_model(
            input_shape=input_shape,
            units=exp['units'],
            layers=exp['layers'],
            dropout=exp['dropout']
        )
        
        # Note: Using 5 epochs for demonstration speed. 
        # Increase this (e.g., to 20 or 50) for better results.
        history, training_time = trainer.train(model, X_train, y_train, epochs=5, batch_size=32)
        
        # Evaluation
        predictions_scaled = trainer.generate_predictions(model, X_test)
        
        # Inverse transform
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        mae = calculate_mae(y_test_original, predictions)
        print(f"Experiment MAE: {mae:.4f}")
        
        results.append({
            'konfiguracja': exp['name'],
            'mae': mae,
            'czas_treningu': training_time,
            'epochs_run': len(history.epoch)
        })
        trained_models[exp['name']] = model

    # 4. Results & Visualization
    results_df = create_results_dataframe(results)
    print("\n=== Experiment Results Summary ===")
    print(results_df)

    visualizer = Visualizer()
    visualizer.plot_experiment_results(results_df)

    # Find best model
    if not results_df.empty:
        best_exp = results_df.loc[results_df['mae'].idxmin()]
        best_model_name = best_exp['konfiguracja']
        print(f"\nBest Model: {best_model_name} with MAE: {best_exp['mae']:.4f}")

        # Plot predictions for best model
        best_model = trained_models[best_model_name]
        
        predictions_scaled = trainer.generate_predictions(best_model, X_test)
        predictions = scaler.inverse_transform(predictions_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        visualizer.plot_predictions(
            y_test_original, 
            predictions, 
            title=f"Best Model Predictions: {best_model_name}",
            filename='best_model_predictions.png'
        )
        print("\nPipeline completed successfully. Check 'output' directory for plots.")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
