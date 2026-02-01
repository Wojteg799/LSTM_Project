import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import optimizers

class ModelBuilder:
    @staticmethod
    def build_lstm_model(input_shape, units=50, layers=2, dropout=0.0, learning_rate=0.001, optimizer_name='adam'):
        """
        Builds a flexible LSTM model based on provided parameters.
        
        Args:
            input_shape (tuple): Shape of the input data (time_steps, features).
            units (int): Number of units in LSTM layers.
            layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_name (str): Name of the optimizer ('adam' or 'sgd').
            
        Returns:
            model: Compiled Keras model.
        """
        model = Sequential()
        
        # Determine return_sequences for the first layer
        # If there are multiple layers, the first must return sequences
        return_sequences = layers > 1
        
        model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        if dropout > 0:
            model.add(Dropout(dropout))
            
        # Add hidden layers
        # Note: range starts from 1 because we already added the first layer (layer 0 logic relative to count)
        # We loop until layers-1 because the last added LSTM layer in the loop (or the first one if loop doesn't run)
        # determines the final output to the Dense layer.
        
        # Actually logic is simpler: we need 'layers' total LSTM layers.
        # We added 1. We need 'layers - 1' more.
        # The last LSTM layer generally should NOT return sequences for a simple regression Dense(1) output, 
        # unless we want 3D output which isn't the case for basic stock prediction 'Close' value.
        
        for i in range(1, layers):
            # If this is not the last layer, return sequences = T
            # If this IS the last layer, return sequences = F
            is_last_layer = (i == layers - 1)
            return_sequences = not is_last_layer
            
            model.add(LSTM(units=units, return_sequences=return_sequences))
            if dropout > 0:
                model.add(Dropout(dropout))
                
        model.add(Dense(units=1))

        if optimizer_name.lower() == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name.lower() == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate)
        else:
            print(f"Warning: Unknown optimizer '{optimizer_name}'. Defaulting to Adam.")
            opt = optimizers.Adam(learning_rate=learning_rate)

        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
        return model
