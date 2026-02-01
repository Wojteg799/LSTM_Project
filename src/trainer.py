import time
from keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self, patience=5):
        """
        Args:
            patience (int): Patience for EarlyStopping callback.
        """
        self.patience = patience

    def train(self, model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Trains the Keras model with EarlyStopping.
        
        Args:
            model: Keras model instance.
            X_train: Training data.
            y_train: Training targets.
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size.
            validation_split (float): Fraction of data to use for validation.
            
        Returns:
            tuple: (history object, training_time in seconds)
        """
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=self.patience, 
            restore_best_weights=True
        )
        
        print("Starting training...")
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds.")
        
        return history, training_time

    def generate_predictions(self, model, X_test):
        """
        Generates predictions using the trained model.
        """
        return model.predict(X_test)
