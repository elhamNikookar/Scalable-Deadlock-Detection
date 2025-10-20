import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ml_data_preparation import MLDataPreparation

class DiningPhilosopherML:
    def __init__(self):
        """Initialize the ML class."""
        self.ml_prep = MLDataPreparation()
        self.scaler = StandardScaler()
        
    def create_classification_model(self, input_shape: int) -> keras.Model:
        """Create a neural network for classification."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_regression_model(self, input_shape: int) -> keras.Model:
        """Create a neural network for regression."""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequence_model(self, vocab_size: int, max_length: int) -> keras.Model:
        """Create an LSTM model for sequence prediction."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, 32, input_length=max_length),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_classification_model(self):
        """Train a classification model to predict reach_2 property."""
        print("Training Classification Model (reach_2 property)...")
        
        # Prepare data
        X, y, feature_names = self.ml_prep.prepare_classification_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        model = self.create_classification_model(X_train.shape[1])
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=8,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        # Plot training history
        self.plot_training_history(history, "Classification Model Training")
        
        return model, history, (X_test_scaled, y_test, y_pred_binary)
    
    def train_deadlock_classification_model(self):
        """Train a classification model to predict deadlocks."""
        print("Training Deadlock Classification Model...")
        
        # Prepare data
        X, y, feature_names = self.ml_prep.prepare_deadlock_classification_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        model = self.create_classification_model(X_train.shape[1])
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=8,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        print("\nDeadlock Classification Report:")
        print(classification_report(y_test, y_pred_binary))
        
        # Plot training history
        self.plot_training_history(history, "Deadlock Classification Model Training")
        
        return model, history, (X_test_scaled, y_test, y_pred_binary)
    
    def train_regression_model(self):
        """Train a regression model to predict fork utilization."""
        print("Training Regression Model...")
        
        # Prepare data
        X, y, feature_names = self.ml_prep.prepare_regression_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        model = self.create_regression_model(X_train.shape[1])
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=8,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        
        print(f"\nRegression Results:")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred.flatten())):.4f}")
        print(f"Root Mean Square Error: {np.sqrt(np.mean((y_test - y_pred.flatten())**2)):.4f}")
        
        # Plot training history
        self.plot_training_history(history, "Regression Model Training")
        
        return model, history, (X_test_scaled, y_test, y_pred)
    
    def train_sequence_model(self):
        """Train an LSTM model for sequence prediction."""
        print("Training Sequence Model...")
        
        # Prepare sequence data
        sequences, actions = self.ml_prep.prepare_sequence_data()
        
        if len(sequences) == 0:
            print("No sequence data available for training.")
            return None, None, None
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in sequences)
        vocab_size = max(max(seq) for seq in sequences) + 1
        
        # Create training data
        X = []
        y = []
        
        for seq in sequences:
            if len(seq) > 1:  # Need at least 2 elements for prediction
                for i in range(1, len(seq)):
                    # Input: sequence up to i-1
                    input_seq = seq[:i]
                    # Pad to max_length
                    padded_seq = input_seq + [0] * (max_length - len(input_seq))
                    X.append(padded_seq)
                    y.append(seq[i])  # Predict next state
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        model = self.create_sequence_model(vocab_size, max_length)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=8,
            verbose=1
        )
        
        # Evaluate model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        print(f"\nSequence Model Results:")
        print(f"Accuracy: {np.mean(y_pred_classes == y_test):.4f}")
        
        return model, history, (X_test, y_test, y_pred_classes)
    
    def plot_training_history(self, history, title: str):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{title} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy/metrics
        if 'accuracy' in history.history:
            ax2.plot(history.history['accuracy'], label='Training Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title(f'{title} - Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
        else:
            ax2.plot(history.history['mae'], label='Training MAE')
            ax2.plot(history.history['val_mae'], label='Validation MAE')
            ax2.set_title(f'{title} - MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
        
        ax2.legend()
        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, model, feature_names: list):
        """Analyze feature importance for the model."""
        # Get feature importance using permutation
        from sklearn.inspection import permutation_importance
        
        X, y, feature_names = self.ml_prep.prepare_classification_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X_test_scaled, y_test, 
            n_repeats=10, random_state=42
        )
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=True)
        
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def generate_predictions(self, model, model_type: str = "classification"):
        """Generate predictions on new data."""
        if model_type == "classification":
            X, y, feature_names = self.ml_prep.prepare_classification_data()
            X_scaled = self.scaler.fit_transform(X)
            predictions = model.predict(X_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'state_id': [f's{i}' for i in range(len(X))],
                'actual': y,
                'predicted_probability': predictions.flatten(),
                'predicted_class': (predictions > 0.5).astype(int).flatten()
            })
            
            print("\nPrediction Results:")
            print(results)
            
            return results
        
        elif model_type == "deadlock_classification":
            X, y, feature_names = self.ml_prep.prepare_deadlock_classification_data()
            X_scaled = self.scaler.fit_transform(X)
            predictions = model.predict(X_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'state_id': [f's{i}' for i in range(len(X))],
                'actual_deadlock': y,
                'predicted_deadlock_probability': predictions.flatten(),
                'predicted_deadlock': (predictions > 0.5).astype(int).flatten()
            })
            
            print("\nDeadlock Prediction Results:")
            print(results)
            
            return results
        
        elif model_type == "regression":
            X, y, feature_names = self.ml_prep.prepare_regression_data()
            X_scaled = self.scaler.fit_transform(X)
            predictions = model.predict(X_scaled)
            
            # Create results DataFrame
            results = pd.DataFrame({
                'state_id': [f's{i}' for i in range(len(X))],
                'actual_fork_utilization': y,
                'predicted_fork_utilization': predictions.flatten(),
                'error': np.abs(y - predictions.flatten())
            })
            
            print("\nPrediction Results:")
            print(results)
            
            return results
    
    def close(self):
        """Close the ML preparation instance."""
        self.ml_prep.close()

def main():
    """Main function to run deep learning examples."""
    print("=== DINING PHILOSOPHER DEEP LEARNING EXAMPLES ===\n")
    
    # Create ML instance
    ml = DiningPhilosopherML()
    
    # Train classification model
    print("1. Training Classification Model (Predict reach_2 property)")
    clf_model, clf_history, clf_results = ml.train_classification_model()
    
    # Train deadlock classification model
    print("\n2. Training Deadlock Classification Model")
    deadlock_model, deadlock_history, deadlock_results = ml.train_deadlock_classification_model()
    
    # Train regression model
    print("\n3. Training Regression Model (Predict fork utilization)")
    reg_model, reg_history, reg_results = ml.train_regression_model()
    
    # Train sequence model
    print("\n4. Training Sequence Model (Predict next state)")
    seq_model, seq_history, seq_results = ml.train_sequence_model()
    
    # Analyze feature importance
    print("\n5. Analyzing Feature Importance")
    importance_df = ml.analyze_feature_importance(clf_model, 
        ['philosopher_count', 'fork_count', 'thinking_philosophers', 
         'hungry_philosophers', 'available_forks', 'held_forks',
         'total_entities', 'philosopher_fork_ratio', 'thinking_ratio',
         'hungry_ratio', 'fork_utilization', 'deadlock_risk', 'resource_contention'])
    
    print("\nFeature Importance Ranking:")
    print(importance_df.sort_values('importance', ascending=False))
    
    # Generate predictions
    print("\n6. Generating Predictions")
    clf_predictions = ml.generate_predictions(clf_model, "classification")
    deadlock_predictions = ml.generate_predictions(deadlock_model, "deadlock_classification")
    reg_predictions = ml.generate_predictions(reg_model, "regression")
    
    # Save models
    clf_model.save('classification_model.h5')
    deadlock_model.save('deadlock_classification_model.h5')
    reg_model.save('regression_model.h5')
    if seq_model:
        seq_model.save('sequence_model.h5')
    
    print("\nModels saved:")
    print("- classification_model.h5")
    print("- deadlock_classification_model.h5")
    print("- regression_model.h5")
    if seq_model:
        print("- sequence_model.h5")
    
    ml.close()
    print("\nDeep learning examples completed successfully!")

if __name__ == "__main__":
    main() 