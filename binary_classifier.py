"""
Binary Channel Classifier

A simple binary classifier for per-channel occupancy detection.
Works with both WiFi (4 channels) and LTE-M (16 channels) data.
"""

import numpy as np
from typing import Tuple, Optional, List
import json


class BinaryChannelClassifier:
    """
    Binary classifier for single-channel occupancy detection.
    
    This classifier determines if a single channel is occupied (1) or free (0)
    based on I/Q data from that channel.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        """
        Initialize the binary classifier.
        
        Args:
            input_size: Size of the input feature vector
            hidden_size: Size of hidden layer
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Simple two-layer network (simulated with random weights for demo)
        # In practice, use PyTorch, TensorFlow, or scikit-learn
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None
        self.bias2 = None
        self.is_trained = False
        
    def extract_features(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Extract features from I/Q data for classification.
        
        Features include:
        - Power statistics (mean, std, max)
        - Spectral features
        - Time-domain characteristics
        
        Args:
            iq_data: Complex I/Q data
            
        Returns:
            Feature vector
        """
        features = []
        
        # Power features
        power = np.abs(iq_data) ** 2
        features.extend([
            np.mean(power),
            np.std(power),
            np.max(power),
            np.median(power)
        ])
        
        # Spectral features (FFT)
        if len(iq_data) > 0:
            fft = np.fft.fft(iq_data)
            fft_mag = np.abs(fft)
            features.extend([
                np.mean(fft_mag),
                np.std(fft_mag),
                np.max(fft_mag)
            ])
        
        # Time-domain features
        features.extend([
            np.mean(np.real(iq_data)),
            np.std(np.real(iq_data)),
            np.mean(np.imag(iq_data)),
            np.std(np.imag(iq_data))
        ])
        
        # Instantaneous phase
        phase = np.angle(iq_data)
        phase_diff = np.diff(phase)
        features.extend([
            np.mean(phase_diff),
            np.std(phase_diff)
        ])
        
        # Zero-crossing rate
        real_crossings = np.sum(np.diff(np.sign(np.real(iq_data))) != 0)
        imag_crossings = np.sum(np.diff(np.sign(np.imag(iq_data))) != 0)
        features.extend([
            real_crossings / len(iq_data),
            imag_crossings / len(iq_data)
        ])
        
        feature_vector = np.array(features)
        
        # Pad or truncate to input_size
        if len(feature_vector) < self.input_size:
            feature_vector = np.pad(feature_vector, 
                                   (0, self.input_size - len(feature_vector)))
        else:
            feature_vector = feature_vector[:self.input_size]
        
        return feature_vector
    
    def init_weights(self):
        """Initialize network weights."""
        # Xavier initialization
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.bias1 = np.zeros(self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, 1) * np.sqrt(2.0 / self.hidden_size)
        self.bias2 = np.zeros(1)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass through the network.
        
        Args:
            x: Input feature vector
            
        Returns:
            Prediction probability (0 to 1)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Layer 1
        h = self.relu(np.dot(x, self.weights1) + self.bias1)
        
        # Layer 2
        out = self.sigmoid(np.dot(h, self.weights2) + self.bias2)
        
        return float(out[0])
    
    def train(self, X: List[np.ndarray], y: List[int], 
              learning_rate: float = 0.01, epochs: int = 100):
        """
        Train the classifier (simplified training for demonstration).
        
        In practice, use proper deep learning frameworks.
        
        Args:
            X: List of I/Q data samples
            y: List of binary labels (0 or 1)
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        # Extract features from all samples
        X_features = np.array([self.extract_features(x) for x in X])
        y_array = np.array(y).reshape(-1, 1)
        
        # Initialize weights
        self.init_weights()
        
        # Simple gradient descent (for demonstration)
        for epoch in range(epochs):
            # Forward pass
            h = self.relu(np.dot(X_features, self.weights1) + self.bias1)
            predictions = self.sigmoid(np.dot(h, self.weights2) + self.bias2)
            
            # Compute loss (binary cross-entropy)
            loss = -np.mean(y_array * np.log(predictions + 1e-10) + 
                           (1 - y_array) * np.log(1 - predictions + 1e-10))
            
            # Backward pass (simplified)
            # d_loss/d_predictions
            d_predictions = (predictions - y_array) / len(y_array)
            
            # d_loss/d_weights2
            d_weights2 = np.dot(h.T, d_predictions)
            d_bias2 = np.sum(d_predictions, axis=0)
            
            # d_loss/d_h
            d_h = np.dot(d_predictions, self.weights2.T)
            d_h[h <= 0] = 0  # ReLU gradient
            
            # d_loss/d_weights1
            d_weights1 = np.dot(X_features.T, d_h)
            d_bias1 = np.sum(d_h, axis=0)
            
            # Update weights
            self.weights1 -= learning_rate * d_weights1
            self.bias1 -= learning_rate * d_bias1
            self.weights2 -= learning_rate * d_weights2
            self.bias2 -= learning_rate * d_bias2
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.is_trained = True
        print(f"Training completed after {epochs} epochs")
    
    def predict(self, iq_data: np.ndarray, threshold: float = 0.5) -> int:
        """
        Predict if a channel is occupied or free.
        
        Args:
            iq_data: Complex I/Q data from a single channel
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            1 if occupied, 0 if free
        """
        features = self.extract_features(iq_data)
        prob = self.forward(features)
        return 1 if prob >= threshold else 0
    
    def predict_proba(self, iq_data: np.ndarray) -> float:
        """
        Get the probability that a channel is occupied.
        
        Args:
            iq_data: Complex I/Q data from a single channel
            
        Returns:
            Probability between 0 and 1
        """
        features = self.extract_features(iq_data)
        return self.forward(features)
    
    def save(self, filepath: str):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'weights1': self.weights1.tolist(),
            'bias1': self.bias1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias2': self.bias2.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.weights1 = np.array(model_data['weights1'])
        self.bias1 = np.array(model_data['bias1'])
        self.weights2 = np.array(model_data['weights2'])
        self.bias2 = np.array(model_data['bias2'])
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


# Example usage
if __name__ == "__main__":
    print("Binary Channel Classifier Example\n")
    print("="*50)
    
    # Create classifier
    classifier = BinaryChannelClassifier(input_size=20, hidden_size=32)
    
    # Generate synthetic training data
    print("\n1. Generating synthetic training data...")
    n_samples = 200
    X_train = []
    y_train = []
    
    for i in range(n_samples):
        # Generate occupied channel (high power)
        if i < n_samples // 2:
            signal = (np.random.randn(100) + 1j * np.random.randn(100)) * 2.0
            label = 1
        # Generate free channel (low power/noise)
        else:
            signal = (np.random.randn(100) + 1j * np.random.randn(100)) * 0.1
            label = 0
        
        X_train.append(signal)
        y_train.append(label)
    
    print(f"   Created {len(X_train)} training samples")
    print(f"   - {sum(y_train)} occupied channels")
    print(f"   - {len(y_train) - sum(y_train)} free channels")
    
    # Train classifier
    print("\n2. Training classifier...")
    classifier.train(X_train, y_train, learning_rate=0.1, epochs=50)
    
    # Test classifier
    print("\n3. Testing classifier...")
    test_samples = [
        ((np.random.randn(100) + 1j * np.random.randn(100)) * 2.0, 1, "Occupied"),
        ((np.random.randn(100) + 1j * np.random.randn(100)) * 0.1, 0, "Free"),
        ((np.random.randn(100) + 1j * np.random.randn(100)) * 1.5, 1, "Occupied"),
        ((np.random.randn(100) + 1j * np.random.randn(100)) * 0.2, 0, "Free"),
    ]
    
    correct = 0
    for signal, true_label, description in test_samples:
        pred = classifier.predict(signal)
        prob = classifier.predict_proba(signal)
        correct += (pred == true_label)
        status = "✓" if pred == true_label else "✗"
        print(f"   {status} {description}: Predicted={pred}, Probability={prob:.3f}, True={true_label}")
    
    accuracy = correct / len(test_samples) * 100
    print(f"\n   Accuracy: {accuracy:.1f}%")
    
    print("\n" + "="*50)
    print("This binary classifier works with BOTH WiFi and LTE-M!")
    print("="*50)
