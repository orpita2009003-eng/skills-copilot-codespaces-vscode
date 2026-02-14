# Per-Channel Binary Sensing for WiFi and LTE-M

## Problem Statement

**The Problem:**
- WiFi signals have 4 channels
- LTE-M signals have 16 channels
- Labels are incompatible between the two datasets
- You cannot directly train a model on one dataset and test on the other

**The Solution:**
Per-channel binary sensing reformulates the problem as:
> "Given a segment of I/Q data from a SINGLE sub-band, is this sub-band occupied (1) or free (0)?"

This is a **binary classification problem**, and it's compatible across both datasets.

## Architecture Overview

```
Multi-Channel Signal (WiFi: 4ch, LTE-M: 16ch)
              ↓
    Channel Extraction
    (Frequency shifting + filtering)
              ↓
    Per-Channel I/Q Data
              ↓
    Feature Extraction
              ↓
    Binary Classifier
              ↓
    Output: 0 (free) or 1 (occupied)
```

## How Per-Channel Data Extraction Works

### Conceptual Explanation

1. **Input**: Multi-channel I/Q signal containing all channels combined
2. **For each channel**:
   - **Frequency Shift**: Move the target channel to baseband (0 Hz)
   - **Low-Pass Filter**: Remove other channels, keep only the target channel
   - **Decimation**: Reduce sample rate to match channel bandwidth
3. **Output**: Isolated I/Q data for a single channel

### Mathematical Approach

For channel `i` with center frequency `f_i`:

1. **Frequency Shift**:
   ```
   s_shifted[n] = s[n] × e^(-j2πf_i·n/fs)
   ```
   Where `fs` is the sampling rate

2. **Low-Pass Filter**:
   ```
   s_filtered = LPF(s_shifted, cutoff = channel_bandwidth/2)
   ```

3. **Decimation**:
   ```
   s_decimated = s_filtered[::decimation_factor]
   ```

### Visual Example

```
Original Multi-Channel Signal:
Frequency: [-10MHz ←→ +10MHz]
    |    |    |    |
   CH0  CH1  CH2  CH3    (WiFi: 4 channels)
    |    |    |    |

After extracting CH1:
1. Shift CH1 to center (0 Hz)
2. Filter out CH0, CH2, CH3
3. Result: Single-channel I/Q data for CH1
```

## Usage Examples

### Example 1: Basic Channel Extraction

```python
from channel_extractor import create_wifi_extractor
import numpy as np

# Create WiFi extractor (4 channels)
extractor = create_wifi_extractor()

# Generate sample I/Q data
iq_data = np.random.randn(1000) + 1j * np.random.randn(1000)

# Extract all channels
channels = extractor.extract_all_channels(iq_data)
print(f"Extracted {len(channels)} channels")

# Extract single channel
channel_0 = extractor.extract_channel(iq_data, channel_idx=0)
print(f"Channel 0 shape: {channel_0.shape}")
```

### Example 2: Training a Binary Classifier

```python
from channel_extractor import create_wifi_extractor
from binary_classifier import BinaryChannelClassifier
import numpy as np

# Setup
extractor = create_wifi_extractor()
classifier = BinaryChannelClassifier(input_size=20)

# Prepare training data
iq_data = np.random.randn(1000) + 1j * np.random.randn(1000)
labels = np.array([1, 0, 1, 0])  # Channels 0,2 occupied; 1,3 free

# Extract per-channel samples
samples, binary_labels = extractor.create_dataset_samples(iq_data, labels)

# Train classifier
classifier.train(samples, binary_labels, epochs=50)

# Make predictions
test_channel = extractor.extract_channel(iq_data, 0)
prediction = classifier.predict(test_channel)
probability = classifier.predict_proba(test_channel)

print(f"Prediction: {prediction} (probability: {probability:.3f})")
```

### Example 3: Cross-Dataset Compatibility

```python
from channel_extractor import create_wifi_extractor, create_ltem_extractor
from binary_classifier import BinaryChannelClassifier

# Train on WiFi
wifi_extractor = create_wifi_extractor()
classifier = BinaryChannelClassifier(input_size=20)

# ... train classifier on WiFi data ...

# Test on LTE-M (different number of channels!)
ltem_extractor = create_ltem_extractor()

# ... extract LTE-M channels and test ...
# The SAME classifier works because each channel is
# evaluated independently as a binary problem!
```

## File Structure

```
.
├── channel_extractor.py     # Channel extraction logic
├── binary_classifier.py     # Binary classification model
├── example_complete.py      # Complete workflow demonstration
└── DOCUMENTATION.md         # This file
```

## Key Classes

### ChannelExtractor

Main class for extracting per-channel data from multi-channel signals.

**Methods:**
- `extract_channel(iq_data, channel_idx)`: Extract a single channel
- `extract_all_channels(iq_data)`: Extract all channels
- `create_dataset_samples(iq_data, labels)`: Create training samples

**Factory Functions:**
- `create_wifi_extractor()`: Configure for WiFi (4 channels, 5 MHz each)
- `create_ltem_extractor()`: Configure for LTE-M (16 channels, 1.4 MHz each)

### BinaryChannelClassifier

Binary classifier for per-channel occupancy detection.

**Methods:**
- `train(X, y)`: Train the classifier
- `predict(iq_data)`: Predict if channel is occupied (0 or 1)
- `predict_proba(iq_data)`: Get probability of occupancy
- `save(filepath)`: Save trained model
- `load(filepath)`: Load trained model

## Running the Examples

### Test Individual Components

```bash
# Test channel extraction
python channel_extractor.py

# Test binary classifier
python binary_classifier.py
```

### Run Complete Example

```bash
# Full workflow: WiFi training + LTE-M testing
python example_complete.py
```

Expected output:
```
Per-Channel Binary Sensing: WiFi + LTE-M Compatibility
======================================================================

[STEP 1] Generate WiFi Training Data (4 channels)
...

[STEP 7] Test WiFi-Trained Classifier on LTE-M Data
*** This is the KEY benefit: Same classifier works on different datasets! ***
LTE-M Test Accuracy: XX.X%
...

SUMMARY: Cross-Dataset Compatibility Achieved!
✓ Trained on:  WiFi (4 channels)
✓ Tested on:   WiFi (4 channels) - XX.X% accuracy
✓ Tested on:   LTE-M (16 channels) - XX.X% accuracy
```

## Benefits of Per-Channel Binary Sensing

1. **Dataset Compatibility**: Train on one dataset, test on another
2. **Scalability**: Works with any number of channels
3. **Simplicity**: Binary classification is simpler than multi-label
4. **Real-world Applicability**: Matches how spectrum sensing works in practice
5. **Better Generalization**: Channel-level features are more transferable

## Technical Details

### Feature Extraction

The classifier extracts these features from each channel's I/Q data:

1. **Power Statistics**
   - Mean, standard deviation, max, median of signal power

2. **Spectral Features**
   - FFT magnitude statistics

3. **Time-Domain Features**
   - I/Q statistics (mean, std of real and imaginary parts)

4. **Phase Features**
   - Instantaneous phase statistics

5. **Zero-Crossing Rate**
   - Frequency of sign changes in I and Q

### Channel Configuration

**WiFi (802.11)**
- Channels: 4
- Channel Bandwidth: 5 MHz
- Sample Rate: 20 MHz
- Total Bandwidth: 20 MHz

**LTE-M**
- Channels: 16
- Channel Bandwidth: 1.4 MHz
- Sample Rate: 30.72 MHz
- Total Bandwidth: 22.4 MHz

## Customization

### Add a New Signal Type

```python
from channel_extractor import ChannelExtractor

# Create custom extractor
custom_extractor = ChannelExtractor(
    num_channels=8,           # Your number of channels
    channel_bandwidth=2e6,    # 2 MHz per channel
    sample_rate=20e6          # 20 MHz sampling rate
)
```

### Modify Classification Features

Edit `BinaryChannelClassifier.extract_features()` in `binary_classifier.py` to add custom features.

### Use Deep Learning

The current implementation uses a simple 2-layer network for demonstration. For better performance, integrate with PyTorch or TensorFlow:

```python
import torch
import torch.nn as nn

class DeepBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)
```

## References

- **Spectrum Sensing**: The process of detecting occupied frequency channels
- **I/Q Data**: In-phase and Quadrature components of complex signals
- **Binary Classification**: Two-class classification (occupied vs. free)
- **Multi-Label Classification**: Predicting multiple labels simultaneously

## Troubleshooting

### Low Accuracy

- Increase training epochs
- Adjust learning rate
- Extract more training samples
- Add more features
- Use a deeper network

### Memory Issues

- Process channels in batches
- Reduce sample length
- Use lower sampling rates

### Compatibility Issues

- Ensure both datasets use the same feature extraction
- Normalize features across datasets
- Verify channel bandwidth settings

## Future Enhancements

1. **Convolutional Neural Networks**: Use CNNs directly on I/Q data
2. **Data Augmentation**: Add noise, fading, etc.
3. **Real Dataset Integration**: Test with actual WiFi/LTE-M captures
4. **Online Learning**: Update classifier with new data
5. **Multi-Task Learning**: Predict both occupancy and signal type

## License

MIT License - Feel free to use and modify for your needs.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.
