# Per-Channel Binary Sensing for WiFi and LTE-M

This repository implements per-channel binary sensing to enable cross-dataset compatibility between WiFi (4 channels) and LTE-M (16 channels) spectrum sensing.

## 🎯 Problem & Solution

**Problem**: WiFi has 4 channels, LTE-M has 16 channels. Labels are incompatible - you cannot directly train on one and test on the other.

**Solution**: Per-channel binary sensing reformulates the problem as:
> "Given I/Q data from a SINGLE sub-band, is it occupied (1) or free (0)?"

This binary classification approach works across both datasets!

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete example
python example_complete.py
```

## 📁 Files

- **`channel_extractor.py`** - Extract per-channel I/Q data from multi-channel signals
- **`binary_classifier.py`** - Binary classifier for channel occupancy detection  
- **`example_complete.py`** - Complete workflow demonstration
- **`DOCUMENTATION.md`** - Detailed technical documentation

## 💡 Key Features

✅ **Cross-Dataset Compatibility**: Train on WiFi, test on LTE-M (or vice versa)  
✅ **Scalable**: Works with any number of channels  
✅ **Simple**: Binary classification instead of complex multi-label  
✅ **Well-Documented**: Comprehensive examples and explanations

## 📖 How It Works

1. **Channel Extraction**: Isolate each channel using frequency shifting and filtering
2. **Feature Extraction**: Extract features from single-channel I/Q data
3. **Binary Classification**: Classify each channel as occupied (1) or free (0)

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed technical explanation.

## 🧪 Example Output

```
Per-Channel Binary Sensing: WiFi + LTE-M Compatibility
======================================================================

✓ Trained on:  WiFi (4 channels)
✓ Tested on:   WiFi (4 channels) - 85.0% accuracy
✓ Tested on:   LTE-M (16 channels) - 82.5% accuracy

The same binary classifier works on both datasets!
```

## 🔧 Usage

```python
from channel_extractor import create_wifi_extractor
from binary_classifier import BinaryChannelClassifier

# Extract channels from WiFi signal
extractor = create_wifi_extractor()
channels = extractor.extract_all_channels(iq_data)

# Train binary classifier
classifier = BinaryChannelClassifier(input_size=20)
classifier.train(channel_samples, binary_labels)

# Predict on new data
prediction = classifier.predict(new_channel_data)
```

## 📚 Learn More

- Read the [detailed documentation](DOCUMENTATION.md)
- Run individual examples: `python channel_extractor.py` or `python binary_classifier.py`
- Explore the complete workflow: `python example_complete.py`
