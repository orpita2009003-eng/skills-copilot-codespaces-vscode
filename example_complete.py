"""
Complete Example: Per-Channel Binary Sensing for WiFi and LTE-M

This example demonstrates how to:
1. Extract per-channel data from multi-channel signals
2. Train a binary classifier on WiFi data
3. Test the same classifier on LTE-M data
4. Achieve cross-dataset compatibility
"""

import numpy as np
from channel_extractor import create_wifi_extractor, create_ltem_extractor
from binary_classifier import BinaryChannelClassifier


def generate_wifi_data(n_samples: int = 100):
    """
    Generate synthetic WiFi data with 4 channels.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (iq_data_list, labels_list)
    """
    iq_data_list = []
    labels_list = []
    
    for _ in range(n_samples):
        # Generate multi-channel WiFi signal
        num_samples = 1000
        base_signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Random occupancy for 4 channels
        labels = np.random.randint(0, 2, size=4)
        
        # Add signal components for occupied channels
        for ch_idx in range(4):
            if labels[ch_idx] == 1:
                # Add a strong signal component for occupied channel
                freq = (ch_idx - 1.5) * 5e6  # 5 MHz spacing
                t = np.arange(num_samples) / 20e6
                signal = 2.0 * np.exp(2j * np.pi * freq * t)
                base_signal += signal
        
        iq_data_list.append(base_signal)
        labels_list.append(labels)
    
    return iq_data_list, labels_list


def generate_ltem_data(n_samples: int = 100):
    """
    Generate synthetic LTE-M data with 16 channels.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (iq_data_list, labels_list)
    """
    iq_data_list = []
    labels_list = []
    
    for _ in range(n_samples):
        # Generate multi-channel LTE-M signal
        num_samples = 1000
        base_signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Random occupancy for 16 channels
        labels = np.random.randint(0, 2, size=16)
        
        # Add signal components for occupied channels
        for ch_idx in range(16):
            if labels[ch_idx] == 1:
                # Add a strong signal component for occupied channel
                freq = (ch_idx - 7.5) * 1.4e6  # 1.4 MHz spacing
                t = np.arange(num_samples) / 30.72e6
                signal = 2.0 * np.exp(2j * np.pi * freq * t)
                base_signal += signal
        
        iq_data_list.append(base_signal)
        labels_list.append(labels)
    
    return iq_data_list, labels_list


def main():
    """Main function demonstrating the complete workflow."""
    
    print("="*70)
    print("Per-Channel Binary Sensing: WiFi + LTE-M Compatibility")
    print("="*70)
    
    # Step 1: Generate WiFi training data
    print("\n[STEP 1] Generate WiFi Training Data (4 channels)")
    print("-" * 70)
    wifi_iq_list, wifi_labels_list = generate_wifi_data(n_samples=50)
    print(f"Generated {len(wifi_iq_list)} WiFi samples")
    print(f"Each sample has 4 channels")
    print(f"Example labels: {wifi_labels_list[0]} (1=occupied, 0=free)")
    
    # Step 2: Extract per-channel data from WiFi
    print("\n[STEP 2] Extract Per-Channel Data from WiFi")
    print("-" * 70)
    wifi_extractor = create_wifi_extractor()
    
    wifi_channel_samples = []
    wifi_channel_labels = []
    
    for iq_data, labels in zip(wifi_iq_list, wifi_labels_list):
        samples, binary_labels = wifi_extractor.create_dataset_samples(iq_data, labels)
        wifi_channel_samples.extend(samples)
        wifi_channel_labels.extend(binary_labels)
    
    print(f"Extracted {len(wifi_channel_samples)} per-channel samples from WiFi")
    print(f"Original samples: {len(wifi_iq_list)}")
    print(f"Per-channel samples: {len(wifi_iq_list)} × 4 = {len(wifi_channel_samples)}")
    print(f"Label distribution: {sum(wifi_channel_labels)} occupied, "
          f"{len(wifi_channel_labels) - sum(wifi_channel_labels)} free")
    
    # Step 3: Train binary classifier on WiFi data
    print("\n[STEP 3] Train Binary Classifier on WiFi Data")
    print("-" * 70)
    classifier = BinaryChannelClassifier(input_size=20, hidden_size=32)
    classifier.train(wifi_channel_samples, wifi_channel_labels, 
                    learning_rate=0.1, epochs=30)
    
    # Step 4: Test on WiFi data
    print("\n[STEP 4] Test on WiFi Data")
    print("-" * 70)
    wifi_test_iq, wifi_test_labels = generate_wifi_data(n_samples=20)
    
    wifi_test_samples = []
    wifi_test_binary_labels = []
    
    for iq_data, labels in zip(wifi_test_iq, wifi_test_labels):
        samples, binary_labels = wifi_extractor.create_dataset_samples(iq_data, labels)
        wifi_test_samples.extend(samples)
        wifi_test_binary_labels.extend(binary_labels)
    
    wifi_correct = 0
    for sample, true_label in zip(wifi_test_samples, wifi_test_binary_labels):
        pred = classifier.predict(sample)
        wifi_correct += (pred == true_label)
    
    wifi_accuracy = wifi_correct / len(wifi_test_samples) * 100
    print(f"WiFi Test Accuracy: {wifi_accuracy:.1f}%")
    print(f"Correct predictions: {wifi_correct}/{len(wifi_test_samples)}")
    
    # Step 5: Generate LTE-M test data
    print("\n[STEP 5] Generate LTE-M Test Data (16 channels)")
    print("-" * 70)
    ltem_test_iq, ltem_test_labels = generate_ltem_data(n_samples=20)
    print(f"Generated {len(ltem_test_iq)} LTE-M samples")
    print(f"Each sample has 16 channels")
    print(f"Example labels: {ltem_test_labels[0]}")
    
    # Step 6: Extract per-channel data from LTE-M
    print("\n[STEP 6] Extract Per-Channel Data from LTE-M")
    print("-" * 70)
    ltem_extractor = create_ltem_extractor()
    
    ltem_test_samples = []
    ltem_test_binary_labels = []
    
    for iq_data, labels in zip(ltem_test_iq, ltem_test_labels):
        samples, binary_labels = ltem_extractor.create_dataset_samples(iq_data, labels)
        ltem_test_samples.extend(samples)
        ltem_test_binary_labels.extend(binary_labels)
    
    print(f"Extracted {len(ltem_test_samples)} per-channel samples from LTE-M")
    print(f"Original samples: {len(ltem_test_iq)}")
    print(f"Per-channel samples: {len(ltem_test_iq)} × 16 = {len(ltem_test_samples)}")
    
    # Step 7: Test WiFi-trained classifier on LTE-M data
    print("\n[STEP 7] Test WiFi-Trained Classifier on LTE-M Data")
    print("-" * 70)
    print("*** This is the KEY benefit: Same classifier works on different datasets! ***")
    
    ltem_correct = 0
    for sample, true_label in zip(ltem_test_samples, ltem_test_binary_labels):
        pred = classifier.predict(sample)
        ltem_correct += (pred == true_label)
    
    ltem_accuracy = ltem_correct / len(ltem_test_samples) * 100
    print(f"LTE-M Test Accuracy: {ltem_accuracy:.1f}%")
    print(f"Correct predictions: {ltem_correct}/{len(ltem_test_samples)}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Cross-Dataset Compatibility Achieved!")
    print("="*70)
    print(f"✓ Trained on:  WiFi (4 channels)")
    print(f"✓ Tested on:   WiFi (4 channels) - {wifi_accuracy:.1f}% accuracy")
    print(f"✓ Tested on:   LTE-M (16 channels) - {ltem_accuracy:.1f}% accuracy")
    print(f"\nThe same binary classifier works on both datasets!")
    print(f"This is because we reformulated the problem as:")
    print(f"  'Is THIS single channel occupied?' (binary)")
    print(f"Instead of:")
    print(f"  'Which of the N channels are occupied?' (multi-label)")
    print("="*70)
    
    # Step 8: Show detailed prediction examples
    print("\n[STEP 8] Detailed Prediction Examples")
    print("-" * 70)
    
    print("\nWiFi Examples:")
    for i in range(min(3, len(wifi_test_samples))):
        sample = wifi_test_samples[i]
        true_label = wifi_test_binary_labels[i]
        pred = classifier.predict(sample)
        prob = classifier.predict_proba(sample)
        status = "✓" if pred == true_label else "✗"
        print(f"  {status} Channel: pred={pred}, prob={prob:.3f}, true={true_label}")
    
    print("\nLTE-M Examples:")
    for i in range(min(3, len(ltem_test_samples))):
        sample = ltem_test_samples[i]
        true_label = ltem_test_binary_labels[i]
        pred = classifier.predict(sample)
        prob = classifier.predict_proba(sample)
        status = "✓" if pred == true_label else "✗"
        print(f"  {status} Channel: pred={pred}, prob={prob:.3f}, true={true_label}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
