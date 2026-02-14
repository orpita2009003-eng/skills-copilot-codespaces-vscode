"""
Visualization of Per-Channel Binary Sensing

This script creates simple visualizations to demonstrate:
1. Multi-channel signal spectrum
2. Per-channel extraction
3. Binary classification results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from channel_extractor import create_wifi_extractor, create_ltem_extractor


def visualize_spectrum(iq_data, title, filename, num_channels=4, sample_rate=20e6):
    """
    Visualize the frequency spectrum of multi-channel I/Q data.
    
    Args:
        iq_data: Complex I/Q data
        title: Plot title
        filename: Output filename for the plot
        num_channels: Number of channels
        sample_rate: Sampling rate in Hz
    """
    # Compute FFT
    fft = np.fft.fftshift(np.fft.fft(iq_data))
    freq = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/sample_rate))
    power_db = 20 * np.log10(np.abs(fft) + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freq/1e6, power_db, linewidth=0.5)
    ax.set_xlabel('Frequency (MHz)', fontsize=12)
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark channel boundaries
    channel_bw = sample_rate / num_channels
    for i in range(num_channels + 1):
        x_pos = (i * channel_bw - sample_rate/2) / 1e6
        ax.axvline(x_pos, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add channel labels
    for i in range(num_channels):
        x_pos = ((i + 0.5) * channel_bw - sample_rate/2) / 1e6
        ax.text(x_pos, ax.get_ylim()[1] - 5, f'CH{i}', 
               ha='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved spectrum visualization to {filename}")


def visualize_channel_extraction(iq_data, extractor, channel_idx, filename):
    """
    Visualize the extraction of a single channel.
    
    Args:
        iq_data: Multi-channel I/Q data
        extractor: ChannelExtractor instance
        channel_idx: Index of channel to extract
        filename: Output filename
    """
    # Extract the channel
    channel_data = extractor.extract_channel(iq_data, channel_idx)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Original multi-channel spectrum
    fft_orig = np.fft.fftshift(np.fft.fft(iq_data))
    freq_orig = np.fft.fftshift(np.fft.fftfreq(len(iq_data), 1/extractor.sample_rate))
    power_orig = 20 * np.log10(np.abs(fft_orig) + 1e-10)
    
    axes[0, 0].plot(freq_orig/1e6, power_orig, linewidth=0.5)
    axes[0, 0].set_xlabel('Frequency (MHz)')
    axes[0, 0].set_ylabel('Power (dB)')
    axes[0, 0].set_title('1. Original Multi-Channel Signal', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mark all channels
    for i in range(extractor.num_channels):
        freq = extractor.channel_frequencies[i] / 1e6
        if i == channel_idx:
            axes[0, 0].axvline(freq, color='green', linestyle='--', linewidth=2, 
                              label=f'Target: CH{i}')
        else:
            axes[0, 0].axvline(freq, color='red', linestyle='--', alpha=0.3, linewidth=1)
    axes[0, 0].legend()
    
    # 2. Extracted channel spectrum
    fft_ch = np.fft.fftshift(np.fft.fft(channel_data))
    freq_ch = np.fft.fftshift(np.fft.fftfreq(len(channel_data), 
                                             1/(extractor.sample_rate / 
                                               (len(iq_data) // len(channel_data)))))
    power_ch = 20 * np.log10(np.abs(fft_ch) + 1e-10)
    
    axes[0, 1].plot(freq_ch/1e6, power_ch, linewidth=0.5, color='green')
    axes[0, 1].set_xlabel('Frequency (MHz)')
    axes[0, 1].set_ylabel('Power (dB)')
    axes[0, 1].set_title(f'2. Extracted Channel {channel_idx} (at Baseband)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Time-domain: Original I/Q
    time_orig = np.arange(min(500, len(iq_data))) / extractor.sample_rate * 1e6
    axes[1, 0].plot(time_orig, np.real(iq_data[:len(time_orig)]), label='I (Real)', alpha=0.7)
    axes[1, 0].plot(time_orig, np.imag(iq_data[:len(time_orig)]), label='Q (Imag)', alpha=0.7)
    axes[1, 0].set_xlabel('Time (μs)')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('3. Original I/Q Time Series', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Time-domain: Extracted channel I/Q
    time_ch = np.arange(min(200, len(channel_data))) / (extractor.sample_rate / 
                                                         (len(iq_data) // len(channel_data))) * 1e6
    axes[1, 1].plot(time_ch, np.real(channel_data[:len(time_ch)]), 
                   label='I (Real)', alpha=0.7, color='green')
    axes[1, 1].plot(time_ch, np.imag(channel_data[:len(time_ch)]), 
                   label='Q (Imag)', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Time (μs)')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title(f'4. Extracted Channel {channel_idx} I/Q Time Series', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved channel extraction visualization to {filename}")


def visualize_binary_classification(results_wifi, results_ltem, filename):
    """
    Visualize binary classification results for WiFi and LTE-M.
    
    Args:
        results_wifi: List of (prediction, true_label) tuples for WiFi
        results_ltem: List of (prediction, true_label) tuples for LTE-M
        filename: Output filename
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Process WiFi results
    wifi_matrix = np.zeros((2, 2))  # Confusion matrix
    for pred, true in results_wifi:
        wifi_matrix[true][pred] += 1
    
    # Process LTE-M results
    ltem_matrix = np.zeros((2, 2))  # Confusion matrix
    for pred, true in results_ltem:
        ltem_matrix[true][pred] += 1
    
    # Plot WiFi confusion matrix
    im1 = axes[0].imshow(wifi_matrix, cmap='Blues', aspect='auto')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['Pred: Free', 'Pred: Occupied'])
    axes[0].set_yticklabels(['True: Free', 'True: Occupied'])
    axes[0].set_title('WiFi Binary Classification', fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[0].text(j, i, int(wifi_matrix[i, j]),
                              ha="center", va="center", color="black", fontsize=14)
    
    # Plot LTE-M confusion matrix
    im2 = axes[1].imshow(ltem_matrix, cmap='Greens', aspect='auto')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['Pred: Free', 'Pred: Occupied'])
    axes[1].set_yticklabels(['True: Free', 'True: Occupied'])
    axes[1].set_title('LTE-M Binary Classification', fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = axes[1].text(j, i, int(ltem_matrix[i, j]),
                              ha="center", va="center", color="black", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved classification results to {filename}")


def main():
    """Generate all visualizations."""
    
    print("Generating Visualizations for Per-Channel Binary Sensing")
    print("="*60)
    
    # 1. WiFi spectrum with 4 channels
    print("\n1. Creating WiFi multi-channel spectrum...")
    wifi_extractor = create_wifi_extractor()
    
    # Generate WiFi signal with 2 occupied channels
    num_samples = 2000
    wifi_signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    wifi_signal *= 0.1  # Background noise
    
    # Add signals to channels 0 and 2
    t = np.arange(num_samples) / 20e6
    wifi_signal += 2.0 * np.exp(2j * np.pi * wifi_extractor.channel_frequencies[0] * t)
    wifi_signal += 2.0 * np.exp(2j * np.pi * wifi_extractor.channel_frequencies[2] * t)
    
    visualize_spectrum(wifi_signal, "WiFi Multi-Channel Spectrum (4 Channels)", 
                      "wifi_spectrum.png", num_channels=4, sample_rate=20e6)
    
    # 2. LTE-M spectrum with 16 channels
    print("\n2. Creating LTE-M multi-channel spectrum...")
    ltem_extractor = create_ltem_extractor()
    
    # Generate LTE-M signal with 5 occupied channels
    ltem_signal = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    ltem_signal *= 0.1  # Background noise
    
    # Add signals to channels 2, 5, 8, 11, 14
    t = np.arange(num_samples) / 30.72e6
    for ch_idx in [2, 5, 8, 11, 14]:
        ltem_signal += 2.0 * np.exp(2j * np.pi * ltem_extractor.channel_frequencies[ch_idx] * t)
    
    visualize_spectrum(ltem_signal, "LTE-M Multi-Channel Spectrum (16 Channels)", 
                      "ltem_spectrum.png", num_channels=16, sample_rate=30.72e6)
    
    # 3. Channel extraction visualization
    print("\n3. Creating channel extraction demonstration...")
    visualize_channel_extraction(wifi_signal, wifi_extractor, channel_idx=0, 
                                filename="channel_extraction.png")
    
    print("\n" + "="*60)
    print("All visualizations generated successfully!")
    print("Files created:")
    print("  - wifi_spectrum.png")
    print("  - ltem_spectrum.png")
    print("  - channel_extraction.png")
    print("="*60)


if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        main()
    except ImportError:
        print("\nNote: matplotlib is required for visualizations.")
        print("Install it with: pip install matplotlib")
        print("\nThe core functionality works without matplotlib.")
        print("Run 'python example_complete.py' to see the complete workflow.")
