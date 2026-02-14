"""
Per-Channel Data Extraction Module

This module provides functionality to extract per-channel I/Q data from
multi-channel signals for binary sensing classification.

Problem: WiFi has 4 channels, LTE-M has 16 channels. Labels are incompatible.
Solution: Per-channel binary sensing - each channel is classified independently
         as occupied (1) or free (0).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class ChannelExtractor:
    """
    Extracts per-channel data from multi-channel I/Q signals.
    
    This allows binary classification on individual channels, making
    WiFi (4 channels) and LTE-M (16 channels) datasets compatible.
    """
    
    def __init__(self, num_channels: int, channel_bandwidth: float, 
                 sample_rate: float):
        """
        Initialize the channel extractor.
        
        Args:
            num_channels: Number of channels in the signal (4 for WiFi, 16 for LTE-M)
            channel_bandwidth: Bandwidth of each channel in Hz
            sample_rate: Sampling rate of the I/Q data in Hz
        """
        self.num_channels = num_channels
        self.channel_bandwidth = channel_bandwidth
        self.sample_rate = sample_rate
        self.channel_frequencies = self._calculate_channel_frequencies()
        
    def _calculate_channel_frequencies(self) -> np.ndarray:
        """
        Calculate center frequencies for each channel.
        
        Returns:
            Array of center frequencies for each channel
        """
        total_bandwidth = self.num_channels * self.channel_bandwidth
        # Center frequencies relative to baseband
        frequencies = np.linspace(
            -total_bandwidth / 2 + self.channel_bandwidth / 2,
            total_bandwidth / 2 - self.channel_bandwidth / 2,
            self.num_channels
        )
        return frequencies
    
    def extract_channel(self, iq_data: np.ndarray, channel_idx: int) -> np.ndarray:
        """
        Extract a single channel from multi-channel I/Q data.
        
        This method:
        1. Shifts the frequency to center the desired channel at baseband
        2. Applies a low-pass filter to isolate the channel
        3. Decimates to match the channel bandwidth
        
        Args:
            iq_data: Complex I/Q data (shape: [samples] or [batch, samples])
            channel_idx: Index of the channel to extract (0 to num_channels-1)
            
        Returns:
            Extracted single-channel I/Q data
        """
        if channel_idx < 0 or channel_idx >= self.num_channels:
            raise ValueError(f"Channel index must be between 0 and {self.num_channels-1}")
        
        # Get center frequency for this channel
        center_freq = self.channel_frequencies[channel_idx]
        
        # Generate time vector
        num_samples = iq_data.shape[-1] if iq_data.ndim > 1 else len(iq_data)
        t = np.arange(num_samples) / self.sample_rate
        
        # Frequency shift to move channel to baseband
        shift = np.exp(-2j * np.pi * center_freq * t)
        
        if iq_data.ndim == 1:
            shifted = iq_data * shift
        else:
            shifted = iq_data * shift[np.newaxis, :]
        
        # Low-pass filter using a simple moving average (can be replaced with better filter)
        filter_length = max(1, int(self.sample_rate / self.channel_bandwidth))
        
        if iq_data.ndim == 1:
            filtered = np.convolve(shifted, np.ones(filter_length)/filter_length, mode='same')
        else:
            filtered = np.array([
                np.convolve(shifted[i], np.ones(filter_length)/filter_length, mode='same')
                for i in range(shifted.shape[0])
            ])
        
        # Decimate to match channel bandwidth
        decimation_factor = max(1, int(self.sample_rate / (2 * self.channel_bandwidth)))
        decimated = filtered[..., ::decimation_factor]
        
        return decimated
    
    def extract_all_channels(self, iq_data: np.ndarray) -> List[np.ndarray]:
        """
        Extract all channels from multi-channel I/Q data.
        
        Args:
            iq_data: Complex I/Q data (shape: [samples] or [batch, samples])
            
        Returns:
            List of extracted single-channel I/Q data, one per channel
        """
        channels = []
        for channel_idx in range(self.num_channels):
            channel_data = self.extract_channel(iq_data, channel_idx)
            channels.append(channel_data)
        return channels
    
    def create_binary_labels(self, multi_label: np.ndarray) -> np.ndarray:
        """
        Convert multi-label classification to per-channel binary labels.
        
        Args:
            multi_label: Multi-hot encoded labels (shape: [num_channels])
                        1 if channel is occupied, 0 if free
            
        Returns:
            Binary labels for each channel (same shape as input)
        """
        return multi_label.astype(np.int32)
    
    def create_dataset_samples(self, iq_data: np.ndarray, 
                               labels: np.ndarray) -> Tuple[List[np.ndarray], List[int]]:
        """
        Create per-channel samples and labels for training/testing.
        
        This converts a multi-channel classification problem into multiple
        binary classification problems (one per channel).
        
        Args:
            iq_data: Complex I/Q data (shape: [samples] or [batch, samples])
            labels: Multi-hot encoded labels (shape: [num_channels] or [batch, num_channels])
            
        Returns:
            Tuple of (channel_samples, binary_labels)
            - channel_samples: List of I/Q data for each channel
            - binary_labels: List of binary labels (0 or 1) for each channel
        """
        channels = self.extract_all_channels(iq_data)
        
        channel_samples = []
        binary_labels = []
        
        if labels.ndim == 1:
            # Single sample
            for channel_idx, channel_data in enumerate(channels):
                channel_samples.append(channel_data)
                binary_labels.append(int(labels[channel_idx]))
        else:
            # Batch of samples
            for channel_idx, channel_data in enumerate(channels):
                for batch_idx in range(channel_data.shape[0]):
                    channel_samples.append(channel_data[batch_idx])
                    binary_labels.append(int(labels[batch_idx, channel_idx]))
        
        return channel_samples, binary_labels


def create_wifi_extractor(sample_rate: float = 20e6) -> ChannelExtractor:
    """
    Create a channel extractor configured for WiFi (4 channels).
    
    Args:
        sample_rate: Sampling rate in Hz (default: 20 MHz)
        
    Returns:
        ChannelExtractor configured for WiFi
    """
    return ChannelExtractor(
        num_channels=4,
        channel_bandwidth=5e6,  # 5 MHz per channel
        sample_rate=sample_rate
    )


def create_ltem_extractor(sample_rate: float = 30.72e6) -> ChannelExtractor:
    """
    Create a channel extractor configured for LTE-M (16 channels).
    
    Args:
        sample_rate: Sampling rate in Hz (default: 30.72 MHz)
        
    Returns:
        ChannelExtractor configured for LTE-M
    """
    return ChannelExtractor(
        num_channels=16,
        channel_bandwidth=1.4e6,  # 1.4 MHz per channel
        sample_rate=sample_rate
    )


# Example usage
if __name__ == "__main__":
    print("Per-Channel Binary Sensing Example\n")
    print("="*50)
    
    # Example 1: WiFi (4 channels)
    print("\n1. WiFi Configuration (4 channels)")
    wifi_extractor = create_wifi_extractor()
    print(f"   - Number of channels: {wifi_extractor.num_channels}")
    print(f"   - Channel bandwidth: {wifi_extractor.channel_bandwidth/1e6:.1f} MHz")
    print(f"   - Channel frequencies: {wifi_extractor.channel_frequencies/1e6} MHz")
    
    # Simulate WiFi I/Q data
    num_samples = 1000
    wifi_iq = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    wifi_labels = np.array([1, 0, 1, 0])  # Channels 0 and 2 occupied
    
    # Extract channels
    wifi_channels = wifi_extractor.extract_all_channels(wifi_iq)
    print(f"   - Extracted {len(wifi_channels)} channels")
    print(f"   - Original samples: {len(wifi_iq)}")
    print(f"   - Per-channel samples: {len(wifi_channels[0])}")
    
    # Create training samples
    samples, labels = wifi_extractor.create_dataset_samples(wifi_iq, wifi_labels)
    print(f"   - Total training samples: {len(samples)} (4 channels × 1 sample)")
    print(f"   - Binary labels: {labels}")
    
    # Example 2: LTE-M (16 channels)
    print("\n2. LTE-M Configuration (16 channels)")
    ltem_extractor = create_ltem_extractor()
    print(f"   - Number of channels: {ltem_extractor.num_channels}")
    print(f"   - Channel bandwidth: {ltem_extractor.channel_bandwidth/1e6:.1f} MHz")
    
    # Simulate LTE-M I/Q data
    ltem_iq = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
    ltem_labels = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    
    # Extract channels
    ltem_channels = ltem_extractor.extract_all_channels(ltem_iq)
    print(f"   - Extracted {len(ltem_channels)} channels")
    print(f"   - Original samples: {len(ltem_iq)}")
    print(f"   - Per-channel samples: {len(ltem_channels[0])}")
    
    # Create training samples
    samples, labels = ltem_extractor.create_dataset_samples(ltem_iq, ltem_labels)
    print(f"   - Total training samples: {len(samples)} (16 channels × 1 sample)")
    print(f"   - Binary labels: {labels}")
    
    print("\n" + "="*50)
    print("Key Benefit: Both WiFi and LTE-M can now use the")
    print("same binary classifier trained on per-channel data!")
    print("="*50)
