# tests/Python/test_model.py
import sys
import os
import torch
import pytest

# Add ml_pipeline to the path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml_pipeline')))
from model import VocalGateModel, DepthwiseSeparableConv

def test_vocalgate_output_shape():
    """Tests if the model processes a standard 1-second MFCC input correctly."""
    # Standard input: (Channels, MFCC_Bins, Time_Frames) -> (1, 40, 61)
    model = VocalGateModel(input_shape=(1, 40, 61), num_classes=1)
    
    # Batch size of 2, 1 channel, 40 mels, 61 frames
    dummy_input = torch.randn(2, 1, 40, 61) 
    output = model(dummy_input)
    
    expected_shape = (2, 1) # Batch size of 2, 1 logit output per batch item
    
    assert output.shape == expected_shape, f"Failed! Expected {expected_shape}, got {output.shape}"

def test_vocalgate_dynamic_flattening():
    """Tests if the model can dynamically adapt to a longer chunk size."""
    # Let's say we change our minds and want 2 seconds of audio (approx 122 frames)
    custom_shape = (1, 40, 122)
    model = VocalGateModel(input_shape=custom_shape, num_classes=1)
    
    dummy_input = torch.randn(1, 1, 40, 122)
    
    try:
        output = model(dummy_input)
    except RuntimeError as e:
        pytest.fail(f"Dynamic flattening failed for custom shape. Error: {e}")
        
    assert output.shape == (1, 1), "Output shape should still be (1, 1) for a single batch item."

def test_depthwise_separable_conv_shape():
    """Ensures the custom Depthwise Separable Conv layer preserves spatial dimensions with padding=1."""
    layer = DepthwiseSeparableConv(in_channels=8, out_channels=16)
    
    # Batch=1, Channels=8, Height=20, Width=30
    dummy_input = torch.randn(1, 8, 20, 30)
    output = layer(dummy_input)
    
    # Spatial dimensions should remain 20x30 because of stride=1 and padding=1
    expected_shape = (1, 16, 20, 30)
    assert output.shape == expected_shape, f"Failed! Expected {expected_shape}, got {output.shape}"

