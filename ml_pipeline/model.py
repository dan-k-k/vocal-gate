# ml_pipeline/model.py
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    A Depthwise Separable Convolution block.
    Splits the spatial and channel-wise computations to save massive CPU cycles.
    """
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class VocalGateModel(nn.Module):
    def __init__(self, input_shape=(1, 40, 61), num_classes=1):
        """
        input_shape: (Channels, Mel_Bins, Time_Frames). 
        Default (1, 40, 61) corresponds to 1 sec of 16kHz audio with hop_length=256 and 40 Mels.
        """
        super(VocalGateModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Layer 2
            DepthwiseSeparableConv(in_channels=8, out_channels=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            DepthwiseSeparableConv(in_channels=16, out_channels=32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # ISSUE 3 FIX: Dynamically calculate the flattened size
        self._flattened_size = None
        self._calculate_flattened_size(input_shape)
        
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.5),           
            # Now we use the dynamically calculated size instead of a hardcoded 1120!
            nn.Linear(self._flattened_size, 64), 
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)   
        )

    def _calculate_flattened_size(self, shape):
        """Passes a dummy tensor through the conv layers to figure out the output math."""
        # Create a dummy tensor of zeros with a batch size of 1
        dummy_input = torch.zeros(1, *shape)
        
        # Pass it through the conv layers without tracking gradients
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
            
        # See how many features come out when we flatten it
        self._flattened_size = output.view(1, -1).size(1)
        print(f"🔧 Model initialized! Dynamic flattened linear size: {self._flattened_size}")

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layers(x)
        return x
    
