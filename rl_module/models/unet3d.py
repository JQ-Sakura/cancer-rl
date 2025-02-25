import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    """Enhanced 3D Convolution Block with SE attention"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_se: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout)
        self.se = SEBlock(out_channels) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.se is not None:
            x = self.se(x)
        return x

class DeepSupervision(nn.Module):
    """Deep Supervision Module"""
    def __init__(self, in_channels: List[int], out_channels: int = 1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(ch, out_channels, kernel_size=1)
            for ch in in_channels
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        return [conv(f) for f, conv in zip(features, self.convs)]

class UNet3D(nn.Module):
    """Enhanced 3D UNet with additional features"""
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 16,
        num_levels: int = 4,
        use_se: bool = True,
        use_deep_supervision: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision
        
        # Encoder
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_levels):
            self.encoders.append(
                ConvBlock(current_channels, base_filters * (2**i),
                         use_se=use_se, dropout=dropout)
            )
            current_channels = base_filters * (2**i)
        
        # Bottleneck
        self.bottleneck = ConvBlock(
            base_filters * (2**(num_levels-1)),
            base_filters * (2**num_levels),
            use_se=use_se,
            dropout=dropout
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.deep_supervision_layers = []
        for i in range(num_levels-1, -1, -1):
            in_ch = base_filters * (2**(i+1)) * 3 if i < num_levels-1 else base_filters * (2**num_levels)
            out_ch = base_filters * (2**i)
            self.decoders.append(
                ConvBlock(in_ch, out_ch, use_se=use_se, dropout=dropout)
            )
            self.deep_supervision_layers.append(out_ch)
        
        # Deep Supervision
        if use_deep_supervision:
            self.deep_supervision = DeepSupervision(self.deep_supervision_layers)
        
        # Final layer
        self.final = nn.Conv3d(base_filters, 1, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module):
        """Initialize model weights"""
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            encoder_outputs.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with deep supervision
        decoder_outputs = []
        for i, decoder in enumerate(self.decoders):
            x = self.up(x)
            x = torch.cat([x, encoder_outputs[-i-1]], dim=1)
            x = decoder(x)
            decoder_outputs.append(x)
        
        # Apply deep supervision if enabled
        if self.use_deep_supervision and self.training:
            deep_outputs = self.deep_supervision(decoder_outputs)
            main_output = self.final(decoder_outputs[-1])
            return [main_output] + deep_outputs
        
        # Final output
        return self.final(decoder_outputs[-1])

class RegionScorer(nn.Module):
    """Region scoring network based on UNet3D features"""
    def __init__(
        self,
        base_model: UNet3D,
        base_filters: int = 16,
        freeze_base: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Scoring head
        self.score_head = nn.Sequential(
            nn.Linear(base_filters * 16, base_filters * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(base_filters * 8, base_filters * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(base_filters * 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get features from base model
        features = self.base_model.bottleneck(x)
        
        # Global average pooling
        pooled = self.gap(features).view(features.size(0), -1)
        
        # Generate score
        score = self.score_head(pooled)
        return score 