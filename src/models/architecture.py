import torch
import torch.nn as nn
import torch.onnx
from typing import List, Optional, Dict, Any
import numpy as np

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for 1D convolutions.
    Uses Conv1d(1x1) as a parameter-efficient substitute for Linear layers.
    """
    def __init__(self, in_channels: int, reduction: int = 4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv1d(reduced_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution block with SE, BN, and GELU.
    Structure: Depthwise Conv -> Pointwise Conv -> SE -> BN -> GELU -> Dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, use_bn: bool, dropout: float, dilation: int):
        super(DepthwiseSeparableConv1d, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        # Depthwise: Grouped convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False, dilation=dilation)
        # Pointwise: 1x1 convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.se = SEBlock(out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else nn.Identity()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.se(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x

class RUL_1D_CNN(nn.Module):
    """
    MobileNet-inspired 1D-CNN for Remaining Useful Life (RUL) regression.
    Features per-block configuration for NAS and strict ONNX compatibility.
    """
    def __init__(
        self,
        input_channels: int = 24,
        num_blocks: int = 2,
        out_channels_list: List[int] = [32, 64],
        kernel_size: int = 3,
        use_bn_list: List[bool] = [True, True],
        dropout_list: List[float] = [0.1, 0.1],
        fc_units: int = 32,
        dilation: int = 1
    ):
        super(RUL_1D_CNN, self).__init__()
        
        # Ensure lists match num_blocks
        if len(out_channels_list) != num_blocks:
            out_channels_list = (out_channels_list * num_blocks)[:num_blocks]
        if len(use_bn_list) != num_blocks:
            use_bn_list = (use_bn_list * num_blocks)[:num_blocks]
        if len(dropout_list) != num_blocks:
            dropout_list = (dropout_list * num_blocks)[:num_blocks]

        layers = []
        in_c = input_channels
        
        for i in range(num_blocks):
            out_c = out_channels_list[i]
            layers.append(DepthwiseSeparableConv1d(
                in_c, out_c, kernel_size, use_bn_list[i], dropout_list[i], dilation=dilation
            ))
            in_c = out_c
            
        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Use nn.Flatten instead of .view for ONNX stability
        self.flatten = nn.Flatten(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_c, fc_units),
            nn.GELU(),
            nn.Dropout(dropout_list[-1]) if dropout_list[-1] > 0 else nn.Identity(),
            nn.Linear(fc_units, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: (Batch, SeqLen, Features)
        """
        # Transpose to (Batch, Features, SeqLen) for Conv1d
        x = x.transpose(1, 2)
        
        x = self.feature_extractor(x)
        x = self.global_pool(x) # (Batch, Channels, 1)
        x = self.flatten(x)     # (Batch, Channels)
        
        x = self.fc(x)
        return x

def export_to_onnx(model: nn.Module, save_path: str, input_shape: tuple = (1, 30, 24)):
    """
    Exports the PyTorch model to ONNX format with dynamic batch axis.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {save_path}")

def validate_onnx(onnx_path: str, dummy_input: np.ndarray):
    """
    Validates the exported ONNX model using onnxruntime.
    """
    import onnxruntime as ort
    
    # Force CPU provider for low-latency streaming context
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    outputs = session.run(None, {input_name: dummy_input.astype(np.float32)})
    return outputs
