import torch
import torch.nn as nn
import torch.onnx
from typing import List, Optional, Dict, Any
import numpy as np

class RUL_1D_CNN(nn.Module):
    """
    Modular 1D-CNN architecture for Remaining Useful Life (RUL) prediction.
    Designed for ONNX compatibility and low-latency CPU inference.
    """
    def __init__(
        self,
        input_channels: int = 24,
        num_blocks: int = 2,
        out_channels_list: List[int] = [32, 64],
        kernel_size: int = 3,
        use_bn: bool = True,
        dropout: float = 0.1,
        fc_units: int = 32
    ):
        super(RUL_1D_CNN, self).__init__()
        
        if len(out_channels_list) != num_blocks:
            if len(out_channels_list) < num_blocks:
                out_channels_list = out_channels_list + [out_channels_list[-1]] * (num_blocks - len(out_channels_list))
            else:
                out_channels_list = out_channels_list[:num_blocks]

        layers = []
        in_c = input_channels
        
        for i in range(num_blocks):
            out_c = out_channels_list[i]
            padding = kernel_size // 2
            
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=padding))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_c = out_c
            
        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_c, fc_units),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fc_units, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x: (Batch, SeqLen, Features)
        """
        # Permute to (Batch, Features, SeqLen) for Conv1d
        x = x.permute(0, 2, 1)
        
        x = self.feature_extractor(x)
        x = self.global_pool(x) # (Batch, Channels, 1)
        x = torch.flatten(x, 1) # (Batch, Channels)
        
        x = self.fc(x)
        return x

def export_to_onnx(model: nn.Module, save_path: str, input_shape: tuple = (1, 30, 24)):
    """
    Exports the PyTorch model to ONNX format.
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
    
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    outputs = session.run(None, {input_name: dummy_input.astype(np.float32)})
    return outputs
