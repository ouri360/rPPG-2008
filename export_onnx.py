"""
This script exports the trained POSNet model to ONNX format, including telemetry outputs for TensorRT optimization.
"""

import torch
import os
from model import POSNet

def export_to_tensorrt_onnx():
    print("Loading PyTorch model weights...")
    # Initialize model with telemetry enabled
    model = POSNet(num_rois=9, return_telemetry=True)
    
    if not os.path.exists('pos_net_weights.pt'):
        print("Error: pos_net_weights.pt not found! Train the model first.")
        return
        
    model.load_state_dict(torch.load('pos_net_weights.pt', map_location='cpu'))
    model.eval()

    # Dummy input: Batch Size = 1, Channels = 2, SeqLen = 48, ROIs = 9
    dummy_input = torch.randn(1, 2, 48, 9, dtype=torch.float32)

    print("Exporting POSNet to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        "pos_net.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pulse', 'weights', 'alpha'], # We explicitly grab the telemetry!
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'seq_len'},
            'pulse': {0: 'batch_size', 1: 'seq_len'},
            'weights': {0: 'batch_size'},
            'alpha': {0: 'batch_size'}
        }
    )
    print("Success! 'pos_net.onnx' created.")

if __name__ == "__main__":
    export_to_tensorrt_onnx()