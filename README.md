# I2Net: Medical Image Segmentation Based on Implicit Parameterized Implicit Functions and Orthogonal Mining

## Overview

I2Net is a deep learning framework for medical image segmentation that leverages implicit parameterized implicit functions and orthogonal mining techniques. The project combines the power of U-Net architecture with innovative I2Net (Implicit Implicit Network) modules to achieve superior segmentation performance.

## Features

- **Implicit Parameterized Implicit Functions**: Utilizes I2Net layers that learn multiple implicit functions through a gating mechanism
- **Orthogonal Mining**: Implements orthogonal gradient constraints to encourage diverse feature learning
- **Feature Pyramid Network (FPN) Integration**: Combines FPN structure with I2Net for multi-scale feature extraction
- **Spatial Encoding**: Advanced position encoding mechanisms for better spatial understanding
- **Flexible Architecture**: Supports various encoder backbones (e.g., ResNet18)

## Project Structure

```
I2Net/
├── config_i2net.yaml          # Configuration file for training and model parameters
├── networks/
│   ├── base.py                # Base network components (PPM, ASPP)
│   ├── dynamics.py             # Core I2Net implementation (I2Layer, I2Net, losses)
│   ├── fpn_i2net.py           # FPN integrated with I2Net
│   ├── i2net.py               # Implicit function network (i2net_simfpn)
│   ├── ifa_utils.py           # IFA utilities (spatial encoding, coordinate generation)
│   ├── unet_i2net.py          # Complete U-Net + I2Net model
│   └── unet_utils.py          # U-Net utility functions
```

## Key Components

### I2Net Module (`networks/dynamics.py`)
- **I2Layer**: Implicit layer that learns K different transformations through a gating mechanism
- **I2Net**: Multi-layer implicit network with configurable depth
- **GateShapingLoss**: Loss function for gate regularization
- **OrthGradLoss**: Orthogonal gradient loss for encouraging diverse feature learning

### IFA Utilities (`networks/ifa_utils.py`)
- **SpatialEncoding**: Fourier-based spatial position encoding
- **PositionEmbeddingLearned**: Learnable position embeddings
- **ifa_feat**: Implicit feature alignment for coordinate and feature extraction

### Model Architecture (`networks/unet_i2net.py`)
- **GlasUNetI2Net**: Complete segmentation model combining encoder, FPN-I2Net decoder, and output layers

## Configuration

The `config_i2net.yaml` file contains all training and model parameters:

- **Training**: batch_size, epochs, learning_rate, optimizer
- **Model**: encoder_model, inner_planes, patch_size
- **I2Net**: K (number of implicit functions), num_layer, gate_layer
- **Position Encoding**: pos_dim, ultra_pe

## Usage

### Basic Usage

```python
from networks.unet_i2net import GlasUNetI2Net
import yaml

# Load configuration
with open('config_i2net.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = GlasUNetI2Net(config)

# Forward pass
output = model(input_images)
```

### Training

Configure your training parameters in `config_i2net.yaml` and run your training script:

```yaml
batch_size: 16
epochs: 200
learning_rate: 1e-3
optimizer: torch.optim.Adam
```

## Dependencies

- PyTorch
- segmentation-models-pytorch
- NumPy
- PyYAML

## Model Architecture Details

The model follows an encoder-decoder structure:

1. **Encoder**: Uses a pre-trained backbone (default: ResNet18) to extract multi-scale features
2. **Decoder**: FPN-I2Net decoder that:
   - Processes features at multiple scales
   - Uses IFA (Implicit Feature Alignment) to align features
   - Applies I2Net for implicit function learning
   - Combines features with orthogonal mining constraints
3. **Output**: Final segmentation mask generation

## Key Innovations

1. **Implicit Parameterization**: Instead of explicit feature transformations, the model learns implicit functions parameterized by multiple learners
2. **Gating Mechanism**: Soft gating mechanism selects and combines multiple implicit functions
3. **Orthogonal Mining**: Orthogonal gradient constraints ensure diverse feature representations
4. **Spatial Awareness**: Advanced position encoding captures spatial relationships effectively

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add your citation here]
```

## Contact

[Add contact information if needed]

