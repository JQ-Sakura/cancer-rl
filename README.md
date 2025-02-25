# Cancer Region Detection with Reinforcement Learning

A deep reinforcement learning approach for automated cancer region detection in medical images, combining PPO (Proximal Policy Optimization) with prompt-based segmentation.

## Project Overview

This project implements an innovative approach to cancer region detection using reinforcement learning. The system combines:

- Distributed PPO (Proximal Policy Optimization) training
- Prompt-based segmentation guidance
- Adaptive reward shaping
- Curriculum learning
- Advanced post-processing techniques

## Technical Architecture

### 1. Core Components

#### 1.1 Actor-Critic Network
- **Enhanced CNN Encoder**:
  - Multi-scale feature extraction
  - Channel and spatial attention mechanisms
  - Residual connections
  - Adaptive batch normalization
  - Dropout for regularization

- **Actor Network**:
  - Multi-layer policy network with LayerNorm
  - Tanh-activated action output
  - Learnable state-dependent action variance
  - Auxiliary value prediction task

- **Critic Network**:
  - Dual-stream value estimation
  - Layer normalization and residual connections
  - Gradient clipping for stability

#### 1.2 Environment (CancerEnv)
- **State Space**:
  - 3D MRI volumes (3 modalities)
  - Current segmentation mask
  - Entropy map for uncertainty estimation
  - History mask for tracking explored regions

- **Action Space**:
  - Continuous 3D coordinates (z, y, x)
  - Normalized to [-1, 1] range
  - Bounded by volume dimensions

- **Reward System**:
  - IoU and Dice coefficient improvements
  - Boundary accuracy metrics
  - Exploration bonuses
  - Adaptive reward shaping

#### 1.3 Region Growing Algorithm
- **Enhanced 3D Region Growing**:
  - Multi-modal similarity metrics
  - Gradient-aware growth
  - Adaptive thresholding
  - Size constraints
  - 26-connectivity support

### 2. Key Features

#### 2.1 Distributed Training
- Multi-GPU support with DDP
- Dynamic batch size adaptation
- Gradient accumulation
- Memory optimization
- Automatic mixed precision (AMP)

#### 2.2 Prompt-based Guidance
- Interactive region detection
- Distance-based influence maps
- Temperature-controlled softening
- Adaptive prompt placement

#### 2.3 Adaptive Mechanisms
- **Dynamic Reward Shaping**:
  - Performance-based scaling
  - Window-based adaptation
  - Multi-component rewards

- **Curriculum Learning**:
  - Progressive difficulty increase
  - Performance-based stage transitions
  - Adaptive exploration rates

- **Post-processing**:
  - Conditional Random Fields (CRF)
  - Connected component analysis
  - Morphological operations
  - Boundary refinement

#### 2.4 Memory Management
- Optimized replay buffer
- Automatic garbage collection
- CUDA memory management
- Batch size adaptation

## Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
nibabel>=3.2.0
scikit-image>=0.19.0
opencv-python>=4.5.0
wandb>=0.12.0
tensorboard>=2.8.0
pydensecrf>=1.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
pandas>=1.3.0
pillow>=8.3.0
pydicom>=2.2.0
SimpleITK>=2.1.0
monai>=0.8.0
gymnasium>=0.29.0
```

## Project Structure

```
.
├── configs/               # Configuration files
│   └── train_config.yaml  # Training hyperparameters
├── model/                # Trained model checkpoints
├── rl_module/            # Core RL implementation
│   ├── agents/          # PPO agent implementation
│   │   ├── ppo_agent.py # Main PPO agent
│   │   └── memory.py    # Replay buffer
│   ├── env/             # Environment implementation
│   │   ├── cancer_env.py    # Main environment
│   │   └── region_growing.py # Region growing algorithm
│   └── utils/           # Utility functions
│       ├── post_processing.py # Post-processing tools
│       ├── prompt_segmentation.py # Prompt-based segmentation
│       ├── adaptive_reward.py # Reward shaping
│       └── losses.py    # Custom loss functions
├── utils/               # General utilities
├── logs/               # Training logs
└── run.py              # Main training script
```

## Training Process

The training process includes several key components:

1. **Data Preparation**:
   - Loading 3D MRI volumes
   - Standardizing dimensions
   - Computing entropy maps

2. **Training Loop**:
   - Distributed training across GPUs
   - Curriculum learning stages
   - Adaptive reward shaping
   - Real-time metric tracking

3. **Optimization**:
   - PPO with clipped objective
   - Gradient accumulation
   - Automatic mixed precision
   - Memory optimization

4. **Monitoring**:
   - Real-time metrics via Weights & Biases
   - TensorBoard integration
   - Performance logging
   - Resource monitoring

To start training:

```bash
python run.py
```

## Configuration

Key configuration parameters in `configs/train_config.yaml`:

```yaml
env:
  similarity_threshold: 0.3
  min_region_size: 8
  max_region_size: 2000
  connectivity: 26
  use_prompt: true
  
  prompt_config:
    distance_threshold: 0.3
    temperature: 0.15
    min_prob: 0.15
    
  reward_shaping:
    window_size: 100
    initial_scale: 1.0
    adaptation_rate: 0.005
```

## Results

The system tracks multiple performance metrics:

- Dice coefficient
- IoU (Intersection over Union)
- Boundary accuracy
- Region detection statistics
- Training stability metrics

## License

[Your License]

## Acknowledgments

[Your Acknowledgments] 