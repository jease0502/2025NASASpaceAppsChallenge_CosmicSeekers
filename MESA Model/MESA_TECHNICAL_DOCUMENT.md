# MESA (Multimodal Exoplanet Semantic Alignment) Technical Documentation

## 📋 Project Overview

This project implements a CLIP-based multimodal exoplanet classification system named **MESA (Multimodal Exoplanet Semantic Alignment)**.
The system learns the **semantic alignment** between the **physical features** and **light curves** of exoplanets, enabling cross-modal reasoning and classification.

## 🎯 Core Features

### Multimodal Architecture

* **Modality 1: Light Curve** — Transformer-based temporal encoder
* **Modality 2: Planet Features** — radius, orbital period, transit depth, etc.
* **Modality 3: Stellar Features** — stellar temperature, radius, etc.

### Semantic Alignment Mechanism

* **Contrastive Learning**: InfoNCE loss for cross-modal alignment
* **Classification Task**: CONFIRMED, CANDIDATE, FALSE POSITIVE
* **Physical Constraints**: Feature design based on astrophysical principles

## 🏗️ Model Architecture

### Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        # Positional encoding + multi-head attention
        # CLS token for global representation
```

### MESA Model

```python
class MESAModel(nn.Module):
    def __init__(self, lightcurve_dim=1, planet_dim=3, stellar_dim=2, 
                 d_model=128, nhead=8, num_layers=4, num_classes=3):
        # Light curve Transformer encoder
        self.lightcurve_encoder = TransformerEncoder(...)
        # Physical feature encoders (MLP)
        self.planet_encoder = nn.Sequential(...)
        self.stellar_encoder = nn.Sequential(...)
        # Projection layers (for contrastive learning)
        self.lightcurve_projection = nn.Sequential(...)
        self.physics_projection = nn.Sequential(...)
        # Classifier
        self.classifier = nn.Sequential(...)
```

## 📊 Data Processing

### Data Sources

This project uses the **same dataset as TSFresh + LightGBM** for fair comparison:

* **Training Data**: `data/train_data.csv`
* **Test Data**: `data/test_data.csv`
* **Format**: Includes light curves, planetary features, stellar features, and labels

### Feature Engineering

```python
# Planet features
planet_features = ['period', 'depth', 'planet_radius']

# Stellar features  
stellar_features = ['stellar_temp', 'stellar_radius']

# Light curve generation
def _generate_lightcurve(self, row):
    # Generate physically realistic light curves
    # Includes transit events, noise, and constraints
```

### Data Augmentation

```python
def _augment_lightcurve(self, lightcurve):
    # Time shifting
    # Noise injection
    # Minor scaling
```

## 🚀 Usage

### 1. Environment Setup

```bash
# Install dependencies
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 2. Train the Model

#### Train from Scratch

```bash
python train.py --data_path data/train_data.csv --epochs 50
```

#### Using Pretrained Weights

```bash
# Use existing model as pretrained weights
python train.py \
    --data_path data/train_data.csv \
    --pretrained_path output/mesa_model.pth \
    --epochs 20

# Freeze backbone for fine-tuning
python train.py \
    --data_path data/train_data.csv \
    --pretrained_path output/mesa_model.pth \
    --freeze_backbone \
    --epochs 10
```

### 3. Test the Model

```bash
# Basic test
python test.py --model_path output/mesa_model.pth --test_data data/test_data.csv

# Generate visualizations
python test.py \
    --model_path output/mesa_model.pth \
    --test_data data/test_data.csv \
    --output_dir test_results \
    --visualize
```

## 🔧 Model Configuration

### Hyperparameters

```python
# Model architecture
d_model = 32          # Embedding dimension
nhead = 2             # Number of attention heads
num_layers = 1        # Transformer layers

# Training parameters
learning_rate = 1e-4   # Learning rate
batch_size = 16        # Batch size
weight_decay = 0.05    # Weight decay

# Loss weights
contrastive_weight = 0.1    # Contrastive loss weight
classification_weight = 1.0 # Classification loss weight
```

### Optimization Strategy

* **AdamW Optimizer**: Adam with weight decay
* **CosineAnnealingLR**: Cosine annealing learning rate scheduler
* **Gradient Clipping**: Prevent gradient explosion
* **Early Stopping**: Prevent overfitting

## 📈 Performance Metrics

### Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* **Classification Report**
* **t-SNE Visualization**
* **PCA Visualization**

### Expected Performance

| Method                           | Accuracy | Training Time | Parameters |
| -------------------------------- | -------- | ------------- | ---------- |
| **MESA (from scratch)**          | 60–65%   | Medium        | ~40K       |
| **MESA (pretrained fine-tuned)** | 70–75%   | Short         | ~40K       |
| **MESA (frozen fine-tune)**      | 65–70%   | Very short    | ~10K       |

## 🎨 Visualization Results

### Generated Plots

1. **Confusion Matrix** (`confusion_matrix.png`)
2. **t-SNE Visualization** (`tsne_visualization.png`)
3. **PCA Visualization** (`pca_visualization.png`)
4. **Training Loss Curve** (`training_loss.png`)
5. **Training Accuracy Curve** (`training_accuracy.png`)

### Result Analysis

* **Semantic Space**: Distribution of categories in embedding space
* **Alignment Quality**: Degree of correspondence between light curves and physical features
* **Decision Boundaries**: Classification boundaries learned by the model

## 🔬 Technical Innovations

### 1. Multimodal Alignment

* **CLIP-style Architecture**: Learns cross-modal semantic alignment
* **Contrastive Learning**: InfoNCE loss function
* **Physical Constraints**: Feature design grounded in astrophysical principles

### 2. Transformer Encoder

* **Positional Encoding**: Handles temporal information
* **Multi-Head Attention**: Captures long-range dependencies
* **CLS Token**: Enables global representation learning

### 3. Loss Function Design

```python
# Combined loss
total_loss = contrastive_weight * contrastive_loss + \
             classification_weight * classification_loss

# Contrastive loss (InfoNCE)
def contrastive_loss(self, lc_proj, physics_proj):
    similarity = torch.matmul(lc_proj, physics_proj.T) / temperature
    # Symmetric loss formulation

# Focal Loss (handles class imbalance)
def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
    # Focuses on hard-to-classify samples
```

## 📁 Directory Structure

```
multimodal_exoplanet_clip/
├── train.py                    # Training script
├── test.py                     # Testing script
├── data/
│   ├── train_data.csv          # Training data
│   └── test_data.csv           # Testing data
├── output/
│   ├── mesa_model.pth          # Trained model
│   ├── mesa_training_history.json
│   └── mesa_config.json
└── test_results/               # Test results
    ├── confusion_matrix.png
    ├── tsne_visualization.png
    ├── pca_visualization.png
    └── test_results.json
```

## 🚀 Quick Start

### 1. Prepare Data

Ensure that `data/train_data.csv` and `data/test_data.csv` are available.

### 2. Train the Model

```bash
# Basic training
python train.py --data_path data/train_data.csv --epochs 30

# Using pretrained weights
python train.py --pretrained_path output/mesa_model.pth --epochs 15
```

### 3. Test the Model

```bash
python test.py --model_path output/mesa_model.pth --test_data data/test_data.csv
```

### 4. View Results

Check the visual outputs under the `test_results/` directory.

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size`
2. **Training Not Converging**: Adjust learning rate or use pretrained weights
3. **Low Accuracy**: Increase epochs or enable data augmentation

### Performance Optimization

* Use GPU acceleration
* Tune batch size
* Use pretrained weights

## 📚 References

1. **CLIP** – *Learning Transferable Visual Representations from Natural Language Supervision*
2. **Transformer** – *Attention Is All You Need*
3. **InfoNCE** – *Representation Learning with Contrastive Predictive Coding*

## 👥 Team Members

* **Model Design**: MESA architecture design and implementation
* **Data Processing**: Consistent with TSFresh + LightGBM dataset
* **Experiment Evaluation**: Verification of multimodal alignment performance

---

**Note**:
This project uses the *exact same dataset* as the TSFresh + LightGBM method to ensure **fairness** and **reproducibility** in experiments.
