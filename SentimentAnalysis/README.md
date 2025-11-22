# Sentiment Analysis with Transformers

## Overview

This project implements a deep learning model for sentiment analysis on social media text using transformer architectures. The model classifies text into three sentiment categories: **Negative (0)**, **Neutral (1)**, and **Positive (2)**.

## Project Structure

```
Sentiment_Analysis/
├── dataset/                    # Dataset directory
│   ├── dataset.csv            # Original dataset (60,000 samples)
│   ├── train.csv              # Training set (42,000 samples)
│   ├── val.csv                # Validation set (9,000 samples)
│   └── test.csv               # Test set (9,000 samples)
├── saved_models/              # Trained model storage
│   ├── checkpoint/            # Best model checkpoint
│   │   ├── config.json        # Model configuration
│   │   ├── model.safetensors  # Model weights
│   │   ├── tokenizer files    # Tokenizer configuration
│   │   ├── *_confusion_matrix.csv      # Confusion matrices
│   │   ├── *_classification_report.txt # Classification reports
│   │   └── training_history.csv        # Training metrics
│   ├── summary.json           # Training summary
│   ├── training_history.png   # Training curves plot
│   ├── confusion_matrices.png # Confusion matrix visualization
│   └── performance_summary.png# Performance comparison plot
├── model.py                   # Model architecture and configuration
├── main.py                    # Training, evaluation, and utility functions
├── data.ipynb                 # Dataset handling and exploration
├── run_training.py            # Training script runner
├── demo_training.py           # Comprehensive training demo
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Model Architecture

### Backbone
- **Base Model**: BERT-base-uncased (110M parameters)
- **Tokenizer**: BERT tokenizer with WordPiece encoding
- **Max Sequence Length**: 128 tokens

### Classification Head
- **Architecture**: Multi-layer perceptron (MLP) with residual connections
- **Input**: 768-dimensional BERT embeddings
- **Hidden Layer**: 1536 dimensions with ReLU activation
- **Output**: 3 classes (Negative, Neutral, Positive)
- **Regularization**: Dropout (0.1) and Layer Normalization

### Key Features
- **Pooling Strategy**: CLS token representation
- **Loss Function**: Cross-entropy loss
- **Optimization**: AdamW with different learning rates for encoder and head
- **Learning Rate Schedule**: Linear warmup + decay

## Environment Setup

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory

### Virtual Environment Setup

1. **Create and activate virtual environment:**
   ```bash
   cd DLLABALL
   python -m venv .venv
   .venv\\Scripts\\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
```
torch>=2.0.0+cu118
transformers>=4.30.0
datasets>=2.14.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```

## Training Configuration

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Model | bert-base-uncased | Pre-trained backbone |
| Epochs | 3-10 | Training epochs |
| Batch Size | 16 | Samples per batch |
| Max Length | 128 | Maximum sequence length |
| Encoder LR | 2e-5 | Learning rate for BERT |
| Head LR | 1e-4 | Learning rate for classifier |
| Dropout | 0.1 | Regularization rate |
| Warmup Ratio | 0.1 | LR warmup proportion |

### Data Split
- **Training**: 70% (42,000 samples)
- **Validation**: 15% (9,000 samples)  
- **Testing**: 15% (9,000 samples)
- **Stratification**: Balanced across sentiment classes

## Usage

### Quick Start Training

1. **Default training (recommended):**
   ```bash
   python run_training.py
   ```

2. **Custom hyperparameters:**
   ```bash
   python run_training.py --epochs 5 --batch_size 32 --lr_encoder 1e-5
   ```

3. **Comprehensive demo:**
   ```bash
   python demo_training.py
   ```

### Training Options
```bash
python run_training.py [OPTIONS]

Options:
  --model_name TEXT       Pre-trained model name [default: bert-base-uncased]
  --epochs INTEGER        Number of training epochs [default: 3]
  --batch_size INTEGER    Batch size [default: 16]
  --max_length INTEGER    Max sequence length [default: 128]
  --dropout FLOAT         Dropout rate [default: 0.1]
  --lr_encoder FLOAT      Encoder learning rate [default: 2e-5]
  --lr_head FLOAT         Head learning rate [default: 1e-4]
  --warmup_ratio FLOAT    Warmup ratio [default: 0.1]
  --out_dir TEXT          Output directory [default: ./saved_models/]
```

### Inference and Testing

```python
from main import predict_sentiment, test_model_inference

# Test on sample sentences
test_model_inference("./saved_models/checkpoint", num_samples=10)

# Predict on custom text
texts = ["I love this product!", "This is terrible", "It's okay"]
results = predict_sentiment(texts)
print(results["labels"])  # ['Positive', 'Negative', 'Neutral']
```

## Model Performance

### Expected Results
- **Training Accuracy**: ~95-98%
- **Validation Accuracy**: ~85-90%
- **Test Accuracy**: ~85-90%
- **Training Time**: ~45-60 minutes (3 epochs, GPU)

### Performance Metrics
- **Primary Metric**: Accuracy
- **Additional**: Precision, Recall, F1-score per class
- **Visualization**: Confusion matrices, training curves

## Visualization

The training pipeline automatically generates:

1. **Training History**: Loss and accuracy curves over epochs
2. **Confusion Matrices**: Performance breakdown by class for all splits
3. **Performance Summary**: Accuracy comparison across train/val/test sets

Plots are saved in `saved_models/` directory as PNG files.

## File Descriptions

### Core Files

- **`model.py`**: Contains model architecture classes (`SentimentClassifier`, `SentimentConfig`, `CustomBlock`)
- **`main.py`**: Training pipeline, evaluation functions, inference utilities, and visualization tools
- **`data.ipynb`**: Dataset exploration and preprocessing notebook

### Utility Scripts

- **`run_training.py`**: Command-line training script with customizable parameters
- **`demo_training.py`**: Comprehensive training demo with fixed optimal parameters

### Output Files

- **`saved_models/checkpoint/`**: Best model weights, configuration, and tokenizer
- **`saved_models/summary.json`**: Training results and model statistics
- **`saved_models/*.png`**: Visualization plots

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size (`--batch_size 8`)
2. **Slow Training**: Ensure GPU is being used (check CUDA installation)
3. **Import Errors**: Verify virtual environment activation and package installation

### Performance Tips

1. **Faster Training**: Use smaller max_length (`--max_length 64`)
2. **Better Accuracy**: Increase epochs (`--epochs 5`) or use larger model
3. **Memory Optimization**: Enable gradient checkpointing for large models

## Results Reproduction

To reproduce the final results:

1. **Set up environment** as described above
2. **Run training** with optimal parameters:
   ```bash
   python demo_training.py
   ```
3. **Expected outputs**:
   - Model checkpoint in `saved_models/checkpoint/`
   - Performance summary in `saved_models/summary.json`
   - Visualization plots in `saved_models/*.png`

## Technical Details

### Model Complexity
- **Parameters**: ~110M total (all trainable)
- **FLOPs**: ~44 GFLOPs per batch
- **Memory**: ~6GB GPU memory (batch_size=16)

### Implementation Features
- **Mixed Precision**: Automatic mixed precision for faster training
- **Gradient Clipping**: Prevents gradient explosion
- **Early Stopping**: Automatic stopping if validation doesn't improve
- **Checkpoint Saving**: Best model based on validation accuracy

## License

This project is for educational purposes as part of Deep Learning coursework.

## Contact

For questions or issues, please refer to the course materials or contact the instructor.