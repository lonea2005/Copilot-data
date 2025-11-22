'''
HW3: Sentiment Analysis with Deep Learning

In this homework, we will explore the fascinating field of sentiment analysis using deep learning techniques. 
Specifically, we will focus on multi-class classification, where the goal is to predict each sentence from social media 
as belonging to the label.

Label definition:
0 -> Negative
1 -> Neutral
2 -> Positive
'''

import os
import re
import gc
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Reproducibility
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


# Text Preprocessing Functions
def clean_text(text: str) -> str:
    """
    Clean and preprocess text for better sentiment analysis results.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove extra whitespaces and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


# FLOPs estimation
def estimate_flops(hidden_size: int, num_layers: int, seq_len: int, batch_size: int) -> float:
    '''
    Roughly estimate the number of floating point operations (FLOPs) 
    per training step for models

    Args:
        hidden_size: model embedding dimension
        num_layers: number of encoder layers
        seq_len: number of tokens per input
        batch_size: number of samples processed per step

    Returns:
        Estimated FLOPs per training step (in GFLOPs)
    '''
    # Transformer attention mechanism FLOPs
    # Self-attention: Q, K, V projections + attention computation + output projection
    attention_flops = 4 * hidden_size * hidden_size * seq_len * batch_size  # QKV + output projections
    attention_flops += 2 * hidden_size * seq_len * seq_len * batch_size      # attention computation
    
    # Feed-forward network FLOPs (typically 4x hidden_size intermediate)
    ffn_flops = 8 * hidden_size * hidden_size * seq_len * batch_size
    
    # Total per layer
    per_layer_flops = attention_flops + ffn_flops
    
    # Total for all layers
    total_flops = per_layer_flops * num_layers
    
    # Add embedding and classification head FLOPs
    embedding_flops = hidden_size * seq_len * batch_size  # rough estimate
    classification_flops = hidden_size * 3 * batch_size   # 3 classes
    
    total_flops += embedding_flops + classification_flops
    
    # Convert to GFLOPs
    gflops = total_flops / 1e9
    
    return gflops


# Dataset
class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer, max_length: int):
        """
        Step1. Load the CSV file using pandas -> HINT: use pd.read_csv(csv_path)
        Step2. Extract text and label columns -> HINT: df["text"].tolist(), df["label"].tolist()
        Step3. Store tokenizer and max_length for later use

        Args:
            csv_path: Path to the CSV file (with columns 'text' and 'label')
            tokenizer: Pre-trained tokenizer from Hugging Face
            max_length: Maximum token length for padding/truncation
        """
        # Step1: Load the CSV file using pandas
        df = pd.read_csv(csv_path)
        
        # Step2: Extract and preprocess text and label columns
        self.texts = [clean_text(text) for text in df["text"].tolist()]
        self.labels = df["label"].tolist()
        
        # Step3: Store tokenizer and max_length for later use
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns:
            Total number of samples in the dataset -> HINT: len(self.texts)
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Step1. Select text and label by index -> HINT: text = self.texts[idx]; label = self.labels[idx]
        Step2. Tokenize the text -> HINT: use self.tokenizer with truncation, padding, max_length, return_tensors="pt"
        Step3. Convert results to proper tensor format -> HINT: enc["input_ids"].squeeze(0)
        Step4. Return a dictionary

        Returns:
            One sample (tokenized text and label)
        """
        # Step1: Select text and label by index
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Step2: Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Step3: Convert results to proper tensor format
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# Import model components from model.py
from model import CustomBlock, SentimentConfig, SentimentClassifier

# Additional imports for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Example of Custom Block
class CustomMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Model Config
# SentimentConfig moved to model.py


# Model (DO NOT change the name "SentimentClassifier")
# SentimentClassifier moved to model.py


# Evaluation
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model accuracy on a given dataset.

    Args:
        model: the trained PyTorch model
        dataloader: DataLoader for validation or test set

    Returns:
        acc: overall accuracy
        all_y: true labels
        all_pred: predicted labels
    """
    model.eval()  
    all_y, all_pred = [], []
    total_loss = 0.0
    num_batches = 0
    
    with torch.inference_mode():  
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move the batch to the correct device (GPU/CPU)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Run a forward pass through the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Get predicted class from logits
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            
            # Save ground-truth and predicted labels
            all_y.extend(labels.cpu().numpy())
            all_pred.extend(predictions.cpu().numpy())
            
            # Accumulate loss
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                num_batches += 1
    
    acc = accuracy_score(all_y, all_pred)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return acc, np.array(all_y), np.array(all_pred)


# Inference Functions
def load_trained_model(model_path: str, device: str = "cuda"):
    """
    Load a trained sentiment analysis model.
    
    Args:
        model_path: Path to the saved model directory
        device: Device to load the model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoTokenizer
    
    # Load model and tokenizer
    model = SentimentClassifier.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    return model, tokenizer


def predict_sentiment(texts: list, model_path: str = "./saved_models/checkpoint", device: str = "cuda"):
    """
    Predict sentiment for a list of texts using trained model.
    
    Args:
        texts: List of text strings to analyze
        model_path: Path to trained model
        device: Device for inference
        
    Returns:
        Dictionary with predictions and probabilities
    """
    model, tokenizer = load_trained_model(model_path, device)
    
    # Use the model's predict method
    results = model.predict(texts, tokenizer, device=device)
    
    return results


def test_model_inference(model_path: str = "./saved_models/checkpoint", num_samples: int = 10):
    """
    Test model inference on sample sentences and display results.
    
    Args:
        model_path: Path to trained model
        num_samples: Number of test samples to show
    """
    print("Testing Model Inference")
    print("=" * 50)
    
    # Sample test sentences
    test_sentences = [
        "I absolutely love this product! It's amazing and exceeded my expectations.",
        "This is the worst thing I've ever bought. Complete waste of money.",
        "It's okay, nothing special but does the job.",
        "Fantastic quality and great customer service. Highly recommended!",
        "Terrible experience. The product broke after one day.",
        "Pretty good value for the price. Would buy again.",
        "I'm not sure how I feel about this. Mixed feelings.",
        "Outstanding! Best purchase I've made in years.",
        "Poor quality control. Several defects found.",
        "Decent product but could be improved."
    ]
    
    # Take only the requested number of samples
    test_sentences = test_sentences[:num_samples]
    
    try:
        # Get predictions
        results = predict_sentiment(test_sentences, model_path)
        
        print(f"Analyzing {len(test_sentences)} sample sentences:\n")
        
        for i, (text, label, probs) in enumerate(zip(
            results["texts"], 
            results["labels"], 
            results["probabilities"]
        )):
            print(f"Text {i+1}: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"Predicted: {label}")
            print(f"Confidence: Negative={probs[0]:.3f}, Neutral={probs[1]:.3f}, Positive={probs[2]:.3f}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error during inference: {e}")


# Visualization Functions
def plot_training_history(history_path: str = "./saved_models/checkpoint/training_history.csv", 
                         save_path: str = "./saved_models/training_plots.png"):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history_path: Path to training history CSV file
        save_path: Path to save the plot
    """
    try:
        import pandas as pd
        
        # Load training history
        history = pd.read_csv(history_path)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training loss
        ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot validation accuracy
        ax2.plot(history['epochs'], history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training history plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting training history: {e}")


def plot_confusion_matrices(checkpoint_dir: str = "./saved_models/checkpoint", 
                          save_path: str = "./saved_models/confusion_matrices.png"):
    """
    Plot confusion matrices for train, validation, and test sets.
    
    Args:
        checkpoint_dir: Directory containing confusion matrix CSV files
        save_path: Path to save the plot
    """
    try:
        import pandas as pd
        
        # Load confusion matrices
        splits = ['train', 'val', 'test']
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, split in enumerate(splits):
            cm_path = os.path.join(checkpoint_dir, f"{split}_confusion_matrix.csv")
            
            if os.path.exists(cm_path):
                cm_df = pd.read_csv(cm_path, index_col=0)
                
                # Create heatmap
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                           ax=axes[i], cbar=i==2)  # Only show colorbar for last plot
                axes[i].set_title(f'{split.capitalize()} Set Confusion Matrix')
                axes[i].set_ylabel('True Label' if i == 0 else '')
                axes[i].set_xlabel('Predicted Label')
            else:
                axes[i].text(0.5, 0.5, f'No {split} data available', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{split.capitalize()} Set (No Data)')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confusion matrices plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting confusion matrices: {e}")


def plot_performance_summary(checkpoint_dir: str = "./saved_models/checkpoint",
                           summary_path: str = "./saved_models/summary.json",
                           save_path: str = "./saved_models/performance_summary.png"):
    """
    Plot performance summary with accuracy comparison across splits.
    
    Args:
        checkpoint_dir: Directory containing results
        summary_path: Path to summary JSON file
        save_path: Path to save the plot
    """
    try:
        import json
        
        # Load summary
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Extract accuracies
        accuracies = {
            'Training': summary.get('train_accuracy', 0),
            'Validation': summary.get('val_accuracy', 0), 
            'Test': summary.get('test_accuracy', 0)
        }
        
        # Create bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        splits = list(accuracies.keys())
        values = list(accuracies.values())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax.bar(splits, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Across Different Splits')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add model info as text
        info_text = f"""Model Info:
• Total Parameters: {summary.get('params_total', 0):,}
• Trainable Parameters: {summary.get('params_trainable', 0):,}
• Epochs Trained: {summary.get('epochs_trained', 0)}
• Final Training Loss: {summary.get('final_train_loss', 0):.4f}"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Performance summary plot saved to: {save_path}")
        
    except Exception as e:
        print(f"Error plotting performance summary: {e}")


def visualize_results(checkpoint_dir: str = "./saved_models/checkpoint"):
    """
    Generate all visualization plots for the trained model.
    
    Args:
        checkpoint_dir: Directory containing model checkpoints and results
    """
    print("Generating Visualization Plots...")
    print("=" * 50)
    
    # Set up matplotlib for better plots
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Generate all plots
    plot_training_history(
        history_path=os.path.join(checkpoint_dir, "training_history.csv"),
        save_path="./saved_models/training_history.png"
    )
    
    plot_confusion_matrices(
        checkpoint_dir=checkpoint_dir,
        save_path="./saved_models/confusion_matrices.png"
    )
    
    plot_performance_summary(
        checkpoint_dir=checkpoint_dir,
        summary_path="./saved_models/summary.json",
        save_path="./saved_models/performance_summary.png"
    )
    
    print("\nAll visualization plots generated successfully!")


# Training Loop
def train(
    model_name: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    out_dir: str,
    epochs: int,
    batch_size: int,
    max_length: int,
    dropout: float = 0.1,
    lr_encoder: float = 2e-5,
    lr_head: float = 1e-4,
    warmup_ratio: float = 0.1,
    head: str = "mlp",
    seed: int = 42,
):
    '''
    Complete training pipeline for sentiment analysis model.
    '''

    # 1. Setup & Reproducibility
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    print(f"  Encoder LR: {lr_encoder}")
    print(f"  Head LR: {lr_head}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Device: {DEVICE}")

    # 2. Prepare datasets and dataloaders (train, val, test)
    print("\nLoading tokenizer and creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    ds_train = SentimentDataset(train_csv, tokenizer, max_length)
    ds_val = SentimentDataset(val_csv, tokenizer, max_length)
    ds_test = SentimentDataset(test_csv, tokenizer, max_length)
    
    print(f"Dataset sizes: Train={len(ds_train)}, Val={len(ds_val)}, Test={len(ds_test)}")
    
    # Create dataloaders
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize the model
    print("\nInitializing model...")
    config = SentimentConfig(
        model_name=model_name,
        num_labels=3,
        head=head,
        dropout=dropout,
        pooling_strategy="cls"
    )
    
    model = SentimentClassifier(config).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    
    # Estimate FLOPs
    try:
        flops = estimate_flops(
            hidden_size=model.hidden_size,
            num_layers=12,  # Assuming BERT-base
            seq_len=max_length,
            batch_size=batch_size
        )
        print(f"Estimated FLOPs per batch: {flops:.2f} GFLOPs")
    except Exception as e:
        print(f"Could not estimate FLOPs: {e}")

    # 4. Set up optimizer and learning rate scheduler
    print("\nSetting up optimizer and scheduler...")
    
    # Different learning rates for encoder and head
    param_groups = [
        {"params": model.encoder.parameters(), "lr": lr_encoder},
        {"params": model.head.parameters(), "lr": lr_head},
        {"params": model.norm.parameters(), "lr": lr_head},
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Calculate total training steps
    total_steps = len(dl_train) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # 5. Run the training loop
    print("\nStarting training...")
    best_val = -1.0
    ckpt_dir = os.path.join(out_dir, "checkpoint") # DO NOT change the file name
    os.makedirs(ckpt_dir, exist_ok=True)
    tokenizer.save_pretrained(ckpt_dir)
    
    training_history = {
        "train_loss": [],
        "val_acc": [],
        "epochs": []
    }

    for epoch in range(1, epochs + 1):
        model.train()  
        running_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            # Move data to GPU/CPU
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Reset gradients 
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step and scheduler update
            optimizer.step()
            scheduler.step()

            # Update running loss
            running_loss += loss.item()
            num_batches += 1

            # Display
            avg_loss = running_loss / num_batches
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}", 
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )

        # Validation Phase
        print(f"\nEvaluating epoch {epoch}...")
        val_acc, _, _ = evaluate(model, dl_val)
        
        # Store training history
        training_history["train_loss"].append(avg_loss)
        training_history["val_acc"].append(val_acc)
        training_history["epochs"].append(epoch)
        
        print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val:
            best_val = val_acc
            print(f"New best validation accuracy: {val_acc:.4f}")
            model.save_pretrained(ckpt_dir)
            config.save_pretrained(ckpt_dir)
            
        # Early stopping check (if validation accuracy doesn't improve for 3 epochs)
        #if epoch >= 3 and val_acc < max(training_history["val_acc"][-3:-1]):
        #    print("Early stopping triggered")
        #    break

    # 6. Evaluation and save results and metrics
    print("\nLoading best model for final evaluation...")
    best_model = SentimentClassifier.from_pretrained(ckpt_dir).to(DEVICE)

    def eval_and_save(split, dl):
        acc, y, yhat = evaluate(best_model, dl)
        
        # Save confusion matrix and classification report
        cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
        cm_df = pd.DataFrame(cm, 
                           index=['Negative', 'Neutral', 'Positive'],
                           columns=['Negative', 'Neutral', 'Positive'])
        cm_df.to_csv(os.path.join(ckpt_dir, f"{split}_confusion_matrix.csv"))
        
        # Classification report
        rpt = classification_report(
            y, yhat, 
            digits=4, 
            labels=[0, 1, 2],
            target_names=['Negative', 'Neutral', 'Positive']
        )
        with open(os.path.join(ckpt_dir, f"{split}_classification_report.txt"), "w") as f:
            f.write(rpt)
            
        print(f"\n{split.upper()} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(rpt)
        
        return float(acc)

    train_acc = eval_and_save("train", dl_train)
    val_acc = eval_and_save("val", dl_val)
    test_acc = eval_and_save("test", dl_test)

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(os.path.join(ckpt_dir, "training_history.csv"), index=False)

    # Save Summary in json format
    summary = {
        "model_name": model_name,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val,
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "epochs_trained": len(training_history["epochs"]),
        "final_train_loss": float(training_history["train_loss"][-1]),
        "hyperparameters": {
            "batch_size": batch_size,
            "max_length": max_length,
            "lr_encoder": lr_encoder,
            "lr_head": lr_head,
            "dropout": dropout,
            "warmup_ratio": warmup_ratio
        }
    }
    
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(json.dumps(summary, indent=2))
    
    # 7. Inference and Visualization Phase
    print("\\n" + "="*50)
    print("INFERENCE AND VISUALIZATION")
    print("="*50)
    
    # Test model inference on sample sentences
    print("\\nTesting model inference on sample sentences...")
    test_model_inference(ckpt_dir, num_samples=5)
    
    # Generate visualization plots
    print("\\nGenerating visualization plots...")
    visualize_results(ckpt_dir)
    
    # Calculate and display average accuracy across all test samples
    print(f"\\nFinal Test Set Performance:")
    print(f"  • Average Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  • Total Test Samples: {len(ds_test)}")
    print(f"  • Correct Predictions: {int(test_acc * len(ds_test))}")
    print(f"  • Model Size: {trainable_params:,} trainable parameters")

    # Cleanup
    try:
        best_model.to("cpu")
        model.to("cpu")
    except Exception:
        pass
    
    del best_model, model, tokenizer, optimizer, scheduler, dl_train, dl_val, dl_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return summary


# Main

def main():
    parser = argparse.ArgumentParser()
    # file paths
    parser.add_argument("--train_csv", type=str, default="./dataset/train.csv")
    parser.add_argument("--test_csv", type=str, default="./dataset/test.csv")
    parser.add_argument("--out_dir", type=str, default="./saved_models/") # DO NOT change the file name

    # model / data
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)

    # architecture
    parser.add_argument("--head", type=str, choices=["mlp"], default="mlp")
    parser.add_argument("--dropout", type=float, default=0.1)

    # optimization
    parser.add_argument("--lr_encoder", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Setup
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load the full dataset
    if not os.path.exists(args.train_csv):
        # If train.csv doesn't exist, we need to split the original dataset
        dataset_path = "./dataset/dataset.csv"
        if os.path.exists(dataset_path):
            print(f"Loading dataset from {dataset_path}")
            full_df = pd.read_csv(dataset_path)
            
            # Shuffle the dataset with fixed seed for reproducibility
            full_df = full_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            
            print(f"Dataset loaded: shape {full_df.shape}")
            print("Label distribution:")
            print(full_df["label"].value_counts().sort_index())
            
            # Split into train (70%), val (15%), test (15%)
            train_df, temp_df = train_test_split(
                full_df, 
                test_size=0.3, 
                random_state=args.seed, 
                stratify=full_df["label"]
            )
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=0.5, 
                random_state=args.seed, 
                stratify=temp_df["label"]
            )
            
            # Create output directory and save splits
            os.makedirs(os.path.dirname(args.train_csv), exist_ok=True)
            
            # Save the splits
            train_split = args.train_csv
            val_split = args.train_csv.replace("train.csv", "val.csv")
            test_split = args.test_csv
            
            train_df.to_csv(train_split, index=False, encoding="utf-8")
            val_df.to_csv(val_split, index=False, encoding="utf-8")
            test_df.to_csv(test_split, index=False, encoding="utf-8")
            
            print(f"Data splits saved:")
            print(f"  Train: {train_split} ({len(train_df)} samples)")
            print(f"  Val: {val_split} ({len(val_df)} samples)")
            print(f"  Test: {test_split} ({len(test_df)} samples)")
        else:
            raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    
    # Start training with the split data
    val_csv = args.train_csv.replace("train.csv", "val.csv")
    
    train(
        model_name=args.model_name,
        train_csv=args.train_csv,
        val_csv=val_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dropout=args.dropout,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        warmup_ratio=args.warmup_ratio,
        head=args.head,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()

