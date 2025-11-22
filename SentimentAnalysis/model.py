"""
Model Architecture for Sentiment Analysis

This file contains:
1. Model configuration class (SentimentConfig)
2. Custom neural network blocks (CustomBlock)
3. Main sentiment classifier model (SentimentClassifier)
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import (
    AutoModel,
    PretrainedConfig,
    PreTrainedModel
)


# Model Architecture Components
class CustomBlock(nn.Module): 
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1): 
        """
        Initialize the layers and parameters of this block.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension  
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-layer perceptron with residual connection if dimensions match
        self.fc1 = nn.Linear(input_dim, input_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(input_dim * 2, output_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # Whether to use residual connection
        self.use_residual = (input_dim == output_dim)

    def forward(self, x): 
        """
        Define how data moves through the block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)

        Returns:
            The transformed output tensor.
        """
        # Store input for potential residual connection
        residual = x
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Forward pass through MLP
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Add residual connection if dimensions match
        if self.use_residual:
            x = x + residual
            
        return x


# Model Configuration
class SentimentConfig(PretrainedConfig):
    model_type = "sentiment_classifier"

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=3,
        head="mlp",
        dropout=0.1,
        hidden_size=None,
        pooling_strategy="cls",
        freeze_backbone=False,
        **kwargs,
    ):
        """
        Configuration for SentimentClassifier.
        
        Args:
            model_name: Pre-trained model backbone name
            num_labels: Number of output classes (3 for sentiment: negative, neutral, positive)
            head: Type of classification head ("mlp" or "linear")
            dropout: Dropout rate for regularization
            hidden_size: Hidden size (will be set from backbone model if None)
            pooling_strategy: Token pooling strategy ("cls", "mean", "max")
            freeze_backbone: Whether to freeze backbone parameters
        """
        super().__init__(**kwargs)
        
        # Save all hyperparameters
        self.model_name = model_name
        self.num_labels = num_labels
        self.head = head
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        self.freeze_backbone = freeze_backbone


# Main Sentiment Classification Model
class SentimentClassifier(PreTrainedModel):
    config_class = SentimentConfig

    def __init__(self, config: Optional[SentimentConfig] = None):
        super().__init__(config)
        
        # Initialize the model components using the configuration
        self.config = config
        
        # Load pre-trained transformer backbone
        self.encoder = AutoModel.from_pretrained(config.model_name)
        
        # Get hidden size from encoder config and update config if needed
        self.hidden_size = self.encoder.config.hidden_size
        if config.hidden_size is None:
            config.hidden_size = self.hidden_size
            
        # Freeze backbone if specified
        if config.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Store head type and pooling strategy
        self.head_type = config.head
        self.pooling_strategy = config.pooling_strategy
        
        # Define classifier head based on head type
        if self.head_type == "mlp":
            self.head = CustomBlock(
                input_dim=self.hidden_size,
                output_dim=config.num_labels,
                dropout=config.dropout
            )
        else:
            # Simple linear classifier as fallback
            self.head = nn.Linear(self.hidden_size, config.num_labels)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
    def _pool_features(self, last_hidden_state, attention_mask=None):
        """
        Pool the token representations to get sentence-level representation.
        
        Args:
            last_hidden_state: Token representations (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Pooled representation (batch_size, hidden_size)
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token representation (first token)
            return last_hidden_state[:, 0, :]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling over all tokens
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(last_hidden_state, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling over all tokens
            if attention_mask is not None:
                # Set padding tokens to large negative value before max pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                last_hidden_state = last_hidden_state + (mask_expanded - 1) * 1e9
            return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            # Default to CLS pooling
            return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass through the sentiment classifier.

        Args:
            input_ids: Tokenized input sequences (batch_size, seq_len)
            attention_mask: Masks for padding tokens (batch_size, seq_len)
            token_type_ids: Token type ids (batch_size, seq_len) - optional
            labels: Ground-truth labels (batch_size,) - optional, for training

        Returns:
            Dictionary with "logits" (and optionally "loss")
        """
        # Pass inputs through the encoder
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
            
        outputs = self.encoder(**encoder_kwargs)
        
        # Get token representations
        last_hidden_state = outputs.last_hidden_state
        
        # Pool to get sentence-level representation
        pooled_output = self._pool_features(last_hidden_state, attention_mask)
        
        # Apply layer normalization and dropout
        feat = self.dropout(self.norm(pooled_output))
        
        # Pass through classifier head
        logits = self.head(feat)
        
        # Prepare result dictionary
        result = {"logits": logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            result["loss"] = loss
            
        return result

    def predict(self, text_list, tokenizer, max_length=128, device="cuda"):
        """
        Predict sentiment for a list of texts.
        
        Args:
            text_list: List of text strings to predict
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
            device: Device to run inference on
            
        Returns:
            Dictionary with predictions, probabilities, and labels
        """
        self.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in text_list:
                # Tokenize text
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                
                # Forward pass
                outputs = self(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                
                # Get probabilities and predictions
                probs = torch.softmax(logits, dim=-1)
                pred = torch.argmax(logits, dim=-1)
                
                predictions.append(pred.cpu().item())
                probabilities.append(probs.cpu().numpy()[0])
        
        # Convert predictions to labels
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        labels = [label_map[pred] for pred in predictions]
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "labels": labels,
            "texts": text_list
        }