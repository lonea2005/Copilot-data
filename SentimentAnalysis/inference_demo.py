#!/usr/bin/env python3

"""
Inference and Visualization Demo
Demonstrates model inference and generates visualization plots
"""

import sys
sys.path.append('.')

from main import (
    predict_sentiment, 
    test_model_inference, 
    visualize_results,
    load_trained_model
)
import os

def run_inference_demo():
    """
    Comprehensive inference and visualization demonstration
    """
    
    print("="*60)
    print("SENTIMENT ANALYSIS - INFERENCE & VISUALIZATION DEMO")
    print("="*60)
    
    model_path = "./saved_models/checkpoint"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run training first using: python run_training.py")
        return
    
    print("✅ Model found! Starting inference demo...\n")
    
    # 1. Test Model Inference
    print("1. TESTING MODEL INFERENCE")
    print("-" * 40)
    test_model_inference(model_path, num_samples=8)
    
    # 2. Custom Text Predictions
    print("\n2. CUSTOM TEXT PREDICTIONS")
    print("-" * 40)
    
    custom_texts = [
        "I absolutely love this new smartphone! The camera quality is incredible.",
        "The service was terrible and the food was cold. Won't be coming back.",
        "It's an okay product. Nothing extraordinary but gets the job done.",
        "Best purchase ever! Highly recommend to everyone.",
        "Waste of money. Poor quality and bad customer service.",
        "Pretty decent for the price. Could be better but not bad.",
        "Amazing experience! The staff was friendly and helpful.",
        "Not worth it. There are better alternatives available."
    ]
    
    print("Analyzing custom text samples...")
    results = predict_sentiment(custom_texts, model_path)
    
    for i, (text, label, probs) in enumerate(zip(
        results["texts"], 
        results["labels"], 
        results["probabilities"]
    )):
        print(f"\n📝 Text {i+1}: {text}")
        print(f"🎯 Prediction: {label}")
        print(f"📊 Confidence: Neg={probs[0]:.3f} | Neu={probs[1]:.3f} | Pos={probs[2]:.3f}")
    
    # 3. Generate Visualizations
    print("\n" + "="*60)
    print("3. GENERATING VISUALIZATION PLOTS")
    print("-" * 40)
    
    visualize_results(model_path)
    
    # 4. Model Summary
    print("\n" + "="*60)
    print("4. MODEL SUMMARY")
    print("-" * 40)
    
    try:
        import json
        summary_path = "./saved_models/summary.json"
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            print(f"📊 Model Performance:")
            print(f"   • Training Accuracy:   {summary.get('train_accuracy', 0):.4f}")
            print(f"   • Validation Accuracy: {summary.get('val_accuracy', 0):.4f}")
            print(f"   • Test Accuracy:       {summary.get('test_accuracy', 0):.4f}")
            print(f"\n🏗️  Model Architecture:")
            print(f"   • Total Parameters:     {summary.get('params_total', 0):,}")
            print(f"   • Trainable Parameters: {summary.get('params_trainable', 0):,}")
            print(f"   • Epochs Trained:       {summary.get('epochs_trained', 0)}")
            
            if 'hyperparameters' in summary:
                hypers = summary['hyperparameters']
                print(f"\n⚙️  Hyperparameters:")
                for key, value in hypers.items():
                    print(f"   • {key.replace('_', ' ').title()}: {value}")
        
    except Exception as e:
        print(f"Could not load model summary: {e}")
    
    # 5. Generated Files Summary
    print("\n" + "="*60)
    print("5. GENERATED FILES")
    print("-" * 40)
    
    files_to_check = [
        ("Model Checkpoint", "./saved_models/checkpoint/model.safetensors"),
        ("Model Config", "./saved_models/checkpoint/config.json"),
        ("Training History", "./saved_models/checkpoint/training_history.csv"),
        ("Summary", "./saved_models/summary.json"),
        ("Training Plots", "./saved_models/training_history.png"),
        ("Confusion Matrices", "./saved_models/confusion_matrices.png"),
        ("Performance Summary", "./saved_models/performance_summary.png")
    ]
    
    for desc, path in files_to_check:
        status = "✅" if os.path.exists(path) else "❌"
        print(f"{status} {desc}: {path}")
    
    print("\n" + "="*60)
    print("🎉 INFERENCE & VISUALIZATION DEMO COMPLETE!")
    print("="*60)
    print("\nGenerated visualizations can be found in the saved_models/ directory.")
    print("Open the PNG files to view training curves, confusion matrices, and performance summaries.")

if __name__ == "__main__":
    run_inference_demo()