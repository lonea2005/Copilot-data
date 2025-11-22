#!/usr/bin/env python3

"""
Complete Sentiment Analysis Demo
Showcases all implemented features: training, inference, and visualization
"""

import sys
sys.path.append('.')

import os
import json
from main import predict_sentiment, visualize_results

def comprehensive_demo():
    """
    Comprehensive demonstration of the sentiment analysis system
    """
    
    print("🎯 COMPREHENSIVE SENTIMENT ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # 1. Model Information
    model_path = "./saved_models/checkpoint"
    summary_path = "./saved_models/summary.json"
    
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("📊 TRAINED MODEL PERFORMANCE")
        print("-" * 30)
        print(f"Training Accuracy:   {summary.get('train_accuracy', 0):.1%}")
        print(f"Validation Accuracy: {summary.get('val_accuracy', 0):.1%}")  
        print(f"Test Accuracy:       {summary.get('test_accuracy', 0):.1%}")
        print(f"Total Parameters:    {summary.get('params_total', 0):,}")
        print(f"Epochs Trained:      {summary.get('epochs_trained', 0)}")
    
    # 2. Real-world Inference Examples
    print("\n🔍 REAL-WORLD SENTIMENT ANALYSIS")
    print("-" * 30)
    
    real_world_examples = [
        # Positive examples
        "Just got the new iPhone and I'm absolutely blown away! The camera quality is incredible.",
        "Amazing customer service! They went above and beyond to help me.",
        "Best vacation ever! The hotel was perfect and the staff was so friendly.",
        
        # Negative examples  
        "Worst experience ever. The product broke after one day and customer service was rude.",
        "Don't waste your money on this. Poor quality and doesn't work as advertised.",
        "Terrible service! Waited 2 hours and the food was cold when it arrived.",
        
        # Neutral examples
        "The product works as expected. Nothing special but does the job.",
        "It's an okay restaurant. Food is decent, prices are reasonable.",
        "The movie was fine. Not great but not terrible either."
    ]
    
    try:
        results = predict_sentiment(real_world_examples, model_path)
        
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        
        for i, (text, label, probs) in enumerate(zip(
            results["texts"], 
            results["labels"], 
            results["probabilities"]
        ), 1):
            # Determine confidence level
            max_prob = max(probs)
            confidence = "High" if max_prob > 0.8 else "Medium" if max_prob > 0.6 else "Low"
            
            # Emoji for each sentiment
            emoji = "😊" if label == "Positive" else "😞" if label == "Negative" else "😐"
            
            print(f"\n{emoji} Example {i}: {label} ({confidence} confidence)")
            print(f"   Text: {text[:70]}{'...' if len(text) > 70 else ''}")
            print(f"   Scores: Pos={probs[2]:.2f} | Neu={probs[1]:.2f} | Neg={probs[0]:.2f}")
            
            sentiment_counts[label] += 1
        
        print(f"\n📈 ANALYSIS SUMMARY")
        print(f"   Positive: {sentiment_counts['Positive']} samples")
        print(f"   Neutral:  {sentiment_counts['Neutral']} samples") 
        print(f"   Negative: {sentiment_counts['Negative']} samples")
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    # 3. Visualization Status
    print(f"\n📊 VISUALIZATION FILES")
    print("-" * 30)
    
    viz_files = [
        ("Training History", "./saved_models/training_history.png"),
        ("Confusion Matrices", "./saved_models/confusion_matrices.png"), 
        ("Performance Summary", "./saved_models/performance_summary.png")
    ]
    
    for desc, path in viz_files:
        status = "✅ Available" if os.path.exists(path) else "❌ Missing"
        print(f"{status}: {desc}")
    
    # 4. Interactive Demo Section
    print(f"\n🎮 INTERACTIVE TESTING")
    print("-" * 30)
    print("You can test the model with custom text using:")
    print("  from main import predict_sentiment")
    print("  results = predict_sentiment(['your text here'])")
    print("  print(results['labels'])")
    
    # 5. Project Structure Summary
    print(f"\n📁 PROJECT STRUCTURE")
    print("-" * 30)
    
    key_files = {
        "model.py": "Model architecture and configuration classes",
        "main.py": "Training pipeline, evaluation, and inference functions", 
        "README.md": "Complete documentation and usage instructions",
        "saved_models/": "Trained model weights and results",
        "dataset/": "Training, validation, and test data splits"
    }
    
    for file_path, description in key_files.items():
        status = "✅" if os.path.exists(file_path) else "❌"
        print(f"{status} {file_path:<15} - {description}")
    
    # 6. Performance Highlights
    if os.path.exists(summary_path):
        print(f"\n🏆 KEY ACHIEVEMENTS")
        print("-" * 30)
        test_acc = summary.get('test_accuracy', 0)
        train_acc = summary.get('train_accuracy', 0)
        
        print(f"• Achieved {test_acc:.1%} test accuracy on social media sentiment")
        print(f"• Trained {summary.get('params_total', 0):,} parameter BERT model")
        print(f"• Generalization gap: {abs(train_acc - test_acc):.1%} (train vs test)")
        print(f"• Complete pipeline: data → training → inference → visualization")
    
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("The sentiment analysis system is ready for use!")
    print("Check the saved_models/ directory for all outputs and visualizations.")

if __name__ == "__main__":
    comprehensive_demo()