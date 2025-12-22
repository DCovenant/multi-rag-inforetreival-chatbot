#!/usr/bin/env python3
"""
MRL (Matryoshka Representation Learning) Fine-Tuning Script

This script fine-tunes an embedding model using your generated dataset with MRL,
allowing the model to work well at multiple embedding dimensions.

WHAT THIS DOES:
1. Loads your train/test datasets (anchor, positive, label)
2. Converts to SentenceTransformers format
3. Trains with MatryoshkaLoss (multi-dimensional)
4. Evaluates at each dimension during training
5. Saves the fine-tuned model

USAGE:
    python train_mrl_embeddings.py
    
    # With custom settings:
    python train_mrl_embeddings.py --epochs 3 --batch-size 32 --model all-MiniLM-L6-v2
"""

import torch
import argparse
from datetime import datetime
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import re

# Default configuration
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
MATRYOSHKA_DIMS = [768, 512, 256, 128, 64]  # Large to small
BATCH_SIZE = 32
EPOCHS = 3
WARMUP_STEPS = 500
EVAL_STEPS = 200

def load_training_data(train_file: str, test_file: str):
    """Load train and test datasets from JSON files."""
    print(f"\nLoading datasets...")
    print(f"  Train: {train_file}")
    print(f"  Test: {test_file}")
    
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    test_dataset = load_dataset("json", data_files=test_file, split="train")
    
    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset

def create_input_examples(dataset):
    """
    Convert dataset to SentenceTransformers InputExample format.
    
    InputExample expects:
    - texts: [anchor, positive]
    - label: similarity score (0.0 to 1.0)
    
    Your dataset has:
    - anchor: query/question
    - positive: relevant document
    - label: relevance score (0.0 to 1.0)
    """
    examples = []
    
    for item in dataset:
        anchor = item.get('anchor', '')
        positive = item.get('positive', '')
        label = item.get('label', 1.0)  # Default to 1.0 if missing
        
        # Skip empty examples
        if not anchor or not positive:
            continue
        
        # Create InputExample
        # texts=[anchor, positive] means we're learning that these should be similar
        # label is how similar they should be (0.0 = not similar, 1.0 = very similar)
        examples.append(InputExample(
            texts=[anchor, positive],
            label=float(label)
        ))
    
    print(f"✓ Created {len(examples)} InputExample objects")
    return examples

def create_evaluator(test_dataset, name="test"):
    """
    Create an evaluator for information retrieval metrics.
    
    This evaluator:
    1. Embeds all queries and documents
    2. Finds top-K most similar docs for each query
    3. Calculates metrics (NDCG@10, MAP@10, etc.)
    4. Does this at each Matryoshka dimension
    """
    # Create corpus and queries
    corpus = dict(
        zip(test_dataset["id"], test_dataset["positive"])
    )
    queries = dict(
        zip(test_dataset["id"], test_dataset["anchor"])
    )
    
    # Map each query to its relevant document(s)
    relevant_docs = {}
    for q_id in queries:
        relevant_docs[q_id] = [q_id]  # Each query's ID matches its relevant doc's ID
    
    # Create evaluators for each dimension
    evaluators = []
    for dim in MATRYOSHKA_DIMS:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"{name}_dim_{dim}",
            truncate_dim=dim,  # Truncate embeddings to this dimension
            score_functions={"cosine": cos_sim},
            show_progress_bar=True
        )
        evaluators.append(ir_evaluator)
    
    # Combine into sequential evaluator
    return SequentialEvaluator(evaluators)

def train_model(model_name: str, train_examples, evaluator, 
                output_dir: str, batch_size: int, epochs: int):
    """
    Fine-tune the model with MRL.
    
    Training Process:
    1. Create DataLoader with training examples
    2. Define loss function (MatryoshkaLoss + CosineSimilarityLoss)
    3. Train with model.fit()
    4. Evaluate during training
    5. Save final model
    6. Return training history with loss values
    """
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Matryoshka dimensions: {MATRYOSHKA_DIMS}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size
    )
    
    # Define loss function
    # CosineSimilarityLoss: Learns to predict similarity scores
    # - Good for your data because you have continuous labels (0.0 to 1.0)
    # - Learns that label=1.0 means cosine_sim should be high
    # - Learns that label=0.0 means cosine_sim should be low
    inner_loss = losses.CosineSimilarityLoss(model)
    
    # Wrap with MatryoshkaLoss for multi-dimensional training
    # This trains the model to work well at ALL dimensions simultaneously
    train_loss = losses.MatryoshkaLoss(
        model,
        inner_loss,
        matryoshka_dims=MATRYOSHKA_DIMS
    )
    
    print(f"✓ Loss function: MatryoshkaLoss(CosineSimilarityLoss)")
    print(f"✓ Training on dimensions: {MATRYOSHKA_DIMS}")
    
    # Calculate training steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(WARMUP_STEPS, total_steps // 10)
    
    print(f"\nTraining details:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Evaluation every: {EVAL_STEPS} steps")
    
    # Train the model
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    # Prepare to capture loss values during training
    loss_history = []
    step_counter = [0]  # Use list to allow modification in nested function
    
    # Monkey-patch the training loop to capture loss values
    # We'll wrap the loss computation to log values
    original_loss_forward = train_loss.forward
    
    def loss_forward_with_logging(sentence_features, labels):
        """Wrapper around loss forward to capture loss values."""
        loss_value = original_loss_forward(sentence_features, labels)
        
        # Increment step counter each time loss is computed (once per batch)
        step_counter[0] += 1
        step = step_counter[0]
        epoch = step / steps_per_epoch
        
        # Record loss every 10 steps
        if step % 10 == 0:
            loss_history.append({
                'step': step,
                'epoch': epoch,
                'loss': loss_value.item()
            })
        
        return loss_value
    
    # Apply the wrapper
    train_loss.forward = loss_forward_with_logging
    
    # Train with standard fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=EVAL_STEPS,
        warmup_steps=warmup_steps,
        output_path=output_dir,
        save_best_model=True,
        show_progress_bar=True,
        checkpoint_save_steps=EVAL_STEPS,
    )
    
    # Restore original loss forward
    train_loss.forward = original_loss_forward
    
    # Save loss history to JSON file
    loss_file = Path(output_dir) / "loss_history.json"
    if loss_history:
        with open(loss_file, 'w') as f:
            json.dump(loss_history, f, indent=2)
        print(f"✓ Loss history saved to: {loss_file} ({len(loss_history)} entries)")
    else:
        print("Warning: No loss values were captured during training.")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Model saved to: {output_dir}")
    
    return model

def evaluate_final_model(model_path: str, test_dataset):
    """Evaluate the final trained model at all dimensions."""
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}\n")
    
    # Load trained model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_path, device=device)
    
    # Create evaluator
    evaluator = create_evaluator(test_dataset, name="final")
    
    # Evaluate
    print("Running final evaluation on test set...")
    results = evaluator(model)
    
    # Print results
    print(f"\n{'='*70}")
    print("FINAL RESULTS (NDCG@10)")
    print(f"{'='*70}")
    for dim in MATRYOSHKA_DIMS:
        key = f"final_dim_{dim}_cosine_ndcg@10"
        score = results.get(key, 0.0)
        print(f"  dim_{dim}: {score:.4f}")
    print(f"{'='*70}\n")
    
    return results


def plot_training_history(output_dir: str):
    """
    Plot training metrics from CSV logs created during training.
    
    This reads the evaluation results and loss values saved by SentenceTransformer
    during training and creates visualizations to help diagnose if more training is needed.
    """
    import csv
    
    output_path = Path(output_dir)
    
    # Check for the actual files generated (one per dimension)
    eval_dir = output_path / "eval"
    
    # Read CSV with evaluation results (one file per dimension)
    steps = []
    scores_by_dim = {dim: [] for dim in MATRYOSHKA_DIMS}
    
    if eval_dir.exists():
        # Read each dimension's CSV file
        for dim in MATRYOSHKA_DIMS:
            csv_path = eval_dir / f"Information-Retrieval_evaluation_dev_dim_{dim}_results.csv"
            
            if not csv_path.exists():
                continue
            
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    dim_steps = []
                    dim_scores = []
                    
                    for row in reader:
                        step = int(row.get('steps', 0))
                        dim_steps.append(step)
                        
                        # Extract NDCG@10 score
                        score = float(row.get('cosine-NDCG@10', 0))
                        dim_scores.append(score)
                    
                    # Store for this dimension
                    scores_by_dim[dim] = dim_scores
                    
                    # Use the steps from the first dimension
                    if not steps and dim_steps:
                        steps = dim_steps
            except Exception as e:
                print(f"Warning: Could not read {csv_path.name}: {e}")
    
    # Read loss history from JSON file
    loss_steps = []
    loss_values = []
    loss_epochs = []
    
    loss_file = output_path / "loss_history.json"
    if loss_file.exists():
        try:
            with open(loss_file, 'r') as f:
                loss_data = json.load(f)
                for entry in loss_data:
                    loss_steps.append(entry['step'])
                    loss_values.append(entry['loss'])
                    loss_epochs.append(entry['epoch'])
            print(f"✓ Loaded {len(loss_values)} loss values from {loss_file.name}")
        except Exception as e:
            print(f"Warning: Could not read loss history: {e}")
    
    # Determine subplot layout
    has_loss = len(loss_values) > 0
    has_eval = len(steps) > 0
    
    if not has_loss and not has_eval:
        print("No training data found to plot.")
        return
    
    # Create figure with appropriate number of subplots
    if has_loss and has_eval:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_loss_step = axes[0, 0]
        ax_loss_epoch = axes[0, 1]
        ax_ndcg = axes[1, 0]
        ax_bars = axes[1, 1]
    elif has_loss:
        fig, (ax_loss_step, ax_loss_epoch) = plt.subplots(1, 2, figsize=(14, 5))
        ax_ndcg = None
        ax_bars = None
    else:
        fig, (ax_ndcg, ax_bars) = plt.subplots(1, 2, figsize=(14, 5))
        ax_loss_step = None
        ax_loss_epoch = None
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Loss vs Training Steps
    if ax_loss_step is not None and has_loss:
        ax_loss_step.set_title('Training Loss vs Steps', fontsize=14, fontweight='bold')
        ax_loss_step.set_xlabel('Training Steps')
        ax_loss_step.set_ylabel('Loss')
        ax_loss_step.grid(True, alpha=0.3)
        
        ax_loss_step.plot(loss_steps, loss_values, color='#e74c3c', 
                         linewidth=1.5, alpha=0.7, label='Training Loss')
        
        # Add smoothed trend line
        if len(loss_values) > 20:
            window = max(5, len(loss_values) // 20)
            smoothed = np.convolve(loss_values, np.ones(window)/window, mode='valid')
            smoothed_steps = loss_steps[window-1:]
            ax_loss_step.plot(smoothed_steps, smoothed, color='#c0392b', 
                             linewidth=2.5, label=f'Smoothed (window={window})')
        
        ax_loss_step.legend(loc='upper right')
        
        # Annotate start and end loss
        ax_loss_step.annotate(f'Start: {loss_values[0]:.4f}', 
                             xy=(loss_steps[0], loss_values[0]),
                             xytext=(10, -20), textcoords='offset points',
                             fontweight='bold', fontsize=10, color='#2c3e50',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        ax_loss_step.annotate(f'End: {loss_values[-1]:.4f}', 
                             xy=(loss_steps[-1], loss_values[-1]),
                             xytext=(-80, 20), textcoords='offset points',
                             fontweight='bold', fontsize=10, color='#2c3e50',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Plot 2: Loss vs Epoch
    if ax_loss_epoch is not None and has_loss:
        ax_loss_epoch.set_title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
        ax_loss_epoch.set_xlabel('Epoch')
        ax_loss_epoch.set_ylabel('Loss')
        ax_loss_epoch.grid(True, alpha=0.3)
        
        ax_loss_epoch.plot(loss_epochs, loss_values, color='#3498db', 
                          linewidth=1.5, alpha=0.7, label='Training Loss')
        
        # Add smoothed trend line
        if len(loss_values) > 20:
            window = max(5, len(loss_values) // 20)
            smoothed = np.convolve(loss_values, np.ones(window)/window, mode='valid')
            smoothed_epochs = loss_epochs[window-1:]
            ax_loss_epoch.plot(smoothed_epochs, smoothed, color='#2980b9', 
                              linewidth=2.5, label=f'Smoothed (window={window})')
        
        ax_loss_epoch.legend(loc='upper right')
        
        # Mark epoch boundaries
        max_epoch = max(loss_epochs) if loss_epochs else 0
        for epoch in range(1, int(max_epoch) + 1):
            ax_loss_epoch.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
    
    # Plot 3: NDCG@10 over training steps for all dimensions
    if ax_ndcg is not None and has_eval:
        ax_ndcg.set_title('Validation NDCG@10 During Training', fontsize=14, fontweight='bold')
        ax_ndcg.set_xlabel('Training Steps')
        ax_ndcg.set_ylabel('NDCG@10')
        ax_ndcg.grid(True, alpha=0.3)
        
        for i, dim in enumerate(MATRYOSHKA_DIMS):
            if scores_by_dim[dim]:
                ax_ndcg.plot(steps[:len(scores_by_dim[dim])], scores_by_dim[dim], 
                           marker='o', label=f'dim_{dim}', 
                           color=colors[i], linewidth=2, markersize=4)
        
        ax_ndcg.legend(loc='best')
    
    # Plot 4: Final dimension comparison (bar chart)
    if ax_bars is not None and has_eval:
        ax_bars.set_title('Final NDCG@10 by Dimension', fontsize=14, fontweight='bold')
        ax_bars.set_xlabel('Embedding Dimension')
        ax_bars.set_ylabel('NDCG@10')
        ax_bars.grid(True, alpha=0.3, axis='y')
        
        final_scores = [scores_by_dim[dim][-1] if scores_by_dim[dim] else 0 
                       for dim in MATRYOSHKA_DIMS]
        bars = ax_bars.bar([str(d) for d in MATRYOSHKA_DIMS], final_scores, 
                          color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, score in zip(bars, final_scores):
            height = bar.get_height()
            ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.4f}',
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / "training_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to: {plot_path}")
    
    # Show plot
    plt.show()
    
    # Analyze if more training is needed
    print(f"\n{'='*70}")
    print("TRAINING ANALYSIS")
    print(f"{'='*70}")
    
    # Analyze loss trend if available
    if has_loss and len(loss_values) >= 2:
        print("\n📉 LOSS ANALYSIS:")
        initial_loss = loss_values[0]
        final_loss = loss_values[-1]
        min_loss = min(loss_values)
        loss_reduction = initial_loss - final_loss
        loss_reduction_pct = (loss_reduction / initial_loss) * 100 if initial_loss > 0 else 0
        
        print(f"  Initial loss: {initial_loss:.6f}")
        print(f"  Final loss:   {final_loss:.6f}")
        print(f"  Minimum loss: {min_loss:.6f}")
        print(f"  Total reduction: {loss_reduction:.6f} ({loss_reduction_pct:.2f}%)")
        
        # Check if loss is still decreasing (compare last 25% vs previous 25%)
        if len(loss_values) >= 20:
            quarter = len(loss_values) // 4
            recent_avg = np.mean(loss_values[-quarter:])
            previous_avg = np.mean(loss_values[-2*quarter:-quarter])
            trend_change = recent_avg - previous_avg
            
            if trend_change < -0.005:
                print(f"  Trend: 📉 STILL DECREASING (recent change: {trend_change:.6f})")
                print(f"  → MORE EPOCHS RECOMMENDED - loss is still improving")
            elif trend_change > 0.005:
                print(f"  Trend: 📈 INCREASING/OVERFITTING (recent change: {trend_change:.6f})")
                print(f"  → STOP TRAINING - model may be overfitting")
            else:
                print(f"  Trend: 📊 PLATEAUED (recent change: {trend_change:.6f})")
                print(f"  → Training has converged, more epochs unlikely to help significantly")
    
    if has_eval:
        print("\n📊 NDCG@10 ANALYSIS:")
        # Check if metrics are still improving
        for dim in MATRYOSHKA_DIMS:
            if len(scores_by_dim[dim]) >= 2:
                improvement = scores_by_dim[dim][-1] - scores_by_dim[dim][0]
                trend = "📈 IMPROVED" if improvement > 0.001 else "📊 UNCHANGED" if abs(improvement) < 0.001 else "📉 DECREASED"
                print(f"  dim_{dim}: {trend} (total change: {improvement:+.4f})")
    
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model with MRL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help=f"Base model to fine-tune (default: {DEFAULT_MODEL})")
    parser.add_argument("--train-file", type=str, default="dataset/train_dataset.json",
                       help="Path to training dataset JSON")
    parser.add_argument("--test-file", type=str, default="dataset/test_dataset.json",
                       help="Path to test dataset JSON")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for trained model (auto-generated if not specified)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                       help=f"Training batch size (default: {BATCH_SIZE})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--eval-steps", type=int, default=EVAL_STEPS,
                       help=f"Evaluate every N steps (default: {EVAL_STEPS})")
    parser.add_argument("--skip-final-eval", action="store_true",
                       help="Skip final evaluation after training")
    
    args = parser.parse_args()
    
    # Generate output directory name if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = args.model.split('/')[-1]
        args.output_dir = f"models/{model_name}_mrl_{timestamp}"
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("MRL EMBEDDING MODEL FINE-TUNING")
    print(f"{'='*70}")
    
    # Load datasets
    train_dataset, test_dataset = load_training_data(args.train_file, args.test_file)
    
    # Convert to InputExample format
    print("\nPreparing training data...")
    train_examples = create_input_examples(train_dataset)
    
    if len(train_examples) == 0:
        print("ERROR: No valid training examples found!")
        print("Make sure your dataset has 'anchor' and 'positive' columns.")
        return
    
    # Create evaluator for test set
    print("\nPreparing evaluator...")
    evaluator = create_evaluator(test_dataset, name="dev")
    
    # Train model
    model = train_model(
        args.model,
        train_examples,
        evaluator,
        args.output_dir,
        args.batch_size,
        args.epochs
    )
    
    # Final evaluation
    if not args.skip_final_eval:
        evaluate_final_model(args.output_dir, test_dataset)
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(args.output_dir)
    
    print(f"\n✓ All done!")
    print(f"✓ Fine-tuned model saved to: {args.output_dir}")
    print(f"\nTo use your model:")
    print(f"  from sentence_transformers import SentenceTransformer")
    print(f"  model = SentenceTransformer('{args.output_dir}')")
    print(f"  embeddings = model.encode(['your text here'], truncate_dim=256)")

if __name__ == "__main__":
    main()
