#!/usr/bin/env python3
"""
Script to visualize training data from saved Q-Net models or JSON files.
"""

import sys
import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_training_data_from_model(model_file):
    """Load training data from a saved model file."""
    try:
        checkpoint = torch.load(model_file, weights_only=False)
        
        if 'episode_rewards' in checkpoint:
            return {
                'episode_numbers': checkpoint['episode_numbers'],
                'episode_rewards': checkpoint['episode_rewards'],
                'episode_frames': checkpoint['episode_frames'],
                'avg_rewards': checkpoint['avg_rewards'],
                'avg_frames': checkpoint['avg_frames'],
                'final_episode_count': checkpoint['episode_count'],
                'final_step_count': checkpoint['step_count'],
                'final_epsilon': checkpoint['epsilon'],
                'final_total_reward': checkpoint['total_reward']
            }
        else:
            print(f"⚠ No training data found in {model_file}")
            return None
            
    except Exception as e:
        print(f"✗ Failed to load model {model_file}: {e}")
        return None

def load_training_data_from_json(json_file):
    """Load training data from a JSON file."""
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Failed to load JSON {json_file}: {e}")
        return None

def create_visualization(data, output_file="training_visualization.png"):
    """Create training visualization from data."""
    if not data:
        print("No data to visualize")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    episode_numbers = data['episode_numbers']
    episode_rewards = data['episode_rewards']
    episode_frames = data['episode_frames']
    avg_rewards = data.get('avg_rewards', [])
    avg_frames = data.get('avg_frames', [])
    
    # Plot 1: Episode Rewards
    ax1.plot(episode_numbers, episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
    if avg_rewards:
        ax1.plot(episode_numbers, avg_rewards, 'r-', linewidth=2, label='Moving Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (Points)')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Frames Survived
    ax2.plot(episode_numbers, episode_frames, 'g-', alpha=0.3, label='Frames Survived')
    if avg_frames:
        ax2.plot(episode_numbers, avg_frames, 'orange', linewidth=2, label='Moving Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Frames Survived')
    ax2.set_title('Training Progress: Frames Survived per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward Distribution
    ax3.hist(episode_rewards, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Frames Distribution
    ax4.hist(episode_frames, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Frames Survived')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Frames Survived Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add summary statistics
    fig.suptitle(f'Q-Net Training Analysis\n'
                f'Episodes: {len(episode_numbers)}, '
                f'Avg Reward: {np.mean(episode_rewards):.2f}, '
                f'Avg Frames: {np.mean(episode_frames):.1f}', 
                fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {output_file}")
    
    # Print summary statistics
    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {len(episode_numbers)}")
    print(f"  Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Average Frames: {np.mean(episode_frames):.1f}")
    print(f"  Max Reward: {np.max(episode_rewards):.2f}")
    print(f"  Min Reward: {np.min(episode_rewards):.2f}")
    print(f"  Max Frames: {np.max(episode_frames):.1f}")
    print(f"  Min Frames: {np.min(episode_frames):.1f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Q-Net training data")
    parser.add_argument('input_file', help='Model file (.pth) or JSON file (.json)')
    parser.add_argument('-o', '--output', default='training_visualization.png',
                       help='Output image file (default: training_visualization.png)')
    
    args = parser.parse_args()
    
    # Determine file type and load data
    if args.input_file.endswith('.pth'):
        print(f"Loading training data from model: {args.input_file}")
        data = load_training_data_from_model(args.input_file)
    elif args.input_file.endswith('.json'):
        print(f"Loading training data from JSON: {args.input_file}")
        data = load_training_data_from_json(args.input_file)
    else:
        print("Error: Input file must be .pth or .json")
        return
    
    if data:
        create_visualization(data, args.output)
    else:
        print("Failed to load training data")

if __name__ == "__main__":
    main() 