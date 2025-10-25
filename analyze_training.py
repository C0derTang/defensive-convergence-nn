import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_training_data(filename="training_data.json"):
    """Analyze the training data to understand win/loss patterns."""
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Extract data
    episode_numbers = data['episode_numbers']
    scores = data['scores']
    winners = data['winners']
    frames_elapsed = data['frames_elapsed']
    
    print(f"Total episodes: {len(episode_numbers)}")
    print(f"Episodes with data: {len(scores)}")
    
    # Analyze winners
    winner_counts = Counter(winners)
    print(f"\nWinner distribution:")
    for winner, count in winner_counts.items():
        percentage = (count / len(winners)) * 100
        print(f"  {winner}: {count} ({percentage:.1f}%)")
    
    # Analyze scores
    score_counts = Counter(scores)
    print(f"\nScore distribution:")
    for score, count in score_counts.most_common():
        percentage = (count / len(scores)) * 100
        print(f"  {score}: {count} ({percentage:.1f}%)")
    
    # Analyze win streaks
    print(f"\nAnalyzing win streaks...")
    current_streak = 0
    max_streak = 0
    streaks = []
    
    for i, winner in enumerate(winners):
        if winner == "Blue":
            current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                max_streak = max(max_streak, current_streak)
            current_streak = 0
    
    if current_streak > 0:
        streaks.append(current_streak)
        max_streak = max(max_streak, current_streak)
    
    print(f"  Max Blue win streak: {max_streak}")
    print(f"  Average Blue win streak: {np.mean(streaks):.1f}")
    print(f"  Total Blue win streaks: {len(streaks)}")
    
    # Analyze episode lengths
    print(f"\nEpisode length analysis:")
    print(f"  Average frames per episode: {np.mean(frames_elapsed):.1f}")
    print(f"  Min frames: {min(frames_elapsed)}")
    print(f"  Max frames: {max(frames_elapsed)}")
    
    # Find episodes where Red wins
    red_wins = []
    for i, winner in enumerate(winners):
        if winner == "Red":
            red_wins.append(i)
    
    print(f"\nRed wins occurred at episodes: {red_wins}")
    
    # Analyze patterns around Red wins
    if red_wins:
        print(f"\nAnalyzing patterns around Red wins:")
        for win_episode in red_wins[:10]:  # Look at first 10 Red wins
            start = max(0, win_episode - 5)
            end = min(len(winners), win_episode + 6)
            context = winners[start:end]
            print(f"  Episode {win_episode}: {context}")
    
    # Create visualization
    create_analysis_plots(episode_numbers, winners, scores, frames_elapsed)

def create_analysis_plots(episodes, winners, scores, frames):
    """Create plots to visualize the training patterns."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Win/Loss over time
    red_wins = [i for i, w in enumerate(winners) if w == "Red"]
    blue_wins = [i for i, w in enumerate(winners) if w == "Blue"]
    
    ax1.scatter(red_wins, [1] * len(red_wins), c='red', alpha=0.6, label='Red Wins')
    ax1.scatter(blue_wins, [0] * len(blue_wins), c='blue', alpha=0.6, label='Blue Wins')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Winner (0=Blue, 1=Red)')
    ax1.set_title('Win/Loss Pattern Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Episode lengths
    ax2.plot(episodes, frames, 'g-', alpha=0.6)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Frames Elapsed')
    ax2.set_title('Episode Length Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Moving average win rate
    window = 50
    red_win_rates = []
    for i in range(window, len(winners)):
        recent_winners = winners[i-window:i]
        red_wins = sum(1 for w in recent_winners if w == "Red")
        red_win_rates.append(red_wins / window)
    
    ax3.plot(episodes[window:], red_win_rates, 'r-', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Red Win Rate (50-episode moving average)')
    ax3.set_title('Red Win Rate Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Score distribution
    score_counts = Counter(scores)
    ax4.bar(score_counts.keys(), score_counts.values(), alpha=0.7)
    ax4.set_xlabel('Score (Red-Blue)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis plots saved to 'training_analysis.png'")

if __name__ == "__main__":
    analyze_training_data() 