import pygame
import sys
import argparse
import time
import os
from grid import Grid
from qnet_agent import QNetAgent
from agent import AIAgent
from constants import *

class TrainingEnvironment:
    """Training environment for Q-Net agent without rendering."""
    
    def __init__(self, target_fps=60, episodes=1000, model_file=None):
        pygame.init()
        self.target_fps = target_fps
        self.episodes = episodes
        self.model_file = model_file
        
        # Initialize game grid
        self.grid = Grid()
        
        # Initialize agents
        self.red_agent = QNetAgent(
            self.grid, TEAM_RED, 7, 3, self.target_fps,
            learning_rate=0.001,  # Higher learning rate for faster learning
            epsilon=1.0,  # Start with full exploration
            epsilon_decay=0.9990,  # Decay to 0.1 by episode 1000
            epsilon_min=0.1,  # Higher minimum exploration
            gamma=0.99,  # Higher discount factor for long-term planning
            memory_size=10000,  # Reasonable memory size
            batch_size=32,  # Standard batch size
            train_frequency=16  # Train every 4 steps
        )
        self.blue_agent = AIAgent(self.grid, TEAM_BLUE, 7, 16, self.target_fps, self.red_agent)
        self.red_agent.enemy_agent = self.blue_agent
        self.agents = [self.red_agent, self.blue_agent]
        
        # Auto-load model if specified or if default model exists
        if model_file:
            self.load_model(model_file)
        else:
            self.auto_load_model()
        
        # Game state
        self.game_over = False
        self.winner = None
        self.frame_count = 0
        
        # Scoreboard
        self.red_score = 0
        self.blue_score = 0
        self.captures_to_win = CAPTURES_TO_WIN
        
        # Training statistics
        self.episode_count = 0
        self.red_wins = 0
        self.blue_wins = 0
        self.avg_episode_length = 0
        self.total_frames = 0
        
        # Training data tracking for JSON export
        self.training_data = {
            'episode_numbers': [],
            'frames_elapsed': [],
            'scores': [],
            'winners': [],
            'episode_times': [],
            'red_win_rates': [],
            'blue_win_rates': [],
            'epsilons': []
        }
        
        # Performance tracking
        self.start_time = time.time()
        self.episode_start_time = time.time()
    
    def auto_load_model(self):
        """Automatically load the most recent model if available."""
        # Priority order for model files
        model_files = [
            "qnet_model_final.pth",
            "qnet_model.pth"
        ]
        
        # Check for episode models (get the highest episode number)
        import glob
        episode_models = glob.glob("qnet_model_episode_*.pth")
        if episode_models:
            # Extract episode numbers and find the highest
            episode_numbers = []
            for model in episode_models:
                try:
                    episode_num = int(model.split("_")[-1].split(".")[0])
                    episode_numbers.append((episode_num, model))
                except (ValueError, IndexError):
                    continue
            
            if episode_numbers:
                # Sort by episode number and get the highest
                episode_numbers.sort(reverse=True)
                model_files.insert(0, episode_numbers[0][1])
        
        # Try to load the first available model
        for model_file in model_files:
            if os.path.exists(model_file):
                self.load_model(model_file)
                return
        
        print("No existing model found. Starting training from scratch.")
    
    def load_model(self, model_file):
        """Load a specific model file."""
        try:
            self.red_agent.load_model(model_file)
            print(f"✓ Loaded model: {model_file}")
            print(f"  - Episode count: {self.red_agent.episode_count}")
            print(f"  - Total steps: {self.red_agent.step_count}")
            print(f"  - Current epsilon: {self.red_agent.epsilon:.3f}")
            print(f"  - Average reward: {self.red_agent.total_reward/max(1, self.red_agent.episode_count):.2f}")
            
            # Update training environment statistics to match loaded model
            self.episode_count = self.red_agent.episode_count
            self.total_frames = sum(self.red_agent.episode_frames) if hasattr(self.red_agent, 'episode_frames') and self.red_agent.episode_frames else 0
            
            # Load historical training data if available
            if hasattr(self.red_agent, 'episode_numbers') and self.red_agent.episode_numbers:
                print(f"✓ Loading historical training data from model...")
                # Populate training data arrays with historical data
                for i, episode_num in enumerate(self.red_agent.episode_numbers):
                    if i < len(self.red_agent.episode_rewards):
                        # Calculate win rates based on historical data
                        # This is a simplified approach - in practice, you might want to store win/loss data in the model
                        red_win_rate = 50.0  # Default assumption
                        blue_win_rate = 50.0
                        
                        # Convert any numpy types to Python types
                        episode_num = int(episode_num) if hasattr(episode_num, 'dtype') else int(episode_num)
                        frames = self.red_agent.episode_frames[i] if i < len(self.red_agent.episode_frames) else 0
                        frames = int(frames) if hasattr(frames, 'dtype') else int(frames)
                        
                        self.training_data['episode_numbers'].append(episode_num)
                        self.training_data['frames_elapsed'].append(frames)
                        self.training_data['scores'].append("0-0")  # Default score
                        self.training_data['winners'].append("Unknown")  # Default winner
                        self.training_data['episode_times'].append(0.0)  # Default time
                        self.training_data['red_win_rates'].append(float(red_win_rate))
                        self.training_data['blue_win_rates'].append(float(blue_win_rate))
                        self.training_data['epsilons'].append(float(self.red_agent.epsilon))  # Use current epsilon
                
                print(f"  - Loaded {len(self.red_agent.episode_numbers)} historical episodes")
        except Exception as e:
            print(f"✗ Failed to load model {model_file}: {e}")
            print("Starting training from scratch.")
    
    def reset_game(self):
        """Reset the game to initial state."""
        # Reset only flags and bases without regenerating the map layout
        self.grid.reset_flags_and_bases()
        
        # Reset agent positions and game state WITHOUT recreating the agents
        self.red_agent.respawn()  # Reset to starting position
        self.blue_agent.respawn()  # Reset to starting position
        
        # Reset episode-specific variables
        self.red_agent.episode_reward = 0
        self.red_agent.episode_frame_count = 0
        self.blue_agent.episode_reward = 0
        self.blue_agent.episode_frame_count = 0
        
        self.frame_count = 0
        
        # Reset game state
        self.game_over = False
        self.winner = None
        
        # Reset scoreboard
        self.red_score = 0
        self.blue_score = 0
    
    def update(self):
        """Update game state."""
        self.frame_count += 1
        
        if self.game_over:
            return
        
        # Update agents and check for flag captures
        for agent in self.agents:
            capture_result = agent.update(self.frame_count)
            if capture_result == "red_capture":
                self.red_score += 1
            elif capture_result == "blue_capture":
                self.blue_score += 1
        
        # Check for bullet collisions with agents
        if any(agent.bullets for agent in self.agents):
            self.check_bullet_collisions()
        
        # Check for win condition
        self.check_win_condition()
    
    def check_win_condition(self):
        """Check if either team has won."""
        if self.red_score >= self.captures_to_win:
            self.game_over = True
            self.winner = "Red"
        elif self.blue_score >= self.captures_to_win:
            self.game_over = True
            self.winner = "Blue"
    
    def check_bullet_collisions(self):
        """Check for collisions between bullets and agents."""
        alive_agents = [agent for agent in self.agents if agent.alive]
        if not alive_agents:
            return
            
        for agent in alive_agents:
            agent_rect = agent.get_rect()
            
            for other_agent in self.agents:
                if other_agent == agent or not other_agent.bullets:
                    continue
                    
                bullets_to_remove = []
                for bullet in other_agent.bullets:
                    dx = bullet.x - agent.x
                    dy = bullet.y - agent.y
                    distance_squared = dx*dx + dy*dy
                    collision_distance = agent.radius + bullet.radius
                    
                    if distance_squared <= collision_distance * collision_distance:
                        agent.take_damage(bullet.damage)
                        bullets_to_remove.append(bullet)
                
                for bullet in bullets_to_remove:
                    if bullet in other_agent.bullets:
                        other_agent.bullets.remove(bullet)
    
    def run_episode(self):
        """Run a single training episode."""
        self.reset_game()
        self.episode_start_time = time.time()
        
        # Run episode until game over
        while not self.game_over:
            self.update()
        
        # End episode for Q-Net agent
        self.red_agent.end_episode()
        
        # Update statistics
        self.episode_count += 1
        self.total_frames += self.frame_count
        
        if self.winner == "Red":
            self.red_wins += 1
        elif self.winner == "Blue":
            self.blue_wins += 1
        
        # Calculate average episode length
        self.avg_episode_length = self.total_frames / self.episode_count
        
        # Calculate episode statistics
        episode_time = time.time() - self.episode_start_time
        red_win_rate = (self.red_wins / self.episode_count) * 100
        blue_win_rate = (self.blue_wins / self.episode_count) * 100
        
        # Store training data for JSON export
        self.training_data['episode_numbers'].append(int(self.episode_count))
        self.training_data['frames_elapsed'].append(int(self.frame_count))
        self.training_data['scores'].append(f"{self.red_score}-{self.blue_score}")
        self.training_data['winners'].append(self.winner)
        self.training_data['episode_times'].append(float(episode_time))
        self.training_data['red_win_rates'].append(float(red_win_rate))
        self.training_data['blue_win_rates'].append(float(blue_win_rate))
        self.training_data['epsilons'].append(float(self.red_agent.epsilon))
        
        # Print episode statistics
        print(f"Episode {self.episode_count}/{self.episodes} - "
              f"Winner: {self.winner}, Score: Red {self.red_score}-{self.blue_score} Blue, "
              f"Frames: {self.frame_count}, Time: {episode_time:.2f}s, "
              f"Red Win Rate: {red_win_rate:.1f}%, Blue Win Rate: {blue_win_rate:.1f}%, "
              f"Epsilon: {self.red_agent.epsilon:.3f}")
    
    def save_training_data(self, filename="training_data.json"):
        """Save training data to JSON file."""
        import json
        import numpy as np
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif hasattr(obj, 'dtype'):  # Handle any numpy scalar types
                return float(obj)
            else:
                return obj
        
        # Add final statistics to training data
        final_data = {
            'episode_numbers': self.training_data['episode_numbers'],
            'frames_elapsed': self.training_data['frames_elapsed'],
            'scores': self.training_data['scores'],
            'winners': self.training_data['winners'],
            'episode_times': self.training_data['episode_times'],
            'red_win_rates': self.training_data['red_win_rates'],
            'blue_win_rates': self.training_data['blue_win_rates'],
            'epsilons': self.training_data['epsilons'],
            'final_statistics': {
                'total_episodes': int(self.episode_count),
                'total_frames': int(self.total_frames),
                'red_wins': int(self.red_wins),
                'blue_wins': int(self.blue_wins),
                'avg_episode_length': float(self.avg_episode_length),
                'final_epsilon': float(self.red_agent.epsilon),
                'final_step_count': int(self.red_agent.step_count),
                'final_episode_count': int(self.red_agent.episode_count),
                'final_total_reward': float(self.red_agent.total_reward)
            }
        }
        
        # Convert all numpy types to Python types
        final_data = convert_numpy_types(final_data)
        
        # Additional safety check - convert any remaining numpy scalars
        def deep_convert_numpy(obj):
            if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, list):
                return [deep_convert_numpy(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: deep_convert_numpy(value) for key, value in obj.items()}
            else:
                return obj
        
        final_data = deep_convert_numpy(final_data)
        
        # Final safety check - convert any numpy types that might have been missed
        def final_convert(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [final_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: final_convert(value) for key, value in obj.items()}
            else:
                return obj
        
        final_data = final_convert(final_data)
        
        try:
            with open(filename, 'w') as f:
                json.dump(final_data, f, indent=2)
            print(f"✓ Training data saved to {filename}")
            print(f"  - Episodes: {len(self.training_data['episode_numbers'])}")
            print(f"  - Total frames: {self.total_frames}")
            print(f"  - Red wins: {self.red_wins}")
            print(f"  - Blue wins: {self.blue_wins}")
        except Exception as e:
            print(f"✗ Failed to save training data: {e}")
            print(f"Error type: {type(e)}")
            # Try to identify the problematic data
            try:
                json.dumps(final_data)
            except Exception as debug_e:
                print(f"JSON serialization debug error: {debug_e}")
                # Try to find the problematic field
                for key, value in final_data.items():
                    try:
                        json.dumps({key: value})
                    except Exception as field_e:
                        print(f"Problem with field '{key}': {field_e}")
                        # If it's a list, check each element
                        if isinstance(value, list):
                            for i, item in enumerate(value):
                                try:
                                    json.dumps({f"{key}[{i}]": item})
                                except Exception as item_e:
                                    print(f"  Problem with {key}[{i}]: {item_e} (type: {type(item)})")
                        # If it's a dict, check each subfield
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                try:
                                    json.dumps({f"{key}.{subkey}": subvalue})
                                except Exception as subfield_e:
                                    print(f"  Problem with {key}.{subkey}: {subfield_e} (type: {type(subvalue)})")
    
    def train(self):
        """Run the training loop."""
        print(f"Starting Q-Net training for {self.episodes} episodes...")
        print(f"Target FPS: {self.target_fps}")
        if self.model_file:
            print(f"Using specified model: {self.model_file}")
        else:
            print("Auto-loading most recent model (if available)")
        print("=" * 80)
        
        # Save initial model if starting from scratch
        if self.red_agent.episode_count == 0:
            print("Saving initial model...")
            self.red_agent.save_model("qnet_model_initial.pth")
        
        try:
            for episode in range(self.episodes):
                self.run_episode()
                
                # Save model every 50 episodes (more frequent)
                if (episode + 1) % 50 == 0:
                    print(f"Saving model at episode {episode + 1}...")
                    self.red_agent.save_model(f"qnet_model_episode_{episode + 1}.pth")
                
                # Save training data only at the end (removed from here to prevent overwriting)
                
                # Print detailed statistics every 25 episodes
                if (episode + 1) % 25 == 0:
                    self.print_detailed_stats()
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            print("Saving current model before exit...")
            self.red_agent.save_model("qnet_model_interrupted.pth")
            print("Model saved as 'qnet_model_interrupted.pth'")
            print("Saving training data...")
            self.save_training_data("training_data_interrupted.json")
        
        except Exception as e:
            print(f"\nTraining error: {e}")
            print("Saving current model before exit...")
            self.red_agent.save_model("qnet_model_error.pth")
            print("Model saved as 'qnet_model_error.pth'")
            print("Saving training data...")
            self.save_training_data("training_data_error.json")
            raise
        
        # Final statistics
        self.print_final_stats()
        
        # Create final training graphs
        print("Creating final training graphs...")
        self.red_agent.create_training_graphs()
        
        # Save training data to JSON
        print("Saving training data to JSON...")
        self.save_training_data()
        
        # Save final model
        print("Saving final model...")
        self.red_agent.save_model("qnet_model_final.pth")
        print("Training completed successfully!")
    
    def print_detailed_stats(self):
        """Print detailed training statistics."""
        total_time = time.time() - self.start_time
        episodes_per_second = self.episode_count / total_time
        
        print("\n" + "=" * 80)
        print("DETAILED TRAINING STATISTICS")
        print("=" * 80)
        print(f"Episodes completed: {self.episode_count}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Episodes per second: {episodes_per_second:.2f}")
        print(f"Average episode length: {self.avg_episode_length:.1f} frames")
        print(f"Red wins: {self.red_wins} ({self.red_wins/self.episode_count*100:.1f}%)")
        print(f"Blue wins: {self.blue_wins} ({self.blue_wins/self.episode_count*100:.1f}%)")
        print(f"Current epsilon: {self.red_agent.epsilon:.3f}")
        print(f"Total steps: {self.red_agent.step_count}")
        print(f"Average reward per episode: {self.red_agent.total_reward/self.episode_count:.2f}")
        print("=" * 80)
    
    def print_final_stats(self):
        """Print final training statistics."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("FINAL TRAINING STATISTICS")
        print("=" * 80)
        print(f"Total episodes: {self.episode_count}")
        print(f"Total training time: {total_time:.2f} seconds")
        print(f"Average episode time: {total_time/self.episode_count:.2f} seconds")
        print(f"Total frames: {self.total_frames}")
        print(f"Average episode length: {self.avg_episode_length:.1f} frames")
        print(f"Red wins: {self.red_wins} ({self.red_wins/self.episode_count*100:.1f}%)")
        print(f"Blue wins: {self.blue_wins} ({self.blue_wins/self.episode_count*100:.1f}%)")
        print(f"Final epsilon: {self.red_agent.epsilon:.3f}")
        print(f"Total steps: {self.red_agent.step_count}")
        print(f"Final average reward: {self.red_agent.total_reward/self.episode_count:.2f}")
        print("=" * 80)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Q-Net agent against AI")
    parser.add_argument('-e', '--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('-f', '--fps', type=int, default=60,
                       help='Target FPS for training (default: 60)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model file to load (auto-loads most recent if not specified)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create training environment with auto-loading
    env = TrainingEnvironment(target_fps=args.fps, episodes=args.episodes, model_file=args.model)
    
    # Start training
    env.train()
    
    pygame.quit()
    sys.exit() 