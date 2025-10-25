import pygame
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from agent import Agent, AIAgent
from constants import *
from grid import CellType

class QNetwork(nn.Module):
    """Neural network for Q-learning."""
    
    def __init__(self, input_size, output_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QNetAgent(Agent):
    """Q-Network based agent for the red team that trains against the AI."""
    
    def __init__(self, grid, team, start_row, start_col, target_fps=60, enemy_agent=None, 
                 learning_rate=0.001, epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1,
                 gamma=0.99, memory_size=10000, batch_size=32, train_frequency=4, demo_mode=False):
        super().__init__(grid, team, start_row, start_col, target_fps)
        self.enemy_agent = enemy_agent
        self.demo_mode = demo_mode
        
        # Q-Network parameters
        self.input_size = 18  # State representation size
        self.output_size = 18  # 9 movement directions × 2 (with/without shooting)
        self.hidden_size = 128
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.train_frequency = train_frequency
        
        # Neural networks
        self.q_network = QNetwork(self.input_size, self.output_size, self.hidden_size)
        self.target_network = QNetwork(self.input_size, self.output_size, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training variables
        self.step_count = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.total_reward = 0
        
        # Training tracking for graphing
        self.episode_rewards = []  # Points per episode
        self.episode_frames = []   # Frames survived per episode
        self.episode_numbers = []  # Episode numbers for x-axis
        self.avg_rewards = []      # Moving average of rewards
        self.avg_frames = []       # Moving average of frames
        self.window_size = 10      # Window for moving average
        self.episode_frame_count = 0  # Current episode frame counter
        
        # Action mapping - 18 actions total
        # Actions 0-8: Movement without shooting
        # Actions 9-17: Movement with shooting
        self.actions = [
            # Movement without shooting (actions 0-8)
            (0, 0, False),   # No movement, no shooting
            (0, -1, False),  # North, no shooting
            (0, 1, False),   # South, no shooting
            (-1, 0, False),  # West, no shooting
            (1, 0, False),   # East, no shooting
            (-1, -1, False), # Northwest, no shooting
            (1, -1, False),  # Northeast, no shooting
            (-1, 1, False),  # Southwest, no shooting
            (1, 1, False),   # Southeast, no shooting
            # Movement with shooting (actions 9-17)
            (0, 0, True),    # No movement, shooting
            (0, -1, True),   # North, shooting
            (0, 1, True),    # South, shooting
            (-1, 0, True),   # West, shooting
            (1, 0, True),    # East, shooting
            (-1, -1, True),  # Northwest, shooting
            (1, -1, True),   # Northeast, shooting
            (-1, 1, True),   # Southwest, shooting
            (1, 1, True),    # Southeast, shooting
        ]
        
        # Note: Model loading is handled externally when needed
    
    def get_state(self):
        """Get the current state representation for the Q-network."""
        state = []
        
        # Agent position (normalized)
        state.append(self.x / WINDOW_WIDTH)
        state.append(self.y / WINDOW_HEIGHT)
        
        # Agent health (normalized)
        state.append(self.health / 100.0)
        
        # Flag carrying status
        state.append(1.0 if self.carrying_flag else 0.0)
        
        # Enemy position and status (if enemy exists)
        if self.enemy_agent and self.enemy_agent.alive:
            state.append(self.enemy_agent.x / WINDOW_WIDTH)
            state.append(self.enemy_agent.y / WINDOW_HEIGHT)
            state.append(self.enemy_agent.health / 100.0)
            state.append(1.0 if self.enemy_agent.carrying_flag else 0.0)
        else:
            state.extend([0.0, 0.0, 0.0, 0.0])
        
        # Distance to enemy flag
        flag_pos = self.find_enemy_flag()
        if flag_pos:
            flag_x, flag_y = flag_pos
            distance = math.sqrt((self.x - flag_x)**2 + (self.y - flag_y)**2)
            state.append(distance / (WINDOW_WIDTH + WINDOW_HEIGHT))  # Normalized
        else:
            state.append(1.0)  # Flag not found
        
        # Distance to own base
        base_pos = self.get_base_position()
        if base_pos:
            base_x, base_y = base_pos
            distance = math.sqrt((self.x - base_x)**2 + (self.y - base_y)**2)
            state.append(distance / (WINDOW_WIDTH + WINDOW_HEIGHT))  # Normalized
        else:
            state.append(1.0)
        
        # Enemy flag position (if known)
        if flag_pos:
            state.append(flag_pos[0] / WINDOW_WIDTH)
            state.append(flag_pos[1] / WINDOW_HEIGHT)
        else:
            state.extend([0.0, 0.0])
        
        # Own base position
        if base_pos:
            state.append(base_pos[0] / WINDOW_WIDTH)
            state.append(base_pos[1] / WINDOW_HEIGHT)
        else:
            state.extend([0.0, 0.0])
        
        # Number of bullets
        state.append(len(self.bullets) / 10.0)  # Normalized
        
        # Enemy bullets (if enemy exists)
        if self.enemy_agent:
            state.append(len(self.enemy_agent.bullets) / 10.0)
        else:
            state.append(0.0)
        
        # Direction to enemy (if enemy exists)
        if self.enemy_agent and self.enemy_agent.alive:
            dx = self.enemy_agent.x - self.x
            dy = self.enemy_agent.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                state.append(dx / distance)  # Normalized direction
                state.append(dy / distance)
            else:
                state.extend([0.0, 0.0])
        else:
            state.extend([0.0, 0.0])
        
        return np.array(state, dtype=np.float32)
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # During exploration, prefer movement actions (avoid action 0) 80% of the time
            if random.random() < 0.8:
                return random.randint(1, self.output_size - 1)  # Skip action 0
            else:
                return random.randint(0, self.output_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def execute_action(self, action):
        """Execute the chosen action and return the actual action executed."""
        if action < len(self.actions):
            dx, dy, should_shoot = self.actions[action]
            
            # Handle movement
            if dx != 0 or dy != 0:
                # Check if the movement would be valid before setting it
                new_x = self.x + dx * self.speed
                new_y = self.y + dy * self.speed
                new_col = int((new_x - GRID_OFFSET_X) // GRID_SIZE)
                new_row = int((new_y - GRID_OFFSET_Y) // GRID_SIZE)
                
                # Only move if the new position is valid and safe
                if (self.grid.is_valid_position(new_row, new_col) and 
                    self.is_position_safe(new_x, new_y)):
                    self.move_direction = (dx, dy)
                    self.moving = True
                else:
                    # If movement is invalid, try a random valid direction
                    return self.try_random_valid_movement()
            else:
                self.moving = False
            
            # Handle shooting
            if should_shoot:
                # Aim at enemy if available, otherwise shoot in current direction
                if self.enemy_agent and self.enemy_agent.alive:
                    dx = self.enemy_agent.x - self.x
                    dy = self.enemy_agent.y - self.y
                    self.shoot_angle = math.degrees(math.atan2(-dy, dx))
                    if self.shoot_angle < 0:
                        self.shoot_angle += 360
                
                self.shooting = True
                self.shoot(pygame.time.get_ticks())
            
            return action  # Return the original action
        else:
            # Invalid action, return no movement action
            return 0
    
    def try_random_valid_movement(self):
        """Try to find a random valid movement direction when the chosen action is invalid."""
        import random
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Simple 4-directional movement
        random.shuffle(directions)
        
        for dx, dy in directions:
            new_x = self.x + dx * self.speed
            new_y = self.y + dy * self.speed
            new_col = int((new_x - GRID_OFFSET_X) // GRID_SIZE)
            new_row = int((new_y - GRID_OFFSET_Y) // GRID_SIZE)
            
            if (self.grid.is_valid_position(new_row, new_col) and 
                self.is_position_safe(new_x, new_y)):
                self.move_direction = (dx, dy)
                self.moving = True
                # Find the action index for this direction (without shooting)
                for i, (action_dx, action_dy, action_shoot) in enumerate(self.actions):
                    if action_dx == dx and action_dy == dy and not action_shoot:
                        return i
                return 0  # Default to no movement if not found
        
        # If no valid direction found, don't move
        self.moving = False
        return 0  # Return no movement action
    
    def get_reward(self, old_state, action, new_state, capture_result=None):
        """Calculate reward for the current state transition."""
        reward = 0
        
        # Flag capture reward (highest priority)
        if capture_result == "red_capture":
            reward += 200  # Increased from 100
        
        # Flag collection reward
        if not old_state[3] and new_state[3]:  # Started carrying flag
            reward += 100  # Increased from 50
        
        # Health loss penalty (reduced)
        health_diff = new_state[2] - old_state[2]
        if health_diff < 0:
            reward += health_diff  # Reduced penalty (was *2)
        
        # Enemy damage reward (increased)
        if self.enemy_agent and self.enemy_agent.alive and len(new_state) > 6:
            enemy_health_diff = new_state[6] - old_state[6]
            if enemy_health_diff < 0:
                reward += abs(enemy_health_diff) * 5  # Increased from 3
        
        # Distance to enemy flag reward (closer is better)
        if len(old_state) > 8 and len(new_state) > 8:
            flag_distance_diff = old_state[8] - new_state[8]
            reward += flag_distance_diff * 20  # Increased from 10
        
        # Distance to base reward (when carrying flag)
        if new_state[3] and len(old_state) > 9 and len(new_state) > 9:  # Carrying flag
            base_distance_diff = old_state[9] - new_state[9]
            reward += base_distance_diff * 30  # Increased from 15
        
        # Movement reward - encourage exploration (increased)
        if len(old_state) > 0 and len(new_state) > 0:
            pos_change = abs(new_state[0] - old_state[0]) + abs(new_state[1] - old_state[1])
            if pos_change > 0.01:  # Actually moved
                reward += 1.0  # Increased movement reward
            else:
                reward -= 0.1  # Penalty for not moving
        
        # Small penalty for each step (reduced)
        reward -= 0.01  # Reduced from 0.05
        
        # Reward for staying alive (encourage survival)
        if self.alive:
            reward += 0.1  # Small positive reward for staying alive
        
        # Death penalty (reduced)
        if not self.alive:
            reward -= 10  # Further reduced penalty
        
        return reward
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the Q-network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % 1000 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update(self, frame_count):
        """Update the Q-Net agent."""
        if not self.alive:
            # Handle respawn timer
            self.respawn_timer += 1
            if self.respawn_timer >= self.respawn_delay:
                self.respawn()
            return None
        
        # Track frames survived in this episode
        self.episode_frame_count += 1
        
        # Get current state
        current_state = self.get_state()
        
        # Choose action
        action = self.choose_action(current_state)
        
        # Debug: Print action choice occasionally
        
        
        # Execute action and get the actual result
        actual_action = self.execute_action(action)
        
        # Update agent (movement, bullets, flag collection)
        capture_result = super().update(frame_count)
        
        # Get new state
        new_state = self.get_state()
        
        # Calculate reward based on actual action executed
        reward = self.get_reward(current_state, actual_action, new_state, capture_result)
        self.episode_reward += reward
        
        # Store experience with actual action (only if not in demo mode)
        done = not self.alive
        if not self.demo_mode:
            self.remember(current_state, actual_action, reward, new_state, done)
        
        # Train network (only if not in demo mode)
        self.step_count += 1
        if not self.demo_mode and self.step_count % self.train_frequency == 0:
            self.train()
        
        # Epsilon decay is now handled in end_episode() for episode-based decay
        
        return capture_result
    
    def find_enemy_flag(self):
        """Find the position of the enemy flag."""
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                cell_type = self.grid.grid[row][col]
                if (self.team == TEAM_RED and cell_type == CellType.BLUE_FLAG):
                    return (col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2,
                           row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2)
        return None
    
    def get_base_position(self):
        """Get the position of our base."""
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                cell_type = self.grid.grid[row][col]
                if (self.team == TEAM_RED and cell_type == CellType.RED_BASE):
                    return (col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2,
                           row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2)
        return None
    
    def save_model(self, filename="qnet_model.pth"):
        """Save the trained model."""
        try:
            checkpoint = {
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'step_count': self.step_count,
                'episode_count': self.episode_count,
                'total_reward': self.total_reward,
                'episode_rewards': self.episode_rewards,
                'episode_frames': self.episode_frames,
                'episode_numbers': self.episode_numbers,
                'avg_rewards': self.avg_rewards,
                'avg_frames': self.avg_frames
            }
            torch.save(checkpoint, filename)
            print(f"✓ Model saved to {filename}")
            print(f"  - Episodes: {self.episode_count}")
            print(f"  - Steps: {self.step_count}")
            print(f"  - Epsilon: {self.epsilon:.3f}")
            
            # Training data is saved by TrainingEnvironment, not here
            # self.save_training_data()
            
        except Exception as e:
            print(f"✗ Failed to save model to {filename}: {e}")
            raise
    
    def load_model(self, filename="qnet_model.pth"):
        """Load a trained model."""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found")
            return
            
        try:
            # Try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(filename, weights_only=False)
            print(f"Model loaded from {filename} (weights_only=False)")
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=False: {e}")
            print("Trying with weights_only=True...")
            try:
                # If that fails, try with weights_only=True and safe globals
                import torch.serialization
                torch.serialization.add_safe_globals(['numpy._core.multiarray.scalar'])
                checkpoint = torch.load(filename, weights_only=True)
                print(f"Model loaded from {filename} (weights_only=True)")
            except Exception as e2:
                print(f"Error: Failed to load model with both methods: {e2}")
                return
        
        try:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.step_count = checkpoint['step_count']
            self.episode_count = checkpoint['episode_count']
            self.total_reward = checkpoint['total_reward']
            
            # Load training data if available
            if 'episode_rewards' in checkpoint:
                # Convert numpy types to Python types
                def convert_to_python_types(data):
                    if isinstance(data, list):
                        return [float(item) if hasattr(item, 'dtype') else item for item in data]
                    return data
                
                self.episode_rewards = convert_to_python_types(checkpoint['episode_rewards'])
                self.episode_frames = convert_to_python_types(checkpoint['episode_frames'])
                self.episode_numbers = convert_to_python_types(checkpoint['episode_numbers'])
                self.avg_rewards = convert_to_python_types(checkpoint['avg_rewards'])
                self.avg_frames = convert_to_python_types(checkpoint['avg_frames'])
                print(f"✓ Training data loaded from checkpoint")
            else:
                print(f"⚠ No training data found in checkpoint")
            
            print(f"✓ Model state loaded successfully")
            print(f"  - Episodes: {self.episode_count}")
            print(f"  - Steps: {self.step_count}")
            print(f"  - Epsilon: {self.epsilon:.3f}")
        except Exception as e:
            print(f"Error loading model state: {e}")
    
    def end_episode(self):
        """Called at the end of each episode."""
        self.episode_count += 1
        self.total_reward += self.episode_reward
        
        # Track episode data for graphing
        self.episode_rewards.append(self.episode_reward)
        self.episode_frames.append(self.episode_frame_count)
        self.episode_numbers.append(self.episode_count)
        
        # Calculate moving averages
        if len(self.episode_rewards) >= self.window_size:
            avg_reward = float(np.mean(self.episode_rewards[-self.window_size:]))
            avg_frame = float(np.mean(self.episode_frames[-self.window_size:]))
        else:
            avg_reward = float(np.mean(self.episode_rewards))
            avg_frame = float(np.mean(self.episode_frames))
        
        self.avg_rewards.append(avg_reward)
        self.avg_frames.append(avg_frame)
        
        # Print training statistics
        if self.episode_count % 10 == 0:
            print(f"Episode {self.episode_count}, Avg Reward: {avg_reward:.2f}, Avg Frames: {avg_frame:.1f}, Epsilon: {self.epsilon:.3f}")
        
        # Create graphs every 50 episodes
        if self.episode_count % 50 == 0:
            self.create_training_graphs()
        
        # Reset episode counters
        self.episode_reward = 0
        self.episode_frame_count = 0
        
        # Decay epsilon every episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def create_training_graphs(self):
        """Create and save training progress graphs."""
        if len(self.episode_numbers) < 2:
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Episode Rewards (Points)
        ax1.plot(self.episode_numbers, self.episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
        ax1.plot(self.episode_numbers, self.avg_rewards, 'r-', linewidth=2, label=f'{self.window_size}-Episode Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward (Points)')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Frames Survived
        ax2.plot(self.episode_numbers, self.episode_frames, 'g-', alpha=0.3, label='Frames Survived')
        ax2.plot(self.episode_numbers, self.avg_frames, 'orange', linewidth=2, label=f'{self.window_size}-Episode Moving Average')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Frames Survived')
        ax2.set_title('Training Progress: Frames Survived per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f'training_progress_episode_{self.episode_count}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training graphs saved: training_progress_episode_{self.episode_count}.png")
    
    def save_training_data(self, filename="training_data.json"):
        """Save training data to JSON file for later analysis."""
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
        
        training_data = {
            'episode_numbers': self.episode_numbers,
            'episode_rewards': self.episode_rewards,
            'episode_frames': self.episode_frames,
            'avg_rewards': self.avg_rewards,
            'avg_frames': self.avg_frames,
            'final_episode_count': self.episode_count,
            'final_step_count': self.step_count,
            'final_epsilon': self.epsilon,
            'final_total_reward': self.total_reward
        }
        
        # Convert all numpy types to Python types
        training_data = convert_numpy_types(training_data)
        
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
        
        training_data = deep_convert_numpy(training_data)
        
        # Final safety check - convert any numpy types that might have been missed
        def final_convert(obj):
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
        
        training_data = final_convert(training_data)
        
        try:
            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2)
            print(f"✓ Training data saved to {filename}")
        except Exception as e:
            print(f"✗ Failed to save training data: {e}")
            print(f"Error type: {type(e)}")
            # Try to identify the problematic data
            try:
                json.dumps(training_data)
            except Exception as debug_e:
                print(f"JSON serialization debug error: {debug_e}")
                # Try to find the problematic field
                for key, value in training_data.items():
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
    
    def load_training_data(self, filename="training_data.json"):
        """Load training data from JSON file."""
        import json
        
        try:
            with open(filename, 'r') as f:
                training_data = json.load(f)
            
            self.episode_numbers = training_data['episode_numbers']
            self.episode_rewards = training_data['episode_rewards']
            self.episode_frames = training_data['episode_frames']
            self.avg_rewards = training_data['avg_rewards']
            self.avg_frames = training_data['avg_frames']
            
            print(f"✓ Training data loaded from {filename}")
            print(f"  - Episodes: {len(self.episode_numbers)}")
            print(f"  - Avg Reward: {np.mean(self.episode_rewards):.2f}")
            print(f"  - Avg Frames: {np.mean(self.episode_frames):.1f}")
            
        except Exception as e:
            print(f"✗ Failed to load training data: {e}") 