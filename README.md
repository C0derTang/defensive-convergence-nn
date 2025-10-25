# Capture the Flag Game with AI Agents

A sophisticated 2D pygame-based Capture the Flag game featuring AI agents, Q-Network reinforcement learning, and multiple game modes. This project demonstrates advanced game AI techniques including rule-based AI, neural network-based agents, and reinforcement learning.

## 🎮 Features

### Core Game Features
- **Symmetric Grid**: Procedurally generated symmetric grid with walls for balanced gameplay
- **Dual Team Gameplay**: Red vs Blue teams with home bases and flags
- **Real-time Combat**: Shooting mechanics with bullet physics and collision detection
- **Health System**: Agents have health, respawn mechanics, and damage system
- **Score System**: First team to capture 3 enemy flags wins

### AI Features
- **Rule-based AI Agent**: Intelligent AI opponent with strategic behavior
- **Q-Network Agent**: Neural network-based reinforcement learning agent
- **Training System**: Headless training environment for efficient AI learning
- **Demo Mode**: Play against trained AI models
- **Performance Optimization**: High FPS training and smooth gameplay

### Game Modes
- **Normal Mode**: Player (Red) vs AI (Blue)
- **Demo Mode**: AI (Blue) vs Trained Q-Net (Red) - Both agents are AI-controlled
- **Training Mode**: AI vs AI for model training

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cleanctf
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the game**:
   ```bash
   # Normal mode: Player vs AI
   python main.py
   
   # Demo mode: Watch AI vs Trained Q-Net (both agents are AI-controlled)
   python main.py --demo qnet_model_final.pth
   ```

## 🎯 Game Controls

### Normal Mode (Player vs AI)
- **WASD**: Move Red agent
- **Left Click**: Shoot
- **R**: Reset game
- **ESC**: Quit

### Demo Mode (AI vs Trained Q-Net)
- **No player input** - Both agents are AI-controlled
- **R**: Reset game
- **ESC**: Quit

## 🤖 AI Training

### Training a Q-Net Agent

Train a Q-Network agent against the rule-based AI:

```bash
# Basic training (1000 episodes)
python train_qnet.py

# Advanced training options
python train_qnet.py --episodes 5000 --fps 120
```

### Training Parameters

The Q-Net agent uses these default parameters:
- **Learning Rate**: 0.001
- **Epsilon**: 1.0 (exploration rate, decays to 0.1)
- **Epsilon Decay**: 0.9990
- **Gamma**: 0.99 (discount factor)
- **Memory Size**: 10,000 experiences
- **Batch Size**: 32
- **Train Frequency**: Every 16 steps

### Training Output

During training, you'll see:
- Episode progress and results
- Win rates for both teams
- Current epsilon value (exploration rate)
- Detailed statistics every 25 episodes
- Model saves every 50 episodes
- Automatic saving on interruption (Ctrl+C)

## 🧠 Q-Network Architecture

### State Representation (18 features)
- Agent position (normalized)
- Agent health
- Flag carrying status
- Enemy position and status
- Distance to enemy flag
- Distance to own base
- Flag and base positions
- Bullet counts
- Direction to enemy

### Actions (18 actions)
- 9 movement directions (including no movement) × 2 shooting states
- Actions 0-8: Movement without shooting
- Actions 9-17: Movement with shooting

### Neural Network
- 3 fully connected layers (128 hidden units each)
- ReLU activation
- Adam optimizer
- Experience replay buffer
- Target network for stable training

## 🏆 Reward System

The Q-Net agent receives rewards for:
- **Flag capture**: +100 points
- **Flag collection**: +50 points
- **Damaging enemy**: +3 points per health point
- **Moving closer to enemy flag**: +10 points per distance unit
- **Moving closer to base (when carrying flag)**: +15 points per distance unit
- **Taking damage**: -2 points per health point
- **Death**: -50 points
- **Each step**: -0.1 points (encourages efficiency)

## 📁 Project Structure

```
├── main.py              # Main game loop and pygame initialization
├── grid.py              # Grid class with wall generation and drawing
├── agent.py             # Base agent class and rule-based AI
├── qnet_agent.py        # Q-Network reinforcement learning agent
├── constants.py          # Game constants and configuration
├── train_qnet.py        # Training script for Q-Net agent
├── analyze_training.py  # Training analysis and visualization
├── visualize_training.py # Training progress visualization
├── test_training.py     # Training testing utilities
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## 🔧 Technical Details

### Framework & Dependencies
- **Pygame**: Game engine and graphics
- **PyTorch**: Neural network implementation
- **NumPy**: Numerical computations
- **Matplotlib**: Training visualization
- **Gymnasium**: Reinforcement learning environment

### Game Mechanics
- **Grid System**: 20x15 cells (800x600 pixels) with 40x40 pixel cells
- **Symmetry Algorithm**: Mirrors wall placement for balanced gameplay
- **Collision Detection**: Efficient distance-based collision system
- **Bullet Physics**: Realistic projectile motion with collision detection
- **Respawn System**: 3-second respawn delay with visual countdown

### Performance Features
- **Font Caching**: Pre-rendered text surfaces for better performance
- **Efficient Rendering**: Optimized drawing with minimal redraws
- **FPS Control**: Configurable target FPS for consistent gameplay
- **Memory Management**: Proper cleanup of bullets and temporary objects

## 📊 Training Analysis

The project includes comprehensive training analysis tools:

```bash
# Analyze training results
python analyze_training.py

# Visualize training progress
python visualize_training.py
```

These tools provide:
- Training progress graphs
- Win rate analysis
- Reward evolution
- Performance metrics
- Strategy analysis

## 🎯 Performance Tips

### Faster Training
- Use higher FPS (120-240) for faster episodes
- Train for more episodes (2000-5000) for better performance
- Use a GPU if available (PyTorch will automatically detect CUDA)

### Better Models
- Train for longer periods (more episodes)
- Let epsilon decay fully (exploration rate goes to 0.01)
- Monitor win rates - good models should achieve >40% win rate against AI

### Demo Mode Performance
- Trained models perform best with epsilon = 0.0 (no exploration)
- Models trained for 1000+ episodes typically show good performance
- The Q-Net may develop different strategies than the rule-based AI

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU training
2. **Slow training**: Increase FPS or reduce episode count for testing
3. **Poor performance**: Train for more episodes or adjust learning parameters
4. **Model not loading**: Check file path and ensure model file exists
5. **Pygame not found**: Make sure you've installed the requirements
6. **Display issues**: Check your Python and pygame versions

### Debugging Tips

- Monitor epsilon decay during training
- Check win rates - should improve over time
- Look for increasing average rewards
- Verify model files are being saved correctly
- Use training analysis tools to identify issues

## 📈 Example Training Session

```bash
# Start training (auto-loads most recent model if available)
python train_qnet.py --episodes 2000 --fps 120

# After training completes, play against the model
python main.py --demo qnet_model_final.pth
```

This will automatically continue training from the most recent model (if available) for 2000 episodes and then let you play against the trained model in demo mode.

## 🤝 Contributing

This project demonstrates advanced AI techniques in game development. Feel free to:
- Experiment with different neural network architectures
- Modify the reward system
- Add new game features
- Improve the AI algorithms
- Create new training scenarios

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Pygame community for the excellent game development framework
- PyTorch team for the powerful deep learning library
- OpenAI Gym/Gymnasium for reinforcement learning environment standards 