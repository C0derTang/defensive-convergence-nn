#!/usr/bin/env python3
"""
Quick test script to verify if the Q-Net model is training properly.
"""

import pygame
import sys
import time
from grid import Grid
from qnet_agent import QNetAgent
from agent import AIAgent
from constants import *

def test_training():
    """Run a quick training test to verify the model is learning."""
    pygame.init()
    
    print("🧪 Starting Q-Net Training Test...")
    print("=" * 50)
    
    # Initialize environment
    grid = Grid()
    
    # Initialize agents with conservative parameters
    red_agent = QNetAgent(
        grid, TEAM_RED, 7, 3, 60,
        learning_rate=0.0001,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.05,
        gamma=0.95,
        memory_size=1000,  # Smaller for quick test
        batch_size=32,
        train_frequency=16
    )
    
    blue_agent = AIAgent(grid, TEAM_BLUE, 7, 16, 60, red_agent)
    red_agent.enemy_agent = blue_agent
    
    agents = [red_agent, blue_agent]
    
    # Test parameters
    test_episodes = 5
    max_frames_per_episode = 3000  # 50 seconds at 60 FPS
    
    print(f"Testing {test_episodes} episodes with max {max_frames_per_episode} frames each")
    print(f"Initial epsilon: {red_agent.epsilon:.3f}")
    print(f"Memory size: {len(red_agent.memory)}")
    print("-" * 50)
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(test_episodes):
        print(f"\n🎮 Episode {episode + 1}/{test_episodes}")
        
        # Reset game
        grid.reset_flags_and_bases()
        red_agent.respawn()
        blue_agent.respawn()
        red_agent.episode_reward = 0
        blue_agent.episode_reward = 0
        
        frame_count = 0
        game_over = False
        winner = None
        red_score = 0
        blue_score = 0
        
        # Run episode
        while not game_over and frame_count < max_frames_per_episode:
            frame_count += 1
            
            # Update agents
            for agent in agents:
                capture_result = agent.update(frame_count)
                if capture_result == "red_capture":
                    red_score += 1
                elif capture_result == "blue_capture":
                    blue_score += 1
            
            # Check bullet collisions
            for agent in agents:
                if agent.bullets:
                    for other_agent in agents:
                        if other_agent != agent:
                            agent_rect = other_agent.get_rect()
                            bullets_to_remove = []
                            for bullet in agent.bullets:
                                dx = bullet.x - other_agent.x
                                dy = bullet.y - other_agent.y
                                distance_squared = dx*dx + dy*dy
                                collision_distance = other_agent.radius + bullet.radius
                                if distance_squared <= collision_distance * collision_distance:
                                    other_agent.take_damage(bullet.damage)
                                    bullets_to_remove.append(bullet)
                            for bullet in bullets_to_remove:
                                if bullet in agent.bullets:
                                    agent.bullets.remove(bullet)
            
            # Check win condition
            if red_score >= CAPTURES_TO_WIN:
                game_over = True
                winner = "Red"
            elif blue_score >= CAPTURES_TO_WIN:
                game_over = True
                winner = "Blue"
        
        # End episode
        red_agent.end_episode()
        
        # Record statistics
        total_rewards.append(red_agent.episode_reward)
        episode_lengths.append(frame_count)
        
        print(f"  Winner: {winner or 'Timeout'}")
        print(f"  Score: Red {red_score}-{blue_score} Blue")
        print(f"  Frames: {frame_count}")
        print(f"  Red Reward: {red_agent.episode_reward:.2f}")
        print(f"  Epsilon: {red_agent.epsilon:.3f}")
        print(f"  Memory: {len(red_agent.memory)} experiences")
        print(f"  Steps: {red_agent.step_count}")
        
        # Check if training is happening
        if len(red_agent.memory) >= red_agent.batch_size:
            print(f"  ✅ Training possible (memory: {len(red_agent.memory)} >= batch: {red_agent.batch_size})")
        else:
            print(f"  ⚠️  Not enough memory for training (memory: {len(red_agent.memory)} < batch: {red_agent.batch_size})")
    
    # Final statistics
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    print(f"Average reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average episode length: {sum(episode_lengths)/len(episode_lengths):.1f} frames")
    print(f"Final epsilon: {red_agent.epsilon:.3f}")
    print(f"Total steps: {red_agent.step_count}")
    print(f"Memory size: {len(red_agent.memory)}")
    print(f"Episodes completed: {red_agent.episode_count}")
    
    if len(red_agent.memory) >= red_agent.batch_size:
        print("✅ MODEL IS TRAINING - Sufficient memory for batch training")
    else:
        print("❌ MODEL NOT TRAINING - Insufficient memory for batch training")
    
    if red_agent.epsilon < 0.99:
        print("✅ EPSILON DECAY WORKING - Exploration rate is decreasing")
    else:
        print("❌ EPSILON DECAY ISSUE - Exploration rate not decreasing")
    
    if red_agent.step_count > 0:
        print("✅ STEPS BEING COUNTED - Agent is taking actions")
    else:
        print("❌ NO STEPS COUNTED - Agent not taking actions")
    
    pygame.quit()
    return red_agent

if __name__ == "__main__":
    test_training() 