import pygame
import sys
import argparse
from grid import Grid
from agent import Agent, AIAgent
from qnet_agent import QNetAgent
from constants import *

class CTFGame:
    def __init__(self, target_fps=60, demo_mode=False, qnet_model=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Capture the Flag")
        self.clock = pygame.time.Clock()
        self.running = True
        self.frame_count = 0
        
        # FPS settings
        self.target_fps = target_fps
        self.current_fps = 0
        self.fps_update_timer = 0
        self.fps_update_interval = 10  # Update FPS display every 10 frames
        
        # Cache fonts to avoid creating them every frame
        self.fonts = {
            'title': pygame.font.Font(None, 36),
            'score': pygame.font.Font(None, 48),
            'score_small': pygame.font.Font(None, 24),
            'health': pygame.font.Font(None, 24),
            'instruction': pygame.font.Font(None, 20),
            'fps': pygame.font.Font(None, 24),
            'game_over': pygame.font.Font(None, 72),
            'game_over_small': pygame.font.Font(None, 48),
            'game_over_instruction': pygame.font.Font(None, 24)
        }
        
        # Initialize game grid
        self.grid = Grid()
        
        # Demo mode settings
        self.demo_mode = demo_mode
        self.qnet_model = qnet_model
        
        # Initialize agents based on mode
        if demo_mode and qnet_model:
            # Demo mode: AI (blue) vs Trained Q-Net (red)
            self.red_agent = QNetAgent(self.grid, TEAM_RED, 7, 3, self.target_fps, demo_mode=True)
            self.red_agent.load_model(qnet_model)
            self.red_agent.epsilon = 0.0  # No exploration in demo mode
            self.blue_agent = AIAgent(self.grid, TEAM_BLUE, 7, 16, self.target_fps, self.red_agent)
            self.red_agent.enemy_agent = self.blue_agent
        else:
            # Normal mode: Player (red) vs AI (blue)
            self.red_agent = Agent(self.grid, TEAM_RED, 7, 3, self.target_fps)
            self.blue_agent = AIAgent(self.grid, TEAM_BLUE, 7, 16, self.target_fps, self.red_agent)
        
        self.agents = [self.red_agent, self.blue_agent]
        
        # Game state
        self.game_over = False
        self.winner = None
        self.game_over_timer = 0
        self.game_over_delay = self.target_fps * 3  # 3 seconds to show game over screen
        
        # Scoreboard
        self.red_score = 0
        self.blue_score = 0
        self.captures_to_win = CAPTURES_TO_WIN  # Number of flag captures needed to win
        
        # Cache frequently rendered text surfaces (after all variables are initialized)
        self.cached_texts = {}
        self.update_cached_texts()
        
        # Performance optimization flags
        self.last_score_update = 0
        self.score_update_interval = 5  # Update score text every 5 frames
    
    def update_cached_texts(self):
        """Update cached text surfaces that change frequently"""
        # Title text
        self.cached_texts['title'] = self.fonts['title'].render("Capture the Flag", True, WHITE)
        self.cached_texts['title_rect'] = self.cached_texts['title'].get_rect(center=(WINDOW_WIDTH // 2, 30))
        
        # Instructions (dynamic based on mode)
        if self.demo_mode:
            instructions = [
                "DEMO MODE: Q-Net (Red) vs AI (Blue)",
                "Both agents are AI-controlled",
                "Watch the trained Q-Net agent in action!",
                "Collect enemy flags and return to your base!",
                "R - Reset Game",
                "ESC - Quit"
            ]
        else:
            instructions = [
                "Movement: WASD (Red agent only)",
                "Red Agent: Left click to shoot",
                "Blue Agent: AI-controlled",
                "Collect enemy flags and return to your base!",
                "R - Reset Game",
                "ESC - Quit"
            ]
        self.cached_texts['instructions'] = []
        for instruction in instructions:
            text_surface = self.fonts['instruction'].render(instruction, True, WHITE)
            self.cached_texts['instructions'].append(text_surface)
        
        # Scoreboard title
        self.cached_texts['scoreboard_title'] = self.fonts['score_small'].render("SCOREBOARD", True, WHITE)
        self.cached_texts['scoreboard_title_rect'] = self.cached_texts['scoreboard_title'].get_rect(center=(WINDOW_WIDTH - 105, 25))
        
        # Win condition text
        self.cached_texts['win_condition'] = self.fonts['score_small'].render(f"First to {self.captures_to_win} wins!", True, WHITE)
        self.cached_texts['win_condition_rect'] = self.cached_texts['win_condition'].get_rect(center=(WINDOW_WIDTH - 105, 95))
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset game
                    self.reset_game()
            elif event.type == pygame.MOUSEBUTTONDOWN and not self.demo_mode:
                if event.button == 1:  # Left click
                    # Red agent shoots (only in normal mode)
                    self.red_agent.shooting = True
            elif event.type == pygame.MOUSEBUTTONUP and not self.demo_mode:
                if event.button == 1:  # Left click
                    self.red_agent.shooting = False
    
    def handle_input(self):
        """Handle continuous input for agents"""
        if self.game_over:
            return
            
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        current_time = pygame.time.get_ticks()
        
        if not self.demo_mode:
            # Normal mode: Handle red agent input (player controls red)
            self.red_agent.handle_movement_input(keys)
            self.red_agent.handle_shooting_input(mouse_pos)
            if self.red_agent.shooting:
                self.red_agent.shoot(current_time)
        # Demo mode: No player input needed - both agents are AI-controlled
    
    def update(self):
        """Update game state"""
        current_time = pygame.time.get_ticks()
        self.frame_count += 1
        
        # Update FPS counter
        self.fps_update_timer += 1
        if self.fps_update_timer >= self.fps_update_interval:
            self.current_fps = int(self.clock.get_fps())
            self.fps_update_timer = 0
        
        if self.game_over:
            self.game_over_timer += 1
            if self.game_over_timer >= self.game_over_delay:
                self.reset_game()
            return
        
        # Update agents and check for flag captures
        for agent in self.agents:
            capture_result = agent.update(self.frame_count)
            if capture_result == "red_capture":
                self.red_score += 1
                print(f"Red team captured Blue flag! Score: Red {self.red_score} - Blue {self.blue_score}")
            elif capture_result == "blue_capture":
                self.blue_score += 1
                print(f"Blue team captured Red flag! Score: Red {self.red_score} - Blue {self.blue_score}")
        
        # Check for bullet collisions with agents (only if there are bullets)
        if any(agent.bullets for agent in self.agents):
            self.check_bullet_collisions()
        
        # Check for win condition
        self.check_win_condition()
    
    def check_win_condition(self):
        """Check if either team has won"""
        if self.red_score >= self.captures_to_win:
            self.game_over = True
            self.winner = "Red"
            print(f"Red team wins! Final score: Red {self.red_score} - Blue {self.blue_score}")
        elif self.blue_score >= self.captures_to_win:
            self.game_over = True
            self.winner = "Blue"
            print(f"Blue team wins! Final score: Red {self.red_score} - Blue {self.blue_score}")
    
    def reset_game(self):
        """Reset the game to initial state"""
        # Reset only flags and bases without regenerating the map layout
        self.grid.reset_flags_and_bases()
        
        # Reset agents based on mode
        if self.demo_mode and self.qnet_model:
            # Demo mode: AI (blue) vs Trained Q-Net (red)
            self.red_agent = QNetAgent(self.grid, TEAM_RED, 7, 3, self.target_fps, demo_mode=True)
            self.red_agent.load_model(self.qnet_model)
            self.red_agent.epsilon = 0.0  # No exploration in demo mode
            self.blue_agent = AIAgent(self.grid, TEAM_BLUE, 7, 16, self.target_fps, self.red_agent)
            self.red_agent.enemy_agent = self.blue_agent
        else:
            # Normal mode: Player (red) vs AI (blue)
            self.red_agent = Agent(self.grid, TEAM_RED, 7, 3, self.target_fps)
            self.blue_agent = AIAgent(self.grid, TEAM_BLUE, 7, 16, self.target_fps, self.red_agent)
        
        self.agents = [self.red_agent, self.blue_agent]
        self.frame_count = 0
        
        # Reset game state
        self.game_over = False
        self.winner = None
        self.game_over_timer = 0
        
        # Reset scoreboard
        self.red_score = 0
        self.blue_score = 0
    
    def check_bullet_collisions(self):
        """Check for collisions between bullets and agents"""
        # Only check if there are bullets and alive agents
        alive_agents = [agent for agent in self.agents if agent.alive]
        if not alive_agents:
            return
            
        for agent in alive_agents:
            agent_rect = agent.get_rect()
            
            # Check bullets from other agents
            for other_agent in self.agents:
                if other_agent == agent or not other_agent.bullets:
                    continue
                    
                bullets_to_remove = []
                for bullet in other_agent.bullets:
                    # Use distance check first for better performance
                    dx = bullet.x - agent.x
                    dy = bullet.y - agent.y
                    distance_squared = dx*dx + dy*dy
                    collision_distance = agent.radius + bullet.radius
                    
                    if distance_squared <= collision_distance * collision_distance:
                        agent.take_damage(bullet.damage)
                        bullets_to_remove.append(bullet)
                
                # Remove bullets that hit
                for bullet in bullets_to_remove:
                    if bullet in other_agent.bullets:
                        other_agent.bullets.remove(bullet)
    
    def render(self):
        """Render the game"""
        self.screen.fill(BLACK)
        
        # Draw the grid
        self.grid.draw(self.screen)
        
        # Draw agents
        for agent in self.agents:
            agent.draw(self.screen)
        
        # Draw UI elements
        self.draw_ui()
        
        # Draw game over screen if game is over
        if self.game_over:
            self.draw_game_over_screen()
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Draw UI elements"""
        # Draw title
        self.screen.blit(self.cached_texts['title'], self.cached_texts['title_rect'])
        
        # Draw scoreboard
        self.draw_scoreboard()
        
        # Draw FPS counter in lower right corner
        self.draw_fps_counter()
        
        # Draw agent health
        health_font = self.fonts['health']
        
        # Red agent health
        if self.red_agent.alive:
            red_health_text = f"Red Agent: {self.red_agent.health} HP"
            red_health_color = RED
        else:
            remaining_frames = self.red_agent.respawn_delay - self.red_agent.respawn_timer
            remaining_seconds = remaining_frames / self.target_fps
            red_health_text = f"Red Agent: Respawning... {remaining_seconds:.1f}s"
            red_health_color = GRAY
        red_health_surface = health_font.render(red_health_text, True, red_health_color)
        self.screen.blit(red_health_surface, (10, 120))
        
        # Blue agent health
        if self.blue_agent.alive:
            blue_health_text = f"Blue Agent: {self.blue_agent.health} HP"
            blue_health_color = BLUE
        else:
            remaining_frames = self.blue_agent.respawn_delay - self.blue_agent.respawn_timer
            remaining_seconds = remaining_frames / self.target_fps
            blue_health_text = f"Blue Agent: Respawning... {remaining_seconds:.1f}s"
            blue_health_color = GRAY
        blue_health_surface = health_font.render(blue_health_text, True, blue_health_color)
        self.screen.blit(blue_health_surface, (10, 145))
        
        # Show mode indicator
        if self.demo_mode:
            mode_text = "DEMO MODE: You (Blue) vs Trained Q-Net (Red)"
            mode_color = YELLOW
        else:
            mode_text = "NORMAL MODE: You (Red) vs AI (Blue)"
            mode_color = WHITE
        mode_surface = health_font.render(mode_text, True, mode_color)
        self.screen.blit(mode_surface, (10, 170))
        
        # Draw instructions based on mode
        if self.demo_mode:
            instructions = [
                "Movement: WASD (Blue agent only)",
                "Blue Agent: Left click to shoot",
                "Red Agent: Trained Q-Net AI",
                "Collect enemy flags and return to your base!",
                "R - Reset Game",
                "ESC - Quit"
            ]
        else:
            instructions = [
                "Movement: WASD (Red agent only)",
                "Red Agent: Left click to shoot",
                "Blue Agent: AI-controlled",
                "Collect enemy flags and return to your base!",
                "R - Reset Game",
                "ESC - Quit"
            ]
        
        for i, instruction in enumerate(instructions):
            text_surface = self.fonts['instruction'].render(instruction, True, WHITE)
            self.screen.blit(text_surface, (10, WINDOW_HEIGHT - 120 + i * 20))

    def draw_scoreboard(self):
        """Draw the scoreboard"""
        # Draw scoreboard background
        scoreboard_rect = pygame.Rect(WINDOW_WIDTH - 200, 10, 190, 80)
        pygame.draw.rect(self.screen, DARK_GRAY, scoreboard_rect)
        pygame.draw.rect(self.screen, WHITE, scoreboard_rect, 2)
        
        # Draw title
        self.screen.blit(self.cached_texts['scoreboard_title'], self.cached_texts['scoreboard_title_rect'])
        
        # Draw scores
        red_score_text = self.fonts['score'].render(f"Red: {self.red_score}", True, RED)
        red_score_rect = red_score_text.get_rect(center=(WINDOW_WIDTH - 105, 50))
        self.screen.blit(red_score_text, red_score_rect)
        
        blue_score_text = self.fonts['score'].render(f"Blue: {self.blue_score}", True, BLUE)
        blue_score_rect = blue_score_text.get_rect(center=(WINDOW_WIDTH - 105, 75))
        self.screen.blit(blue_score_text, blue_score_rect)
        
        # Draw win condition
        self.screen.blit(self.cached_texts['win_condition'], self.cached_texts['win_condition_rect'])
    
    def draw_game_over_screen(self):
        """Draw the game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Game over text
        font = self.fonts['game_over']
        game_over_text = font.render("GAME OVER", True, WHITE)
        game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 50))
        self.screen.blit(game_over_text, game_over_rect)
        
        # Winner text
        winner_color = RED if self.winner == "Red" else BLUE
        winner_text = font.render(f"{self.winner} Team Wins!", True, winner_color)
        winner_rect = winner_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 10))
        self.screen.blit(winner_text, winner_rect)
        
        # Final score
        score_font = self.fonts['game_over_small']
        final_score_text = score_font.render(f"Final Score: Red {self.red_score} - Blue {self.blue_score}", True, WHITE)
        final_score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 70))
        self.screen.blit(final_score_text, final_score_rect)
        
        # Instructions
        instruction_font = self.fonts['game_over_instruction']
        instruction_text = instruction_font.render("Press R to play again or ESC to quit", True, WHITE)
        instruction_rect = instruction_text.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120))
        self.screen.blit(instruction_text, instruction_rect)
    
    def draw_fps_counter(self):
        """Draw FPS counter in lower right corner"""
        fps_font = self.fonts['fps']
        fps_text = f"FPS: {self.current_fps}"
        fps_surface = fps_font.render(fps_text, True, WHITE)
        
        # Position in lower right corner with some padding
        fps_rect = fps_surface.get_rect()
        fps_rect.bottomright = (WINDOW_WIDTH - 10, WINDOW_HEIGHT - 10)
        
        # Draw background for better visibility
        bg_rect = fps_rect.inflate(10, 5)
        pygame.draw.rect(self.screen, DARK_GRAY, bg_rect)
        pygame.draw.rect(self.screen, WHITE, bg_rect, 1)
        
        self.screen.blit(fps_surface, fps_rect)

    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_events()
            self.handle_input()
            self.update()
            self.render()
            self.clock.tick(self.target_fps)
        
        pygame.quit()
        sys.exit()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Capture the Flag Game")
    parser.add_argument('-f', '--fps', type=int, default=60, 
                       help='Target FPS for the game (default: 60)')
    parser.add_argument('--demo', type=str, metavar='MODEL_FILE',
                       help='Demo mode: play against trained Q-Net model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.demo:
        # Demo mode: Player vs Trained Q-Net
        game = CTFGame(target_fps=args.fps, demo_mode=True, qnet_model=args.demo)
        print(f"Demo mode: Playing against trained Q-Net model: {args.demo}")
    else:
        # Normal mode: Player vs AI
        game = CTFGame(target_fps=args.fps)
        print("Normal mode: Playing against AI")
    
    game.run() 