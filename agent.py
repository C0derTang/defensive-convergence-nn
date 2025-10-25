import pygame
import math
import random
from constants import *
from grid import CellType

class Agent:
    def __init__(self, grid, team, start_row, start_col, target_fps=60):
        """
        Initialize an agent with movement and shooting capabilities.
        
        Args:
            grid: The game grid for collision detection
            team: Team color ("red" or "blue")
            start_row: Starting row position
            start_col: Starting column position
            target_fps: Target FPS for frame-consistent timing (default: 60)
        """
        self.grid = grid
        self.team = team
        self.row = start_row
        self.col = start_col
        self.x = start_col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2
        self.y = start_row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2
        
        # Movement properties
        self.speed = 2
        self.moving = False
        self.move_direction = (0, 0)
        
        # Shooting properties
        self.shooting = False
        self.shoot_angle = 0  # Angle in degrees
        self.bullets = []
        self.bullet_speed = 5
        self.last_shot_time = 0
        self.shot_cooldown = 500  # milliseconds
        
        # Visual properties
        self.radius = GRID_SIZE // 3
        self.color = RED if team == TEAM_RED else BLUE
        self.team_color = self.color
        
        # Health and status
        self.health = 100
        self.alive = True
        self.respawn_timer = 0
        self.target_fps = target_fps
        self.respawn_delay = target_fps * 3  # 3 seconds at target FPS for frame-consistent timing
        self.original_row = start_row
        self.original_col = start_col
        
        # Flag collection properties
        self.carrying_flag = False
        self.carried_flag_type = None  # CellType.RED_FLAG or CellType.BLUE_FLAG
        
        # Cache fonts to avoid creating them every frame
        try:
            self.fonts = {
                'respawn': pygame.font.Font(None, 36),
                'team': pygame.font.Font(None, 24),
                'flag': pygame.font.Font(None, 24)
            }
            
            # Cache team text
            self.team_text = self.fonts['team'].render("R" if team == TEAM_RED else "B", True, WHITE)
            self.team_text_rect = self.team_text.get_rect()
            
            # Cache flag text
            self.flag_text = self.fonts['flag'].render("F", True, WHITE)
            self.flag_text_rect = self.flag_text.get_rect()
        except pygame.error:
            # Handle case where pygame is not fully initialized (e.g., during testing)
            self.fonts = None
            self.team_text = None
            self.team_text_rect = None
            self.flag_text = None
            self.flag_text_rect = None
    
    def handle_movement_input(self, keys):
        """
        Handle keyboard input for 8-directional movement (WASD only).
        
        Args:
            keys: pygame key state
        """
        # Reset movement
        self.moving = False
        self.move_direction = (0, 0)
        
        # 8-directional movement with WASD
        if keys[pygame.K_w]:  # North
            self.move_direction = (0, -1)
            self.moving = True
        elif keys[pygame.K_s]:  # South
            self.move_direction = (0, 1)
            self.moving = True
        elif keys[pygame.K_a]:  # West
            self.move_direction = (-1, 0)
            self.moving = True
        elif keys[pygame.K_d]:  # East
            self.move_direction = (1, 0)
            self.moving = True
    
    def handle_shooting_input(self, mouse_pos):
        """
        Handle mouse input for 360-degree shooting.
        
        Args:
            mouse_pos: Current mouse position (x, y)
        """
        # Calculate angle from agent to mouse
        dx = mouse_pos[0] - self.x
        dy = mouse_pos[1] - self.y
        self.shoot_angle = math.degrees(math.atan2(-dy, dx))
        
        # Normalize angle to 0-360 degrees
        if self.shoot_angle < 0:
            self.shoot_angle += 360
    
    def shoot(self, current_time):
        """
        Shoot a bullet in the current direction.
        
        Args:
            current_time: Current game time in milliseconds
        """
        if current_time - self.last_shot_time > self.shot_cooldown:
            # Create new bullet
            bullet = Bullet(self.x, self.y, self.shoot_angle, self.team)
            self.bullets.append(bullet)
            self.last_shot_time = current_time
    
    def update(self, frame_count):
        """
        Update agent position and bullets.
        
        Args:
            frame_count: Current frame count for respawn timing
        """
        if not self.alive:
            # Handle respawn timer
            self.respawn_timer += 1
            if self.respawn_timer >= self.respawn_delay:
                self.respawn()
            return
            
        # Update movement
        if self.moving:
            self.move()
        
        # Update bullets
        self.update_bullets()
        
        # Check for flag collection
        self.check_flag_collection()
        
        # Check for flag capture (returning to base with enemy flag)
        capture_result = self.check_flag_capture()
        return capture_result
    
    def move(self):
        """Move the agent in the current direction if possible."""
        if not self.moving:
            return
            
        dx, dy = self.move_direction
        
        # Calculate new position
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        # Convert to grid coordinates
        new_col = int((new_x - GRID_OFFSET_X) // GRID_SIZE)
        new_row = int((new_y - GRID_OFFSET_Y) // GRID_SIZE)
        
        # Check if new position is valid
        if self.grid.is_valid_position(new_row, new_col):
            # Check if we're not too close to walls
            if self.is_position_safe(new_x, new_y):
                self.x = new_x
                self.y = new_y
                self.row = new_row
                self.col = new_col
    
    def is_position_safe(self, x, y):
        """
        Check if a position is safe (not too close to walls).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            bool: True if position is safe
        """
        # Check surrounding grid cells
        col = int((x - GRID_OFFSET_X) // GRID_SIZE)
        row = int((y - GRID_OFFSET_Y) // GRID_SIZE)
        
        for r in range(max(0, row - 1), min(self.grid.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.grid.cols, col + 2)):
                if self.grid.grid[r][c] == CellType.WALL:
                    # Check distance to wall - reduced safety margin for better navigation
                    wall_x = c * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2
                    wall_y = r * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2
                    distance = math.sqrt((x - wall_x)**2 + (y - wall_y)**2)
                    if distance < self.radius + 2:  # Reduced safety margin
                        return False
        return True
    
    def check_flag_collection(self):
        """Check if the agent can collect a flag."""
        if self.carrying_flag:
            return  # Already carrying a flag
            
        # Check if we're on a flag cell
        cell_type = self.grid.grid[self.row][self.col]
        
        # Can only collect enemy flags
        if (self.team == TEAM_RED and cell_type == CellType.BLUE_FLAG):
            self.carrying_flag = True
            self.carried_flag_type = CellType.BLUE_FLAG
            # Remove flag from grid
            self.grid.grid[self.row][self.col] = CellType.EMPTY
        elif (self.team == TEAM_BLUE and cell_type == CellType.RED_FLAG):
            self.carrying_flag = True
            self.carried_flag_type = CellType.RED_FLAG
            # Remove flag from grid
            self.grid.grid[self.row][self.col] = CellType.EMPTY
    
    def check_flag_capture(self):
        """Check if the agent can capture a flag by returning to base."""
        if not self.carrying_flag:
            return
            
        # Check if we're on our own base
        cell_type = self.grid.grid[self.row][self.col]
        
        if (self.team == TEAM_RED and cell_type == CellType.RED_BASE and 
            self.carried_flag_type == CellType.BLUE_FLAG):
            # Red team captured blue flag
            self.carrying_flag = False
            self.carried_flag_type = None
            # Respawn the captured flag at its original position
            self.grid.respawn_flag(CellType.BLUE_FLAG)
            return "red_capture"
        elif (self.team == TEAM_BLUE and cell_type == CellType.BLUE_BASE and 
              self.carried_flag_type == CellType.RED_FLAG):
            # Blue team captured red flag
            self.carrying_flag = False
            self.carried_flag_type = None
            # Respawn the captured flag at its original position
            self.grid.respawn_flag(CellType.RED_FLAG)
            return "blue_capture"
        
        return None
    

    
    def check_and_respawn_dropped_flags(self):
        """Check if flags are missing and respawn them at original positions."""
        # This method is now simplified since flags always respawn at original positions
        # Only check if flags are missing and not being carried
        if self.team == TEAM_BLUE:
            # Check if blue flag is missing from original position
            if not self.grid.is_flag_at_original_position(CellType.BLUE_FLAG):
                # Check if any agent is carrying the blue flag
                blue_flag_carried = False
                if self.enemy_agent and self.enemy_agent.carrying_flag and self.enemy_agent.carried_flag_type == CellType.BLUE_FLAG:
                    blue_flag_carried = True
                
                # If flag is not being carried, respawn it
                if not blue_flag_carried:
                    self.grid.respawn_flag(CellType.BLUE_FLAG)
                    print("Blue flag was missing - respawned at original position")
            
            # Check if red flag is missing from original position
            if not self.grid.is_flag_at_original_position(CellType.RED_FLAG):
                # Check if any agent is carrying the red flag
                red_flag_carried = False
                if self.carrying_flag and self.carried_flag_type == CellType.RED_FLAG:
                    red_flag_carried = True
                
                # If flag is not being carried, respawn it
                if not red_flag_carried:
                    self.grid.respawn_flag(CellType.RED_FLAG)
                    print("Red flag was missing - respawned at original position")
    
    def drop_flag(self):
        """Drop the carried flag and respawn it at original position."""
        if self.carrying_flag:
            # Respawn the flag at its original position instead of dropping it
            self.grid.respawn_flag(self.carried_flag_type)
            
            # Debug message
            flag_name = "Red" if self.carried_flag_type == CellType.RED_FLAG else "Blue"
            print(f"{self.team.title()} agent killed - {flag_name} flag respawned at original position")
            
            self.carrying_flag = False
            self.carried_flag_type = None
            print(f"{self.team.title()} agent flag state reset: carrying_flag={self.carrying_flag}, carried_flag_type={self.carried_flag_type}")
    
    def update_bullets(self):
        """Update all bullets and remove ones that are out of bounds or hit walls."""
        bullets_to_remove = []
        
        for bullet in self.bullets:
            bullet.update()
            
            # Check if bullet is out of bounds
            if (bullet.x < 0 or bullet.x > WINDOW_WIDTH or 
                bullet.y < 0 or bullet.y > WINDOW_HEIGHT):
                bullets_to_remove.append(bullet)
                continue
            
            # Check if bullet hit a wall
            grid_pos = self.grid.get_cell_at_pos((bullet.x, bullet.y))
            if grid_pos:
                row, col = grid_pos
                if 0 <= row < self.grid.rows and 0 <= col < self.grid.cols:
                    if self.grid.grid[row][col] == CellType.WALL:
                        bullets_to_remove.append(bullet)
                        continue
        
        # Remove bullets that need to be removed
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
    
    def draw(self, screen):
        """Draw the agent and its bullets."""
        if not self.alive:
            # Draw respawn timer
            if self.fonts:  # Only draw if fonts are available
                remaining_frames = self.respawn_delay - self.respawn_timer
                remaining_seconds = remaining_frames / self.target_fps
                respawn_text = f"Respawn: {remaining_seconds:.1f}s"
                text = self.fonts['respawn'].render(respawn_text, True, WHITE)
                text_rect = text.get_rect(center=(int(self.x), int(self.y)))
                screen.blit(text, text_rect)
            return
            
        # Draw agent
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius, 2)
        
        # Draw team indicator
        if self.team_text and self.team_text_rect:
            screen.blit(self.team_text, self.team_text_rect.move(int(self.x), int(self.y)))
        
        # Draw AI state indicator for blue team
        if self.team == TEAM_BLUE and hasattr(self, 'current_state'):
            try:
                if not hasattr(self, 'state_font'):
                    self.state_font = pygame.font.Font(None, 16)
                
                state_text = self.current_state.replace('_', ' ').title()
                
                # Add enemy status to state text
                if hasattr(self, 'enemy_agent') and self.enemy_agent:
                    if not self.enemy_agent.alive:
                        state_text += " (Enemy Dead)"
                
                # Add middle area indicator
                map_center_x = GRID_OFFSET_X + (self.grid.cols // 2) * GRID_SIZE
                distance_from_center = abs(self.x - map_center_x)
                if distance_from_center < 2 * GRID_SIZE and self.current_state == "seek_flag":
                    state_text += " (Middle)"
                
                state_surface = self.state_font.render(state_text, True, YELLOW)
                state_rect = state_surface.get_rect(center=(int(self.x), int(self.y) + self.radius + 20))
                screen.blit(state_surface, state_rect)
            except pygame.error:
                pass  # Skip drawing if font is not available
        
        # Draw flag indicator if carrying flag and alive
        if self.carrying_flag and self.alive and self.flag_text and self.flag_text_rect:
            flag_color = RED if self.carried_flag_type == CellType.RED_FLAG else BLUE
            # Draw flag icon above agent
            flag_rect = pygame.Rect(self.x - 8, self.y - self.radius - 15, 16, 12)
            pygame.draw.rect(screen, flag_color, flag_rect)
            pygame.draw.rect(screen, WHITE, flag_rect, 1)
            
            # Draw "F" for flag
            screen.blit(self.flag_text, self.flag_text_rect.move(flag_rect.centerx - 8, flag_rect.centery - 6))
        

        
        # Draw shooting direction indicator
        if self.shooting:
            end_x = self.x + math.cos(math.radians(self.shoot_angle)) * (self.radius + 10)
            end_y = self.y - math.sin(math.radians(self.shoot_angle)) * (self.radius + 10)
            pygame.draw.line(screen, YELLOW, (self.x, self.y), (end_x, end_y), 3)
        
        # Draw bullets
        for bullet in self.bullets:
            bullet.draw(screen)
    
    def take_damage(self, damage):
        """
        Take damage from being hit.
        
        Args:
            damage: Amount of damage to take
        """
        self.health -= damage
        if self.health <= 0:
            # Drop flag if carrying one
            if self.carrying_flag:
                self.drop_flag()
                # Ensure flag state is completely reset
                self.carrying_flag = False
                self.carried_flag_type = None
            
            self.alive = False
            self.respawn_timer = 0  # Start respawn timer
    
    def respawn(self):
        """Respawn the agent at its original position."""
        self.alive = True
        self.health = 100
        self.respawn_timer = 0
        
        # Reset position to original spawn point
        self.row = self.original_row
        self.col = self.original_col
        self.x = self.original_col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2
        self.y = self.original_row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2
        
        # Clear bullets
        self.bullets.clear()
        
        # Reset movement and shooting state
        self.moving = False
        self.move_direction = (0, 0)
        self.shooting = False
        
        # Reset flag carrying state - flags should be dropped when agent dies
        self.carrying_flag = False
        self.carried_flag_type = None
        print(f"{self.team.title()} agent respawned - flag state reset: carrying_flag={self.carrying_flag}, carried_flag_type={self.carried_flag_type}")
    
    def get_rect(self):
        """Get the collision rectangle for the agent."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)


class AIAgent(Agent):
    """AI-controlled agent with intelligent behavior for the blue team."""
    
    def __init__(self, grid, team, start_row, start_col, target_fps=60, enemy_agent=None):
        super().__init__(grid, team, start_row, start_col, target_fps)
        self.enemy_agent = enemy_agent
        
        # AI behavior states
        self.current_state = "seek_flag"  # seek_flag, return_flag, defend_base, attack_enemy
        self.state_timer = 0
        self.state_duration = 0
        
        # Pathfinding
        self.target_position = None
        self.path = []
        self.path_index = 0
        self.stuck_timer = 0
        self.last_position = (self.x, self.y)
        
        # Combat
        self.last_shot_time = 0
        self.shot_cooldown = 300  # Faster shooting
        self.engagement_range = 200  # Increased range
        self.retreat_health = 20  # More aggressive
        self.attack_cooldown = 0
        
        # Decision making
        self.decision_timer = 0
        self.decision_interval = 25  # Less frequent decisions for better performance
        
        # Memory
        self.last_enemy_position = None
        self.enemy_visible = False
        self.base_threat_level = 0
        self.stuck_threshold = 5  # Even more sensitive stuck detection
        self.last_successful_position = (self.x, self.y)
        self.position_history = []
        
    def update(self, frame_count):
        """Update AI agent with intelligent behavior."""
        if not self.alive:
            # Handle respawn timer
            self.respawn_timer += 1
            if self.respawn_timer >= self.respawn_delay:
                self.respawn()
            return
        
        # Check if stuck - more sensitive detection
        current_pos = (round(self.x, 1), round(self.y, 1))  # Round to avoid floating point issues
        
        # Add to position history (keep last 10 positions)
        self.position_history.append(current_pos)
        if len(self.position_history) > 10:
            self.position_history.pop(0)
        
        # Simple stuck detection - just check if we're not moving
        if current_pos == self.last_position:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0
        
        self.last_position = current_pos
        
        # Update decision making
        self.decision_timer += 1
        if self.decision_timer >= self.decision_interval:
            self.make_decision()
            self.decision_timer = 0
        
        # Update state timer
        self.state_timer += 1
        
        # Execute current behavior
        self.execute_current_state()
        
        # Check if we're in a tight space and need to escape
        if self.is_in_tight_space() and self.stuck_timer > 5:
            self.escape_tight_space()
        

        
        # Update movement
        if self.moving:
            self.move()
        
        # Update bullets
        self.update_bullets()
        
        # Check for flag collection
        self.check_flag_collection()
        
        # Check for dropped flags and respawn them
        self.check_and_respawn_dropped_flags()
        
        # Check for flag capture (carrying flag)
        capture_result = self.check_flag_capture()
        return capture_result
    
    def make_decision(self):
        """Make high-level decisions about what to do."""
        # Update threat assessment
        self.assess_threats()
        
        # Check if enemy is dead - if so, prioritize flag seeking
        enemy_dead = not self.enemy_agent or not self.enemy_agent.alive
        
        # Determine current state based on situation
        if self.carrying_flag:
            # If carrying flag, prioritize returning over engagement completely
            self.current_state = "return_flag"
        elif enemy_dead:
            # If enemy is dead, focus on flag seeking
            self.current_state = "seek_flag"
        elif self.enemy_visible and self.health > self.retreat_health:
            # More aggressive - attack enemy if visible and healthy
            self.current_state = "attack_enemy"
        elif self.is_enemy_near_base():
            self.current_state = "defend_base"
        else:
            self.current_state = "seek_flag"
    
    def assess_threats(self):
        """Assess current threats and update enemy information."""
        if self.enemy_agent and self.enemy_agent.alive:
            # Calculate distance to enemy
            dx = self.enemy_agent.x - self.x
            dy = self.enemy_agent.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Update enemy visibility - increased range
            self.enemy_visible = distance < 300  # Increased visibility range
            
            if self.enemy_visible:
                self.last_enemy_position = (self.enemy_agent.x, self.enemy_agent.y)
                
                # Check if enemy is near our base
                base_distance = self.get_distance_to_base()
                if distance < base_distance + 100:  # Enemy is closer to base than we are
                    self.base_threat_level = min(10, self.base_threat_level + 1)
                else:
                    self.base_threat_level = max(0, self.base_threat_level - 1)
        else:
            # Enemy is dead or doesn't exist
            self.enemy_visible = False
            self.base_threat_level = max(0, self.base_threat_level - 1)
    
    def execute_current_state(self):
        """Execute the current AI state behavior."""
        if self.current_state == "seek_flag":
            self.seek_flag_behavior()
        elif self.current_state == "return_flag":
            self.return_flag_behavior()
        elif self.current_state == "defend_base":
            self.defend_base_behavior()
        elif self.current_state == "attack_enemy":
            self.attack_enemy_behavior()
    
    def seek_flag_behavior(self):
        """Behavior for seeking the enemy flag."""
        # Find enemy flag position
        flag_pos = self.find_enemy_flag()
        if flag_pos:
            self.move_towards_position(flag_pos)
        else:
            # If enemy is dead, be more aggressive in exploration
            enemy_dead = not self.enemy_agent or not self.enemy_agent.alive
            if enemy_dead:
                # More systematic exploration when enemy is dead
                self.systematic_exploration()
            else:
                # Random exploration if flag not found
                self.explore_randomly()
    
    def return_flag_behavior(self):
        """Behavior for returning captured flag to base."""
        # Shoot at enemy if visible for self-defense while returning
        if self.enemy_visible and self.enemy_agent and self.enemy_agent.alive:
            self.aim_at_enemy(self.enemy_agent)
            self.shoot(pygame.time.get_ticks())
        
        # Always prioritize moving towards base
        base_pos = self.get_base_position()
        if base_pos:
            self.move_towards_position(base_pos)
    
    def defend_base_behavior(self):
        """Behavior for defending the base from enemy."""
        if self.enemy_visible and self.last_enemy_position and self.enemy_agent and self.enemy_agent.alive:
            # Shoot at enemy first (priority)
            self.aim_at_enemy(self.enemy_agent)
            self.shoot(pygame.time.get_ticks())
            
            # Move towards enemy aggressively
            self.move_towards_position(self.last_enemy_position)
        else:
            # Patrol around base
            self.patrol_base()
    
    def attack_enemy_behavior(self):
        """Behavior for attacking the enemy."""
        if self.enemy_visible and self.last_enemy_position and self.enemy_agent and self.enemy_agent.alive:
            # Calculate distance to enemy
            dx = self.enemy_agent.x - self.x
            dy = self.enemy_agent.y - self.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Shoot at enemy first (priority)
            self.aim_at_enemy(self.enemy_agent)
            self.shoot(pygame.time.get_ticks())
            
            # Move towards enemy if not too close
            if distance > 50:  # Keep some distance for better shooting
                self.move_towards_position(self.last_enemy_position)
            else:
                # Strafe around enemy
                self.strafe_around_enemy()
        else:
            # Return to seeking flag if enemy lost or dead
            self.current_state = "seek_flag"
    
    def find_enemy_flag(self):
        """Find the position of the enemy flag."""
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                cell_type = self.grid.grid[row][col]
                if (self.team == TEAM_BLUE and cell_type == CellType.RED_FLAG):
                    return (col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2,
                           row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2)
        return None
    
    def get_base_position(self):
        """Get the position of our base."""
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                cell_type = self.grid.grid[row][col]
                if (self.team == TEAM_BLUE and cell_type == CellType.BLUE_BASE):
                    return (col * GRID_SIZE + GRID_OFFSET_X + GRID_SIZE // 2,
                           row * GRID_SIZE + GRID_OFFSET_Y + GRID_SIZE // 2)
        return None
    
    def move_towards_position(self, target_pos):
        """Move towards a target position with simple collision detection."""
        if not target_pos:
            return
        
        # Store target position for navigation methods
        self.last_target_pos = target_pos
        
        target_x, target_y = target_pos
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 5:  # Only move if not close enough
            # Simple direction selection - prioritize the larger difference
            if abs(dx) > abs(dy):
                # Move horizontally first
                if dx > 0 and self.is_direction_valid((1, 0)):
                    self.move_direction = (1, 0)
                    self.moving = True
                elif dx < 0 and self.is_direction_valid((-1, 0)):
                    self.move_direction = (-1, 0)
                    self.moving = True
                else:
                    # Wall detected - store desired direction and try to go around it
                    if dx > 0:
                        self.last_desired_direction = (1, 0)  # Right
                    else:
                        self.last_desired_direction = (-1, 0)  # Left
                    self.navigate_around_wall_simple(target_x)
            else:
                # Move vertically first
                if dy > 0 and self.is_direction_valid((0, 1)):
                    self.move_direction = (0, 1)
                    self.moving = True
                elif dy < 0 and self.is_direction_valid((0, -1)):
                    self.move_direction = (0, -1)
                    self.moving = True
                else:
                    # Wall detected - store desired direction and try to go around it
                    if dy > 0:
                        self.last_desired_direction = (0, 1)  # Down
                    else:
                        self.last_desired_direction = (0, -1)  # Up
                    self.navigate_around_wall_simple(target_x)
        else:
            self.moving = False
    
    def is_direction_valid(self, direction):
        """Check if a movement direction is valid (no walls)."""
        dx, dy = direction
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        # Convert to grid coordinates
        new_col = int((new_x - GRID_OFFSET_X) // GRID_SIZE)
        new_row = int((new_y - GRID_OFFSET_Y) // GRID_SIZE)
        
        # Check if new position is valid
        if not self.grid.is_valid_position(new_row, new_col):
            return False
        
        # Check if position is safe (not too close to walls)
        if not self.is_position_safe(new_x, new_y):
            return False
        
        # Simplified look-ahead check for better performance
        look_ahead_x = self.x + dx * self.speed * 1.5
        look_ahead_y = self.y + dy * self.speed * 1.5
        look_ahead_col = int((look_ahead_x - GRID_OFFSET_X) // GRID_SIZE)
        look_ahead_row = int((look_ahead_y - GRID_OFFSET_Y) // GRID_SIZE)
        
        if not self.grid.is_valid_position(look_ahead_row, look_ahead_col):
            return False
        
        return True
    
    def try_alternative_directions(self, target_pos):
        """Try alternative directions when primary direction is blocked."""
        target_x, target_y = target_pos
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Try perpendicular directions first (wall following)
        directions = []
        if abs(dx) > abs(dy):
            # Primary is horizontal, try vertical first
            if dy > 0:
                directions.append((0, 1))
            else:
                directions.append((0, -1))
            # Then try opposite horizontal
            if dx > 0:
                directions.append((-1, 0))
            else:
                directions.append((1, 0))
            # Finally try the other vertical
            if dy > 0:
                directions.append((0, -1))
            else:
                directions.append((0, 1))
        else:
            # Primary is vertical, try horizontal first
            if dx > 0:
                directions.append((1, 0))
            else:
                directions.append((-1, 0))
            # Then try opposite vertical
            if dy > 0:
                directions.append((0, -1))
            else:
                directions.append((0, 1))
            # Finally try the other horizontal
            if dx > 0:
                directions.append((-1, 0))
            else:
                directions.append((1, 0))
        
        # Try each direction
        for direction in directions:
            if self.is_direction_valid(direction):
                self.move_direction = direction
                self.moving = True
                return
        
        # If all fail, try random direction
        random_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for direction in random_directions:
            if self.is_direction_valid(direction):
                self.move_direction = direction
                self.moving = True
                return
        
        # If still stuck, stop moving
        self.moving = False
    
    def try_alternative_path(self, target_pos):
        """Try to find an alternative path when stuck."""
        # Reset stuck timer and clear position history
        self.stuck_timer = 0
        self.position_history.clear()
        
        # Try to find a path around the wall
        target_x, target_y = target_pos
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Try perpendicular directions first (wall following)
        if abs(dx) > abs(dy):
            # Target is more horizontal, try vertical directions
            directions = [(0, -1), (0, 1)]
            # Add horizontal directions as backup
            if dx > 0:
                directions.extend([(-1, 0), (1, 0)])
            else:
                directions.extend([(1, 0), (-1, 0)])
        else:
            # Target is more vertical, try horizontal directions
            directions = [(-1, 0), (1, 0)]
            # Add vertical directions as backup
            if dy > 0:
                directions.extend([(0, -1), (0, 1)])
            else:
                directions.extend([(0, 1), (0, -1)])
        
        # Try each direction
        for direction in directions:
            if self.is_direction_valid(direction):
                self.move_direction = direction
                self.moving = True
                return
        
        # If all fail, try random directions with more variety
        random_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(random_directions)  # Randomize order
        for direction in random_directions:
            if self.is_direction_valid(direction):
                self.move_direction = direction
                self.moving = True
                return
        
        # If still stuck, try to move towards the most open area
        self.move_towards_open_area()
        
        # If still stuck, stop moving
        if not self.moving:
            self.moving = False
    
    def move_towards_open_area(self):
        """Move towards the most open area when stuck."""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        best_direction = None
        best_openness = -1
        
        for direction in directions:
            if not self.is_direction_valid(direction):
                continue
            
            # Calculate how "open" this direction is
            dx, dy = direction
            openness = 0
            
            # Check multiple steps ahead
            for steps in range(1, 6):  # Check further ahead
                test_x = self.x + dx * self.speed * steps
                test_y = self.y + dy * self.speed * steps
                test_col = int((test_x - GRID_OFFSET_X) // GRID_SIZE)
                test_row = int((test_y - GRID_OFFSET_Y) // GRID_SIZE)
                
                if self.grid.is_valid_position(test_row, test_col):
                    openness += 1
                else:
                    break
            
            if openness > best_openness:
                best_openness = openness
                best_direction = direction
        
        if best_direction:
            self.move_direction = best_direction
            self.moving = True
    
    def explore_randomly(self):
        """Explore the map randomly when no clear objective."""
        # Simple random exploration
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # Try current direction first if it's valid
        if hasattr(self, 'move_direction') and self.is_direction_valid(self.move_direction):
            self.moving = True
        else:
            # Pick a random valid direction
            random.shuffle(directions)
            for direction in directions:
                if self.is_direction_valid(direction):
                    self.move_direction = direction
                    self.moving = True
                    break
        
        # If no valid direction found, don't move
        if not hasattr(self, 'move_direction') or not self.moving:
            self.moving = False
    
    def systematic_exploration(self):
        """Systematic exploration when enemy is dead - more aggressive flag seeking."""
        # Simple direct movement towards enemy side
        target_x = GRID_OFFSET_X + 3 * GRID_SIZE  # Move towards enemy side
        
        # Just try to move towards the target directly
        if self.x < target_x:
            # Move right towards enemy
            if self.is_direction_valid((1, 0)):
                self.move_direction = (1, 0)
                self.moving = True
            else:
                # Wall detected - try to go around it
                self.navigate_around_wall(target_x)
        else:
            # We're past the target, move left back towards enemy
            if self.is_direction_valid((-1, 0)):
                self.move_direction = (-1, 0)
                self.moving = True
            else:
                # Wall detected - try to go around it
                self.navigate_around_wall(target_x)
    
    def navigate_around_wall(self, target_x):
        """Simple wall-following behavior to navigate around obstacles."""
        # Check which side of the wall we should go around
        # If we're trying to go right (towards enemy), try going up first, then down
        if self.x < target_x:
            # Try going up first
            if self.is_direction_valid((0, -1)):
                self.move_direction = (0, -1)
                self.moving = True
            # Then try going down
            elif self.is_direction_valid((0, 1)):
                self.move_direction = (0, 1)
                self.moving = True
            else:
                # If both vertical directions are blocked, try random exploration
                self.explore_randomly()
        else:
            # We're trying to go left (back towards enemy), same logic
            if self.is_direction_valid((0, -1)):
                self.move_direction = (0, -1)
                self.moving = True
            elif self.is_direction_valid((0, 1)):
                self.move_direction = (0, 1)
                self.moving = True
            else:
                self.explore_randomly()
    
    def navigate_around_wall_simple(self, target_x):
        """Wall-following navigation - move perpendicular until original direction is available."""
        # Get the original desired direction from the calling context
        original_direction = None
        if hasattr(self, 'last_desired_direction'):
            original_direction = self.last_desired_direction
        
        if original_direction:
            dx, dy = original_direction
            
            if dx != 0:  # Original direction was horizontal (left/right)
                # Try to move up or down until we can move horizontally
                if self.is_direction_valid((0, -1)):  # Up
                    self.move_direction = (0, -1)
                    self.moving = True
                elif self.is_direction_valid((0, 1)):  # Down
                    self.move_direction = (0, 1)
                    self.moving = True
                else:
                    # If can't move vertically, try opposite horizontal direction
                    if dx > 0 and self.is_direction_valid((-1, 0)):  # Was going right, try left
                        self.move_direction = (-1, 0)
                        self.moving = True
                    elif dx < 0 and self.is_direction_valid((1, 0)):  # Was going left, try right
                        self.move_direction = (1, 0)
                        self.moving = True
                    else:
                        self.explore_randomly()
            
            else:  # Original direction was vertical (up/down)
                # Try to move left or right until we can move vertically
                if self.is_direction_valid((-1, 0)):  # Left
                    self.move_direction = (-1, 0)
                    self.moving = True
                elif self.is_direction_valid((1, 0)):  # Right
                    self.move_direction = (1, 0)
                    self.moving = True
                else:
                    # If can't move horizontally, try opposite vertical direction
                    if dy > 0 and self.is_direction_valid((0, -1)):  # Was going down, try up
                        self.move_direction = (0, -1)
                        self.moving = True
                    elif dy < 0 and self.is_direction_valid((0, 1)):  # Was going up, try down
                        self.move_direction = (0, 1)
                        self.moving = True
                    else:
                        self.explore_randomly()
        else:
            # Fallback if no original direction stored
            if self.is_direction_valid((0, -1)):  # Up
                self.move_direction = (0, -1)
                self.moving = True
            elif self.is_direction_valid((1, 0)):  # Right
                self.move_direction = (1, 0)
                self.moving = True
            elif self.is_direction_valid((0, 1)):  # Down
                self.move_direction = (0, 1)
                self.moving = True
            elif self.is_direction_valid((-1, 0)):  # Left
                self.move_direction = (-1, 0)
                self.moving = True
            else:
                self.explore_randomly()
    

    

    

    

    
    def patrol_base(self):
        """Patrol around the base area."""
        base_pos = self.get_base_position()
        if base_pos:
            # Move in a small circle around the base
            base_x, base_y = base_pos
            dx = self.x - base_x
            dy = self.y - base_y
            
            # Calculate patrol radius
            patrol_radius = 80
            current_distance = math.sqrt(dx*dx + dy*dy)
            
            if current_distance > patrol_radius + 20:
                # Move towards base
                self.move_towards_position(base_pos)
            else:
                # Patrol in a circle
                angle = math.atan2(dy, dx) + 0.1  # Rotate slowly
                patrol_x = base_x + math.cos(angle) * patrol_radius
                patrol_y = base_y + math.sin(angle) * patrol_radius
                self.move_towards_position((patrol_x, patrol_y))
    
    def aim_at_enemy(self, enemy):
        """Aim at the enemy agent."""
        if not enemy or not enemy.alive:
            return
        
        dx = enemy.x - self.x
        dy = enemy.y - self.y
        self.shoot_angle = math.degrees(math.atan2(-dy, dx))
        
        if self.shoot_angle < 0:
            self.shoot_angle += 360
    
    def is_enemy_near_base(self):
        """Check if enemy is near our base."""
        if not self.enemy_agent or not self.enemy_agent.alive:
            return False
        
        base_pos = self.get_base_position()
        if not base_pos:
            return False
        
        base_x, base_y = base_pos
        dx = self.enemy_agent.x - base_x
        dy = self.enemy_agent.y - base_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        return distance < 120  # Base defense range
    
    def get_distance_to_base(self):
        """Get distance to our base."""
        base_pos = self.get_base_position()
        if not base_pos:
            return float('inf')
        
        base_x, base_y = base_pos
        dx = self.x - base_x
        dy = self.y - base_y
        return math.sqrt(dx*dx + dy*dy)
    
    def strafe_around_enemy(self):
        """Strafe around the enemy for better positioning."""
        if not self.enemy_agent or not self.enemy_agent.alive:
            return
        
        # Calculate perpendicular direction to enemy
        dx = self.enemy_agent.x - self.x
        dy = self.enemy_agent.y - self.y
        
        # Normalize and rotate 90 degrees for strafe
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx = dx / length
            dy = dy / length
            
            # Strafe direction (perpendicular)
            strafe_x = -dy
            strafe_y = dx
            
            # Convert to grid movement
            if abs(strafe_x) > abs(strafe_y):
                if strafe_x > 0:
                    direction = (1, 0)
                else:
                    direction = (-1, 0)
            else:
                if strafe_y > 0:
                    direction = (0, 1)
                else:
                    direction = (0, -1)
            
            # Check if strafe direction is valid
            if self.is_direction_valid(direction):
                self.move_direction = direction
                self.moving = True
            else:
                # Try opposite direction
                opposite_direction = (-direction[0], -direction[1])
                if self.is_direction_valid(opposite_direction):
                    self.move_direction = opposite_direction
                    self.moving = True
                else:
                    self.moving = False
    
    def is_in_tight_space(self):
        """Check if the AI is in a tight space or corner."""
        # Count how many valid directions are available
        valid_directions = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for direction in directions:
            if self.is_direction_valid(direction):
                valid_directions += 1
        
        # If only 1 or 0 directions available, we're in a tight space
        return valid_directions <= 1
    
    def escape_tight_space(self):
        """Try to escape from a tight space."""
        # Try to move towards the most open area
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        best_direction = None
        best_score = -1
        
        for direction in directions:
            if self.is_direction_valid(direction):
                # Calculate how "open" this direction is
                dx, dy = direction
                test_x = self.x + dx * self.speed * 3
                test_y = self.y + dy * self.speed * 3
                test_col = int((test_x - GRID_OFFSET_X) // GRID_SIZE)
                test_row = int((test_y - GRID_OFFSET_Y) // GRID_SIZE)
                
                if self.grid.is_valid_position(test_row, test_col):
                    # Count valid directions from the test position
                    open_directions = 0
                    for test_dir in directions:
                        test_dx, test_dy = test_dir
                        new_x = test_x + test_dx * self.speed
                        new_y = test_y + test_dy * self.speed
                        new_col = int((new_x - GRID_OFFSET_X) // GRID_SIZE)
                        new_row = int((new_y - GRID_OFFSET_Y) // GRID_SIZE)
                        
                        if self.grid.is_valid_position(new_row, new_col):
                            open_directions += 1
                    
                    if open_directions > best_score:
                        best_score = open_directions
                        best_direction = direction
        
        if best_direction:
            self.move_direction = best_direction
            self.moving = True
    
    def shoot(self, current_time):
        """Shoot with AI timing."""
        if current_time - self.last_shot_time > self.shot_cooldown:
            bullet = Bullet(self.x, self.y, self.shoot_angle, self.team)
            self.bullets.append(bullet)
            self.last_shot_time = current_time


class Bullet:
    def __init__(self, x, y, angle, team):
        """
        Initialize a bullet.
        
        Args:
            x: Starting X position
            y: Starting Y position
            angle: Shooting angle in degrees
            team: Team that fired the bullet
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.team = team
        self.speed = 8
        self.radius = 3
        self.damage = 25
        
        # Calculate velocity
        self.vx = math.cos(math.radians(angle)) * self.speed
        self.vy = -math.sin(math.radians(angle)) * self.speed
        
        # Visual properties
        self.color = RED if team == TEAM_RED else BLUE
    
    def update(self):
        """Update bullet position."""
        self.x += self.vx
        self.y += self.vy
    
    def draw(self, screen):
        """Draw the bullet."""
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.radius, 1)
    
    def get_rect(self):
        """Get the collision rectangle for the bullet."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2) 