import pygame
import random
from constants import *
from enum import Enum

# Set a fixed random seed to make the grid the same every time
random.seed(111)

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    RED_BASE = 2
    BLUE_BASE = 3
    RED_FLAG = 4
    BLUE_FLAG = 5

class Grid:
    def __init__(self):
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.grid = [[CellType.EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Cache fonts to avoid creating them every frame
        self.fonts = {
            'base': pygame.font.Font(None, 30)
        }
        
        # Cache base text surfaces
        self.red_base_text = self.fonts['base'].render("R", True, WHITE)
        self.blue_base_text = self.fonts['base'].render("B", True, WHITE)
        
        self.generate_symmetric_grid()
        self.place_home_bases()
        
    def generate_symmetric_grid(self):
        """Generate a symmetric grid with walls"""
        # Clear the grid first
        self.grid = [[CellType.EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Generate walls in the top-left quadrant, then mirror to all quadrants
        for row in range(self.rows // 2):
            for col in range(self.cols // 2):
                
                if random.random() < WALL_PROBABILITY:
                    if(row == 4 and col == 3): continue
                    # Place wall in all four quadrants for perfect symmetry
                    self.grid[row][col] = CellType.WALL  # Top-left
                    self.grid[row][self.cols - 1 - col] = CellType.WALL  # Top-right
                    self.grid[self.rows - 1 - row][col] = CellType.WALL  # Bottom-left
                    self.grid[self.rows - 1 - row][self.cols - 1 - col] = CellType.WALL  # Bottom-right
        
        # Handle the middle row and column if grid has odd dimensions
        if self.rows % 2 == 1:
            middle_row = self.rows // 2
            for col in range(self.cols // 2):
                if random.random() < WALL_PROBABILITY:
                    self.grid[middle_row][col] = CellType.WALL
                    self.grid[middle_row][self.cols - 1 - col] = CellType.WALL
        
        if self.cols % 2 == 1:
            middle_col = self.cols // 2
            for row in range(self.rows // 2):
                if random.random() < WALL_PROBABILITY:
                    self.grid[row][middle_col] = CellType.WALL
                    self.grid[self.rows - 1 - row][middle_col] = CellType.WALL
    
    def place_home_bases(self):
        """Place home bases and flags for both teams"""
        # Red team base (left side)
        red_base_row = self.rows // 2
        red_base_col = 2
        
        # Blue team base (right side)
        blue_base_row = self.rows // 2
        blue_base_col = self.cols - 3
        
        # Clear areas around bases
        self.clear_base_area(red_base_row, red_base_col)
        self.clear_base_area(blue_base_row, blue_base_col)
        
        # Place bases
        self.grid[red_base_row][red_base_col] = CellType.RED_BASE
        self.grid[blue_base_row][blue_base_col] = CellType.BLUE_BASE
        
        # Place flags
        self.grid[red_base_row][red_base_col - 1] = CellType.RED_FLAG
        self.grid[blue_base_row][blue_base_col + 1] = CellType.BLUE_FLAG
        
        # Store original flag positions for respawning
        self.red_flag_pos = (red_base_row, red_base_col - 1)
        self.blue_flag_pos = (blue_base_row, blue_base_col + 1)
    
    def clear_base_area(self, row, col):
        """Clear a 3x3 area around the base"""
        for r in range(max(0, row - 1), min(self.rows, row + 2)):
            for c in range(max(0, col - 1), min(self.cols, col + 2)):
                self.grid[r][c] = CellType.EMPTY
    
    def get_cell_rect(self, row, col):
        """Get the pygame rect for a grid cell"""
        x = GRID_OFFSET_X + col * GRID_SIZE
        y = GRID_OFFSET_Y + row * GRID_SIZE
        return pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
    
    def get_cell_at_pos(self, pos):
        """Get grid coordinates from screen position"""
        x, y = pos
        col = int((x - GRID_OFFSET_X) // GRID_SIZE)
        row = int((y - GRID_OFFSET_Y) // GRID_SIZE)
        
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None
    
    def is_valid_position(self, row, col):
        """Check if a position is valid and walkable"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        
        cell_type = self.grid[row][col]
        return cell_type in [CellType.EMPTY, CellType.RED_FLAG, CellType.BLUE_FLAG, CellType.RED_BASE, CellType.BLUE_BASE]
    
    def draw(self, screen):
        """Draw the grid on the screen"""
        # Draw grid lines
        for row in range(self.rows + 1):
            y = GRID_OFFSET_Y + row * GRID_SIZE
            pygame.draw.line(screen, DARK_GRAY, 
                           (GRID_OFFSET_X, y), 
                           (GRID_OFFSET_X + self.cols * GRID_SIZE, y))
        
        for col in range(self.cols + 1):
            x = GRID_OFFSET_X + col * GRID_SIZE
            pygame.draw.line(screen, DARK_GRAY, 
                           (x, GRID_OFFSET_Y), 
                           (x, GRID_OFFSET_Y + self.rows * GRID_SIZE))
        
        # Draw grid cells
        for row in range(self.rows):
            for col in range(self.cols):
                cell_rect = self.get_cell_rect(row, col)
                cell_type = self.grid[row][col]
                
                if cell_type == CellType.WALL:
                    pygame.draw.rect(screen, GRAY, cell_rect)
                    pygame.draw.rect(screen, DARK_GRAY, cell_rect, 2)
                
                elif cell_type == CellType.RED_BASE:
                    pygame.draw.rect(screen, RED, cell_rect)
                    pygame.draw.rect(screen, WHITE, cell_rect, 3)
                    
                    # Draw "R" for Red base
                    text_rect = self.red_base_text.get_rect(center=cell_rect.center)
                    screen.blit(self.red_base_text, text_rect)
                
                elif cell_type == CellType.BLUE_BASE:
                    pygame.draw.rect(screen, BLUE, cell_rect)
                    pygame.draw.rect(screen, WHITE, cell_rect, 3)
                    
                    # Draw "B" for Blue base
                    text_rect = self.blue_base_text.get_rect(center=cell_rect.center)
                    screen.blit(self.blue_base_text, text_rect)
                
                elif cell_type == CellType.RED_FLAG:
                    # Draw flag pole
                    pole_rect = pygame.Rect(cell_rect.centerx - 2, cell_rect.top + 5, 4, GRID_SIZE - 10)
                    pygame.draw.rect(screen, WHITE, pole_rect)
                    
                    # Draw flag
                    flag_rect = pygame.Rect(cell_rect.centerx + 2, cell_rect.top + 5, FLAG_SIZE, FLAG_SIZE // 2)
                    pygame.draw.rect(screen, RED, flag_rect)
                    pygame.draw.rect(screen, WHITE, flag_rect, 1)
                
                elif cell_type == CellType.BLUE_FLAG:
                    # Draw flag pole
                    pole_rect = pygame.Rect(cell_rect.centerx - 2, cell_rect.top + 5, 4, GRID_SIZE - 10)
                    pygame.draw.rect(screen, WHITE, pole_rect)
                    
                    # Draw flag
                    flag_rect = pygame.Rect(cell_rect.centerx + 2, cell_rect.top + 5, FLAG_SIZE, FLAG_SIZE // 2)
                    pygame.draw.rect(screen, BLUE, flag_rect)
                    pygame.draw.rect(screen, WHITE, flag_rect, 1)
    
    def respawn_flag(self, flag_type):
        """Respawn a flag at its original position"""
        if flag_type == CellType.RED_FLAG and hasattr(self, 'red_flag_pos'):
            row, col = self.red_flag_pos
            # Only respawn if the position is empty (not occupied by another flag)
            if self.grid[row][col] == CellType.EMPTY:
                self.grid[row][col] = CellType.RED_FLAG
        elif flag_type == CellType.BLUE_FLAG and hasattr(self, 'blue_flag_pos'):
            row, col = self.blue_flag_pos
            # Only respawn if the position is empty (not occupied by another flag)
            if self.grid[row][col] == CellType.EMPTY:
                self.grid[row][col] = CellType.BLUE_FLAG
    
    def is_flag_at_original_position(self, flag_type):
        """Check if a flag is at its original position"""
        if flag_type == CellType.RED_FLAG and hasattr(self, 'red_flag_pos'):
            row, col = self.red_flag_pos
            return self.grid[row][col] == CellType.RED_FLAG
        elif flag_type == CellType.BLUE_FLAG and hasattr(self, 'blue_flag_pos'):
            row, col = self.blue_flag_pos
            return self.grid[row][col] == CellType.BLUE_FLAG
        return False
    
    def reset(self):
        """Reset the grid to initial state"""
        self.generate_symmetric_grid()
        self.place_home_bases()
    
    def reset_flags_and_bases(self):
        """Reset only flags and bases without regenerating walls"""
        # Clear all flags and bases
        for row in range(self.rows):
            for col in range(self.cols):
                cell_type = self.grid[row][col]
                if cell_type in [CellType.RED_FLAG, CellType.BLUE_FLAG, CellType.RED_BASE, CellType.BLUE_BASE]:
                    self.grid[row][col] = CellType.EMPTY
        
        # Place home bases and flags back
        self.place_home_bases() 