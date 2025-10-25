# Window and display constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Grid constants
GRID_SIZE = 40  # Size of each grid cell in pixels
GRID_ROWS = 15
GRID_COLS = 20
GRID_OFFSET_X = (WINDOW_WIDTH - GRID_COLS * GRID_SIZE) // 2
GRID_OFFSET_Y = (WINDOW_HEIGHT - GRID_ROWS * GRID_SIZE) // 2

# Game constants
WALL_PROBABILITY = 0.15  # Probability of a cell being a wall
FLAG_SIZE = 20
BASE_SIZE = 60
CAPTURES_TO_WIN = 3  # Number of flag captures needed to win

# Team constants
TEAM_RED = "red"
TEAM_BLUE = "blue" 