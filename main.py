import pygame
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

pygame.init()

# Window size and frame rate
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600 # Initial screen size
LOGICAL_WIDTH, LOGICAL_HEIGHT = 800, 600 # Fixed game world size
size = width, height = SCREEN_WIDTH, SCREEN_HEIGHT # Current screen size
FPS = 60
RESIZABLE_WINDOW = True

# Scaling and offset variables
game_surface_scale = 1.0
game_surface_offset = (0, 0)
game_surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
BRONZE = (205, 127, 50)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load background images and scale to window size
TRACK_FRONT = pygame.image.load('Images/bg7.png')
TRACK_FRONT = pygame.transform.scale(TRACK_FRONT, size)
TRACK_BACK = pygame.image.load('Images/bg4.png')
TRACK_BACK = pygame.transform.scale(TRACK_BACK, size)

# Convert TRACK_BACK to a NumPy array for faster pixel access
TRACK_BACK_ARRAY = pygame.surfarray.array3d(TRACK_BACK)

# Start/Finish line coordinates and thickness
START_LINE_Y = 250
START_LINE_X_MIN = 0
START_LINE_X_MAX = 110
LINE_THICKNESS = 20

# Define the start/finish line rectangle
START_FINISH_LINE_RECT = pygame.Rect(
    START_LINE_X_MIN,
    START_LINE_Y - LINE_THICKNESS // 2,
    START_LINE_X_MAX - START_LINE_X_MIN,
    LINE_THICKNESS
)

# Fonts (loaded once)
FONT_LARGE = pygame.font.Font('freesansbold.ttf', 24)
FONT_MEDIUM = pygame.font.Font('freesansbold.ttf', 18)
FONT_SMALL = pygame.font.Font('freesansbold.ttf', 14)
FONT_TINY = pygame.font.Font('freesansbold.ttf', 12)


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.hypot(x2 - x1, y2 - y1)


# Precompute rotation matrices for common angles (0-360 degrees)
rotation_matrices = {}
for angle in range(360):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotation_matrices[angle] = (cos_a, sin_a)

def rotate_point(origin, point, angle):
    """Rotate a point around a given origin by an angle using precomputed rotation matrices."""
    ox, oy = origin
    px, py = point
    angle = int(angle % 360)  # Ensure the angle is within 0-359 degrees
    cos_a, sin_a = rotation_matrices[angle]
    qx = ox + cos_a * (px - ox) - sin_a * (py - oy)
    qy = oy + sin_a * (px - ox) + cos_a * (py - oy)
    return qx, qy


def move_point(point, angle, distance):
    """Move a point in a specified direction by a certain distance."""
    x, y = point
    rad = math.radians(-angle % 360)
    x += distance * math.sin(rad)
    y += distance * math.cos(rad)
    return x, y


def leaky_relu(z):
    """Leaky ReLU activation function."""
    return np.where(z > 0, z, z * 0.01)


def rank_selection(cars):
    """Select a parent using rank-based selection."""
    cars = sorted(cars, key=lambda c: c.fitness, reverse=True)
    ranks = np.arange(len(cars), 0, -1)
    total_rank = np.sum(ranks)
    probs = ranks / total_rank
    return np.random.choice(cars, p=probs)


class NeuralNetwork:
    """Neural network class for the car AI."""

    def __init__(self, sizes):
        self.sizes = sizes
        # Initialize weights and biases with He initialization
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]

    def feedforward(self, inputs):
        """Feedforward the inputs through the network."""
        activation = inputs.copy()
        for i in range(len(self.weights) - 1):
            activation = leaky_relu(np.dot(self.weights[i], activation) + self.biases[i])
        activation = np.tanh(np.dot(self.weights[-1], activation) + self.biases[-1])
        return activation

    def mutate(self, mutation_rate, mutation_strength):
        """Mutate the neural network's weights and biases."""
        for i in range(len(self.weights)):
            mutation = np.random.randn(*self.weights[i].shape) * mutation_strength
            mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += mutation * mask
        for i in range(len(self.biases)):
            mutation = np.random.randn(*self.biases[i].shape) * mutation_strength
            mask = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += mutation * mask

    def crossover(self, other):
        """Perform crossover with another neural network."""
        child = NeuralNetwork(self.sizes)
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, self.weights[i], other.weights[i])
        for i in range(len(self.biases)):
            mask = np.random.rand(*self.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask, self.biases[i], other.biases[i])
        return child


class Car:
    """Car class representing a car controlled by a neural network."""

    car_counter = 1  # Class variable to assign unique IDs
    font_small = FONT_TINY  # Class variable for font

    def __init__(self, sizes, generation, color=WHITE, parent1=None, parent2=None):
        self.neural_network = NeuralNetwork(sizes)

        # Car properties
        self.x = 55
        self.y = START_LINE_Y + 100  # Start the car before the start line
        self.prev_position = (self.x, self.y)
        self.center = self.x, self.y
        self.height = 30
        self.width = 16
        self.velocity = 3  # Minimum speed
        self.acceleration = 0
        self.angle = 180
        self.collided = False
        self.color = color

        # Initialize car_rect to avoid errors when accessing it
        self.car_rect = pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

        # Speed limits
        self.min_speed = 2
        self.max_speed = 30  # Increased maximum speed

        # Sensors and inputs
        self.sensor_distances = np.zeros((6, 1)) # 5 sensors, 1 velocity
        self.show_sensors = False  # Toggle sensor lines display
        self.fitness = 0  # Fitness score
        self.show_car_number = True

        # Lap timing
        self.lap_count = 0
        self.lap_times = []  # Store all lap times
        self.lap_time = 0
        self.lap_start_time = None
        self.completed_lap = False
        self.can_count_lap = True  # To prevent multiple lap counts in one crossing

        # Distance tracking
        self.lap1_distance = 0  # Total distance covered in lap 1

        # Assign unique ID to each car
        self.generation = generation
        self.car_number = Car.car_counter
        self.car_id = f'{self.generation}-{self.car_number}'
        Car.car_counter += 1

        # Store parent IDs if they exist
        self.parent1 = parent1
        self.parent2 = parent2

        # Render the car number text only once during initialization
        self.car_number_text = Car.font_small.render(f'{self.car_id}', True, WHITE)

    def set_acceleration(self, accel):
        """Set the acceleration of the car."""
        self.acceleration = accel

    def rotate(self, rotation_angle):
        """Rotate the car by a certain angle."""
        self.angle += rotation_angle
        self.angle %= 360

    def get_max_steering_angle(self):
        """Calculate the maximum steering angle based on current speed."""
        max_steering_angle = 7  # degrees at minimum speed
        min_steering_angle = 1  # degrees at maximum speed
        speed_ratio = (self.velocity - self.min_speed) / (self.max_speed - self.min_speed)
        steering_angle = max_steering_angle - (speed_ratio * (max_steering_angle - min_steering_angle))
        return steering_angle

    def update(self):
        """Update the car's position, sensors, and fitness score."""
        # Apply acceleration
        self.velocity += self.acceleration
        # Enforce speed limits
        self.velocity = max(self.min_speed, min(self.velocity, self.max_speed))

        # Move the car
        self.x, self.y = move_point((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        # Update car corners
        self.update_corners()

        # Update sensors
        self.update_sensors()

        # Calculate distance travelled
        distance = calculate_distance(self.prev_position[0], self.prev_position[1], self.x, self.y)
        if self.lap_count == 0:
            self.lap1_distance += distance
            self.fitness = self.lap1_distance * 0.01
        self.prev_position = (self.x, self.y)

        # Update lap timing
        self.update_lap()

    def apply_friction(self):
        friction_coefficient = 0.05  # Adjust as needed
        if self.velocity > 0:
            self.velocity -= friction_coefficient
            self.velocity = max(self.velocity, self.min_speed)
        elif self.velocity < 0:
            self.velocity += friction_coefficient
            self.velocity = min(self.velocity, self.min_speed)

    def update_corners(self):
        """Update the positions of the car's corners."""
        # Define corners relative to the center
        # Define corners relative to the center
        corners = [
            (-self.width / 2, -self.height / 2),
            (self.width / 2, -self.height / 2),
            (self.width / 2, self.height / 2),
            (-self.width / 2, self.height / 2)
        ]
        angle_rad = math.radians(-self.angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        self.corners = []
        for dx, dy in corners:
            rotated_x = self.center[0] + dx * cos_angle - dy * sin_angle
            rotated_y = self.center[1] + dx * sin_angle + dy * cos_angle
            self.corners.append((rotated_x, rotated_y))

    def update_sensors(self):
        angles = [0, 45, -45, 90, -90]
        max_distance = 120  # Maximum sensor range
        step_size = 2  # Step size for sensor detection
        sensor_count = len(angles)
        # Precompute sensor direction vectors
        radians = np.radians(-(np.array(angles) + self.angle) % 360)
        directions = np.stack((np.sin(radians), np.cos(radians)), axis=1)  # Shape: (5, 2)
        # Compute all possible distances
        steps = np.arange(step_size, max_distance + step_size, step_size)
        # Calculate sensor points
        sensor_points = self.center + directions[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
        # Convert to integer indices
        sensor_indices = sensor_points.astype(int)
        # Check boundaries
        within_bounds = (sensor_indices[:, :, 0] >= 0) & (sensor_indices[:, :, 0] < width) & \
                        (sensor_indices[:, :, 1] >= 0) & (sensor_indices[:, :, 1] < height)
        # Initialize distances
        distances = np.full(sensor_count, max_distance, dtype=float)
        for i in range(sensor_count):
            valid_indices = sensor_indices[i][within_bounds[i]]
            # Find first collision with white pixels
            collision = np.all(TRACK_BACK_ARRAY[valid_indices[:,0], valid_indices[:,1]] == 255, axis=1)
            if np.any(collision):
                first_collision = np.argmax(collision) * step_size + step_size
                distances[i] = min(first_collision, max_distance)
        # Normalize distances
        self.sensor_distances[:sensor_count, 0] = distances / max_distance
        # Normalize velocity
        self.sensor_distances[5] = self.velocity / self.max_speed
    
    def update_car_number_text(self):
        """Update the car number text with appropriate contrast color."""
        text_color = get_contrast_color(self.color)
        self.car_number_text = Car.font_small.render(f'{self.car_id}', True, text_color)
    
    def draw_car_body(self, display):
        """Draws the car body as a rounded rectangle."""
        car_color = self.color  # Car color
        car_rect = pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

        # Draw the car's main body as a rounded rectangle (or just a rectangle if needed)
        pygame.draw.rect(display, car_color, car_rect, border_radius=5)

    def draw(self, display):
        """Draw the car on the display."""
        # Draw the car using shapes
        self.draw_car(display)
        # Display car number above the car
        if self.show_car_number:
            display.blit(self.car_number_text, (self.center[0] - 10, self.center[1] - 40)) 
        
        if self.show_sensors:
            angles = [0, 45, -45, 90, -90]
            max_distance = 120
            for i, angle_offset in enumerate(angles):
                sensor_angle = self.angle + angle_offset
                sensor_end = move_point(self.center, sensor_angle, self.sensor_distances[i][0] * max_distance)
                pygame.draw.line(display, LINE_COLOR, self.center, sensor_end, 2)
        return self.car_rect # return car's rectangle
    
    def draw_car(self, display):
        """Draw the car as a rotated rectangle with wheels."""
        # Create a new surface with per-pixel alpha
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_rect = car_surface.get_rect(center=(self.width / 2, self.height / 2))

        # Draw the car body as a rounded rectangle
        pygame.draw.rect(car_surface, self.color, car_rect, border_radius=5)

        # Draw the wheels
        wheel_color = BLACK
        wheel_width, wheel_height = 4, 10  # Dimensions of each wheel
        wheel_positions = [
            (-3, 2),  # Top right
            (self.width - wheel_width + 3, 2),  # Top left
            (-3, self.height - wheel_height - 2),  # Bottom right
            (self.width - wheel_width + 3, self.height - wheel_height - 2)  # Bottom left
        ]
        for pos in wheel_positions:
            wheel_rect = pygame.Rect(pos, (wheel_width, wheel_height))
            pygame.draw.rect(car_surface, wheel_color, wheel_rect, border_radius=2)

        # Rotate the car surface
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=self.center)

        # Update the car's rectangle for collision detection
        self.car_rect = rotated_rect

        # Blit the rotated car onto the main display
        display.blit(rotated_surface, rotated_rect)

    def feedforward(self):
        """Feedforward the sensor inputs through the neural network."""
        self.output = self.neural_network.feedforward(self.sensor_distances)

    def take_action(self):
        """Take actions based on the neural network's output."""
        steering = self.output[0][0]
        accel_input = self.output[1][0]

        # Apply steering
        steering_angle = steering * self.get_max_steering_angle()
        self.rotate(steering_angle)

        # Apply acceleration
        max_acceleration = 0.15
        self.set_acceleration(accel_input * max_acceleration)

    def collision(self):
        """Check for collisions with the track boundaries."""
        for corner in self.corners:
            x, y = int(corner[0]), int(corner[1])
            if 0 <= x < width and 0 <= y < height:
                # Check if pixel is white (255, 255, 255)
                pixel = TRACK_BACK_ARRAY[x, y]
                if all(pixel == 255):  # Check if all RGB values are 255 (white)
                    return True
            else:
                return True  # Consider out-of-bounds as collision
        return False

    def reset_position(self):
        """Reset the car to the starting position."""
        self.x = 55
        self.y = START_LINE_Y + 100
        self.prev_position = (self.x, self.y)
        self.center = self.x, self.y
        self.velocity = self.min_speed
        self.acceleration = 0
        self.angle = 180
        self.collided = False
        self.fitness = 0
        self.lap_count = 0
        self.lap_times = []
        self.lap_time = 0
        self.lap_start_time = None
        self.completed_lap = False
        self.can_count_lap = True
        self.lap1_distance = 0

    def update_lap(self):
        """Update lap timing and count, adjusting fitness based on lap time."""
        car_point = self.center
        if START_FINISH_LINE_RECT.collidepoint(car_point) and self.can_count_lap:
            self.can_count_lap = False
            # Only increment lap count and record lap time if lap_start_time is set
            if self.lap_start_time is not None:
                self.lap_count += 1
                self.lap_time = time.time() - self.lap_start_time
                self.lap_times.append(self.lap_time)
                self.fastest_lap=min(self.lap_times)

                # Update fitness with lap time consideration
                lap_reward = 300  # Base reward for completing a lap
                fitness_adjustment = (30- (self.lap1_distance * 0.01)) + lap_reward / (self.fastest_lap + 1)  # +1 to avoid division by zero

                if self.lap_count == 1:
                    # First lap completed
                    self.fitness += fitness_adjustment
                else:
                    # Subsequent laps
                    self.fitness += fitness_adjustment
            # Set lap_start_time to current time
            self.lap_start_time = time.time()

        # Reset can_count_lap when car moves away from the line
        elif not START_FINISH_LINE_RECT.collidepoint(car_point) and not self.can_count_lap:
            self.can_count_lap = True


def redraw_game_window(display, cars, player_car, show_info, top_5_lap_times, population_size, generation, show_car_number, lap_count_next_gen, auto_next_generation=False):

    # First, draw the static background only once per frame
    display.blit(TRACK_FRONT, (0, 0))
   
    """Redraw the game window with more precise dirty rect optimization."""
    dirty_rects = []

    alive = 0
    for car in cars:
        if not car.collided:
            prev_rect = car.car_rect.copy()  # Store the previous rectangle
            car.update()

            if car.collision():
                car.collided = True
                car.fitness *= 0.9  # Penalize for collision
            else:
                car.feedforward()
                car.take_action()
            
            # Synchronize the car's show_car_number attribute
            car.show_car_number = show_car_number
            car.update_car_number_text()

            # Draw car and mark the new rectangle as dirty
            car_rect = car.draw(display)
            dirty_rects.append(prev_rect)  # Add previous rect as dirty to clear it
            dirty_rects.append(car_rect)  # Add new rect as dirty to update car

        if not car.collided:
            alive += 1

    # Draw the start/finish line only once
    if generation == 1:
        pygame.draw.rect(display, LINE_COLOR, START_FINISH_LINE_RECT)
        dirty_rects.append(START_FINISH_LINE_RECT)

    if player_car:
        player_car.update_car_number_text()
        player_car.show_car_number = show_car_number
        prev_rect = player_car.car_rect.copy()
        player_car.update()
        if player_car.collision():
            player_car.reset_position()

        # Clear the player's previous position and redraw
        player_car_rect = player_car.draw(display)
        dirty_rects.append(prev_rect)
        dirty_rects.append(player_car_rect)

    if show_info:
        info_rects = display_texts(display, alive, top_5_lap_times, population_size, show_car_number, generation, lap_count_next_gen, auto_next_generation)
        dirty_rects.extend(info_rects)
        

    # Update only the dirty rectangles (areas that have changed)
    pygame.display.flip()

def display_texts(display, alive, top_5_lap_times, population_size, show_car_number, generation, lap_count_next_gen, auto_next_generation=False):
    """Display game information and return a list of rectangles to update."""
    info_x, info_y = 20, 20
    dirty_rects = []  # Initialize an empty list to store dirty rects

    auto_gen_status = "Enabled" if auto_next_generation else "Disabled"
    texts = [
        f'Generation: {generation}',
        f'Alive: {alive}/{population_size}',
        'L - Sensor Lines',
        'A - Player',
        'D - Info',
        'N - Start Next Gen',
        'R - Reset',
        f'Z/X Adjust Laps: {lap_count_next_gen}',
        f'S - Car Number: {"On" if show_car_number else "Off"}',
        f'T - Auto Next Gen: {auto_gen_status}',
        '1-9: Adjust Pop (10-90)'
    ]
    
    # Display the game information and store the areas that need updating
    for i, text in enumerate(texts):
        rendered_text = FONT_SMALL.render(text, True, WHITE)
        rect = display.blit(rendered_text, (info_x + 5, info_y + i * 20))
        dirty_rects.append(rect)  # Append the rectangle to the list

    # Display top 5 fastest lap times
    lap_time_x, lap_time_y = width - 220, 20
    title = FONT_MEDIUM.render('Top 5 Fastest Laps:', True, WHITE)
    title_rect = display.blit(title, (lap_time_x, lap_time_y))
    dirty_rects.append(title_rect)  # Append the title rect

    # Display each lap time and store the rects
    for i, (car_id, lap_time) in enumerate(top_5_lap_times):
        lap_text = f'#{car_id}: {lap_time:.2f}s'
        rendered_lap = FONT_SMALL.render(lap_text, True, WHITE)
        lap_rect = display.blit(rendered_lap, (lap_time_x + 5, lap_time_y + 25 + i * 20))
        dirty_rects.append(lap_rect)  # Append the lap time rect

    return dirty_rects  # Return the list of dirty rectangles

def get_contrast_color(rgb_color):
    """Return black or white depending on contrast with the given color."""
    r, g, b = rgb_color
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    if luminance > 0.5:
        return BLACK  # dark text on light background
    else:
        return WHITE  # light text on dark background

def display_menu(display, cars, best_lap_times, population_size, generation, auto_next_generation=False, auto_proceed=False):
    """Display the menu screen before starting a new generation without resizing."""
    running_menu = True

    # Total table width
    column_widths = [60, 60, 80, 120, 120, 150]  # Column widths
    total_table_width = sum(column_widths)
    stats_x = (width - total_table_width) // 2  # Center the table horizontally
    stats_y = 120  # Adjusted Y position

    # Alive and Dead counts under the title
    alive_count = sum(1 for car in cars if not car.collided)
    dead_count = population_size - alive_count

    # Generate graph surface once
    graph_surface = None
    if best_lap_times:
        graph_surface = plot_best_lap_times(best_lap_times)

    # Track whether to show stats or graph
    show_stats = True  # Initially show the stats table

    # Timing for auto_proceed
    start_time = pygame.time.get_ticks()
    display_duration = 2500  # 5000 milliseconds = 2.5 seconds

    while running_menu:
        # Fill background
        display.fill(BLACK)

        # Title
        title_text = FONT_LARGE.render(f'Generation {generation} Summary', True, WHITE)
        display.blit(title_text, (width // 2 - title_text.get_width() // 2, 20))

        # Alive and Dead counts under the title
        alive_dead_y = 60  # Adjusted Y position under the title
        alive_text = FONT_MEDIUM.render(f'Alive: {alive_count}', True, GREEN)
        dead_text = FONT_MEDIUM.render(f'Dead: {dead_count}', True, (255, 0, 0))
        display.blit(alive_text, (width // 2 - alive_text.get_width() - 10, alive_dead_y))
        display.blit(dead_text, (width // 2 + 10, alive_dead_y))

        if show_stats:
            # Draw background rectangle for stats table
            table_rect_x = stats_x - 20
            table_rect_y = stats_y - 20
            table_rect_width = total_table_width + 40
            row_height = 30  # Adjusted row height
            table_rect_height = row_height * (len(cars[:10]) + 2) + 20
            table_background = pygame.Surface((table_rect_width, table_rect_height))
            table_background.set_alpha(200)  # Transparency
            table_background.fill((30, 30, 30))  # Dark gray background
            display.blit(table_background, (table_rect_x, table_rect_y))

            # Display stats section
            top_10_cars = sorted(cars, key=lambda c: c.fitness, reverse=True)[:10]
            headers = ['Rank', 'Car #', 'Laps', 'Fastest Lap', 'Fitness', 'Parents']

            # Render headers
            header_y_offset = stats_y
            col_x = stats_x
            for i, header in enumerate(headers):
                header_text = FONT_MEDIUM.render(header, True, WHITE)
                col_width = column_widths[i]
                # Center header in the column
                display.blit(header_text, (col_x + col_width // 2 - header_text.get_width() // 2, header_y_offset))
                col_x += col_width

            # Draw horizontal line under headers
            pygame.draw.line(display, WHITE, (stats_x, header_y_offset + row_height - 5),
                             (stats_x + total_table_width, header_y_offset + row_height - 5), 2)

            # Render top 10 cars stats
            for idx, car in enumerate(top_10_cars):
                row_y = header_y_offset + row_height + idx * row_height

                # Draw background rectangle for elite cars
                if car.color != WHITE:
                    row_rect = pygame.Rect(stats_x, row_y, total_table_width, row_height)
                    pygame.draw.rect(display, car.color, row_rect)
                    # Get contrast color for text
                    text_color = get_contrast_color(car.color)
                else:
                    text_color = WHITE  # Default text color for non-elite cars

                stats = [
                    f'{idx + 1}',
                    f'{car.car_id}',
                    f'{car.lap_count}',
                    f'{min(car.lap_times) if car.lap_times else 0:.2f}s',
                    f'{car.fitness:.1f}',
                    f'{car.parent1 or "N/A"} + {car.parent2 or "N/A"}'
                ]
                col_x = stats_x
                for i, stat in enumerate(stats):
                    stat_text = FONT_SMALL.render(stat, True, text_color)
                    col_width = column_widths[i]
                    text_y = row_y + (row_height - stat_text.get_height()) // 2
                    display.blit(stat_text, (col_x + col_width // 2 - stat_text.get_width() // 2,
                                                text_y))
                    col_x += col_width

                # Draw horizontal line after each row
                pygame.draw.line(display, (70, 70, 70),
                                    (stats_x, header_y_offset + row_height * (idx + 2) - 5),
                                    (stats_x + total_table_width, header_y_offset + row_height * (idx + 2) - 5), 1)

                # Draw vertical lines between columns
                col_x = stats_x
                for i in range(len(headers) + 1):
                    pygame.draw.line(display, (70, 70, 70),
                                    (col_x, header_y_offset),
                                    (col_x, header_y_offset + row_height * (len(top_10_cars) + 1)), 1)
                    if i < len(column_widths):
                        col_x += column_widths[i]

        else:
            # Display the lap time graph instead of stats
            if graph_surface:
                graph_width = graph_surface.get_width()
                graph_height = graph_surface.get_height()
                graph_x = (width - graph_width) // 2
                graph_y = stats_y  # Positioned where the table would be
                display.blit(graph_surface, (graph_x, graph_y))

        # Draw the toggle button
        button_text = FONT_MEDIUM.render('Show Graph' if show_stats else 'Show Stats', True, WHITE)
        button_x = (width - button_text.get_width()) // 2
        button_y = stats_y + 370  # Positioned beneath the table/graph
        pygame.draw.rect(display, WHITE, (button_x - 10, button_y - 5,
                                          button_text.get_width() + 20, button_text.get_height() + 10), 2)
        display.blit(button_text, (button_x, button_y))

        # Display options at the bottom
        option_text = FONT_MEDIUM.render('Press N to Start Next Generation or Q to Quit', True, WHITE)
        option_y = height - 50
        display.blit(option_text, (width // 2 - option_text.get_width() // 2, option_y))

        pygame.display.update()

        # Handle auto_proceed timing
        if auto_proceed:
            current_time = pygame.time.get_ticks()
            elapsed_time = current_time - start_time
            if elapsed_time >= display_duration:
                running_menu = False  # Exit the menu after 5 seconds
                break

        # Event handling for menu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    running_menu = False
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Check if the button was clicked
                if button_x - 10 <= mouse_x <= button_x + button_text.get_width() + 10 and \
                   button_y - 5 <= mouse_y <= button_y + button_text.get_height() + 5:
                    show_stats = not show_stats  # Toggle between stats and graph


def plot_best_lap_times(best_lap_times):
    """Plot the best lap times over generations and return as a Pygame surface without resizing the window."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Set the figure size to the desired pixel size
    graph_width = int(width * 0.7)
    graph_height = int(height * 0.6)

    # Calculate figure size in inches
    dpi = 100  # Standard DPI
    fig_width = graph_width / dpi
    fig_height = graph_height / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black', constrained_layout=True)

    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(range(1, len(best_lap_times) + 1), best_lap_times, marker='.', color='red')
    ax.set_title('Best Lap Time per Generation', fontsize=12, color='white')
    ax.set_xlabel('Generation', fontsize=10, color='white')
    ax.set_ylabel('Lap Time (s)', fontsize=10, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=dpi)
    buf.seek(0)
    plt.close(fig)

    # Load the image from the buffer into Pygame
    graph_surface = pygame.image.load(buf).convert_alpha()

    return graph_surface
import pygame
import random
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

pygame.init()

# Window size and frame rate
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600 # Initial screen size
LOGICAL_WIDTH, LOGICAL_HEIGHT = 800, 600 # Fixed game world size
size = width, height = SCREEN_WIDTH, SCREEN_HEIGHT # Current screen size
FPS = 60
RESIZABLE_WINDOW = True

# Scaling and offset variables
game_surface_scale = 1.0
game_surface_offset = (0, 0)
game_surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
LINE_COLOR = (255, 0, 0)
BLACK = (0, 0, 0)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
BRONZE = (205, 127, 50)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load background images and scale to window size
TRACK_FRONT = pygame.image.load('Images/bg7.png')
TRACK_FRONT = pygame.transform.scale(TRACK_FRONT, size)
TRACK_BACK = pygame.image.load('Images/bg4.png')
TRACK_BACK = pygame.transform.scale(TRACK_BACK, size)

# Convert TRACK_BACK to a NumPy array for faster pixel access
TRACK_BACK_ARRAY = pygame.surfarray.array3d(TRACK_BACK)

# Start/Finish line coordinates and thickness
START_LINE_Y = 250
START_LINE_X_MIN = 0
START_LINE_X_MAX = 110
LINE_THICKNESS = 20

# Define the start/finish line rectangle
START_FINISH_LINE_RECT = pygame.Rect(
    START_LINE_X_MIN,
    START_LINE_Y - LINE_THICKNESS // 2,
    START_LINE_X_MAX - START_LINE_X_MIN,
    LINE_THICKNESS
)

# Fonts (loaded once)
FONT_LARGE = pygame.font.Font('freesansbold.ttf', 24)
FONT_MEDIUM = pygame.font.Font('freesansbold.ttf', 18)
FONT_SMALL = pygame.font.Font('freesansbold.ttf', 14)
FONT_TINY = pygame.font.Font('freesansbold.ttf', 12)


def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.hypot(x2 - x1, y2 - y1)


# Precompute rotation matrices for common angles (0-360 degrees)
rotation_matrices = {}
for angle in range(360):
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotation_matrices[angle] = (cos_a, sin_a)

def rotate_point(origin, point, angle):
    """Rotate a point around a given origin by an angle using precomputed rotation matrices."""
    ox, oy = origin
    px, py = point
    angle = int(angle % 360)  # Ensure the angle is within 0-359 degrees
    cos_a, sin_a = rotation_matrices[angle]
    qx = ox + cos_a * (px - ox) - sin_a * (py - oy)
    qy = oy + sin_a * (px - ox) + cos_a * (py - oy)
    return qx, qy


def move_point(point, angle, distance):
    """Move a point in a specified direction by a certain distance."""
    x, y = point
    rad = math.radians(-angle % 360)
    x += distance * math.sin(rad)
    y += distance * math.cos(rad)
    return x, y


def leaky_relu(z):
    """Leaky ReLU activation function."""
    return np.where(z > 0, z, z * 0.01)


def rank_selection(cars):
    """Select a parent using rank-based selection."""
    cars = sorted(cars, key=lambda c: c.fitness, reverse=True)
    ranks = np.arange(len(cars), 0, -1)
    total_rank = np.sum(ranks)
    probs = ranks / total_rank
    return np.random.choice(cars, p=probs)


class NeuralNetwork:
    """Neural network class for the car AI."""

    def __init__(self, sizes):
        self.sizes = sizes
        # Initialize weights and biases with He initialization
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]

    def feedforward(self, inputs):
        """Feedforward the inputs through the network."""
        activation = inputs.copy()
        for i in range(len(self.weights) - 1):
            activation = leaky_relu(np.dot(self.weights[i], activation) + self.biases[i])
        activation = np.tanh(np.dot(self.weights[-1], activation) + self.biases[-1])
        return activation

    def mutate(self, mutation_rate, mutation_strength):
        """Mutate the neural network's weights and biases."""
        for i in range(len(self.weights)):
            mutation = np.random.randn(*self.weights[i].shape) * mutation_strength
            mask = np.random.rand(*self.weights[i].shape) < mutation_rate
            self.weights[i] += mutation * mask
        for i in range(len(self.biases)):
            mutation = np.random.randn(*self.biases[i].shape) * mutation_strength
            mask = np.random.rand(*self.biases[i].shape) < mutation_rate
            self.biases[i] += mutation * mask

    def crossover(self, other):
        """Perform crossover with another neural network."""
        child = NeuralNetwork(self.sizes)
        for i in range(len(self.weights)):
            mask = np.random.rand(*self.weights[i].shape) > 0.5
            child.weights[i] = np.where(mask, self.weights[i], other.weights[i])
        for i in range(len(self.biases)):
            mask = np.random.rand(*self.biases[i].shape) > 0.5
            child.biases[i] = np.where(mask, self.biases[i], other.biases[i])
        return child


class Car:
    """Car class representing a car controlled by a neural network."""

    car_counter = 1  # Class variable to assign unique IDs
    font_small = FONT_TINY  # Class variable for font

    def __init__(self, sizes, generation, color=WHITE, parent1=None, parent2=None):
        self.neural_network = NeuralNetwork(sizes)

        # Car properties
        self.x = 55
        self.y = START_LINE_Y + 100  # Start the car before the start line
        self.prev_position = (self.x, self.y)
        self.center = self.x, self.y
        self.height = 30
        self.width = 16
        self.velocity = 3  # Minimum speed
        self.acceleration = 0
        self.angle = 180
        self.collided = False
        self.color = color

        # Initialize car_rect to avoid errors when accessing it
        self.car_rect = pygame.Rect(self.x - self.width / 2, self.y - self.height / 2, self.width, self.height)

        # Speed limits
        self.min_speed = 2
        self.max_speed = 30  # Increased maximum speed

        # Sensors and inputs
        self.sensor_distances = np.zeros((6, 1)) # 5 sensors, 1 velocity
        self.show_sensors = False  # Toggle sensor lines display
        self.fitness = 0  # Fitness score
        self.show_car_number = True

        # Lap timing
        self.lap_count = 0
        self.lap_times = []  # Store all lap times
        self.lap_time = 0
        self.lap_start_time = None
        self.completed_lap = False
        self.can_count_lap = True  # To prevent multiple lap counts in one crossing

        # Distance tracking
        self.lap1_distance = 0  # Total distance covered in lap 1

        # Assign unique ID to each car
        self.generation = generation
        self.car_number = Car.car_counter
        self.car_id = f'{self.generation}-{self.car_number}'
        Car.car_counter += 1

        # Store parent IDs if they exist
        self.parent1 = parent1
        self.parent2 = parent2

        # Render the car number text only once during initialization
        self.car_number_text = Car.font_small.render(f'{self.car_id}', True, WHITE)

    def set_acceleration(self, accel):
        """Set the acceleration of the car."""
        self.acceleration = accel

    def rotate(self, rotation_angle):
        """Rotate the car by a certain angle."""
        self.angle += rotation_angle
        self.angle %= 360

    def get_max_steering_angle(self):
        """Calculate the maximum steering angle based on current speed."""
        max_steering_angle = 7  # degrees at minimum speed
        min_steering_angle = 1  # degrees at maximum speed
        speed_ratio = (self.velocity - self.min_speed) / (self.max_speed - self.min_speed)
        steering_angle = max_steering_angle - (speed_ratio * (max_steering_angle - min_steering_angle))
        return steering_angle

    def update(self):
        """Update the car's position, sensors, and fitness score."""
        # Apply acceleration
        self.velocity += self.acceleration
        # Enforce speed limits
        self.velocity = max(self.min_speed, min(self.velocity, self.max_speed))

        # Move the car
        self.x, self.y = move_point((self.x, self.y), self.angle, self.velocity)
        self.center = self.x, self.y

        # Update car corners
        self.update_corners()

        # Update sensors
        self.update_sensors()

        # Calculate distance travelled
        distance = calculate_distance(self.prev_position[0], self.prev_position[1], self.x, self.y)
        if self.lap_count == 0:
            self.lap1_distance += distance
            self.fitness = self.lap1_distance * 0.01
        self.prev_position = (self.x, self.y)

        # Update lap timing
        self.update_lap()

    def apply_friction(self):
        friction_coefficient = 0.05  # Adjust as needed
        if self.velocity > 0:
            self.velocity -= friction_coefficient
            self.velocity = max(self.velocity, self.min_speed)
        elif self.velocity < 0:
            self.velocity += friction_coefficient
            self.velocity = min(self.velocity, self.min_speed)

    def update_corners(self):
        """Update the positions of the car's corners."""
        # Define corners relative to the center
        # Define corners relative to the center
        corners = [
            (-self.width / 2, -self.height / 2),
            (self.width / 2, -self.height / 2),
            (self.width / 2, self.height / 2),
            (-self.width / 2, self.height / 2)
        ]
        angle_rad = math.radians(-self.angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        self.corners = []
        for dx, dy in corners:
            rotated_x = self.center[0] + dx * cos_angle - dy * sin_angle
            rotated_y = self.center[1] + dx * sin_angle + dy * cos_angle
            self.corners.append((rotated_x, rotated_y))

    def update_sensors(self):
        angles = [0, 45, -45, 90, -90]
        max_distance = 120  # Maximum sensor range
        step_size = 2  # Step size for sensor detection
        sensor_count = len(angles)
        # Precompute sensor direction vectors
        radians = np.radians(-(np.array(angles) + self.angle) % 360)
        directions = np.stack((np.sin(radians), np.cos(radians)), axis=1)  # Shape: (5, 2)
        # Compute all possible distances
        steps = np.arange(step_size, max_distance + step_size, step_size)
        # Calculate sensor points
        sensor_points = self.center + directions[:, np.newaxis, :] * steps[np.newaxis, :, np.newaxis]
        # Convert to integer indices
        sensor_indices = sensor_points.astype(int)
        # Check boundaries
        within_bounds = (sensor_indices[:, :, 0] >= 0) & (sensor_indices[:, :, 0] < width) & \
                        (sensor_indices[:, :, 1] >= 0) & (sensor_indices[:, :, 1] < height)
        # Initialize distances
        distances = np.full(sensor_count, max_distance, dtype=float)
        for i in range(sensor_count):
            valid_indices = sensor_indices[i][within_bounds[i]]
            # Find first collision with white pixels
            collision = np.all(TRACK_BACK_ARRAY[valid_indices[:,0], valid_indices[:,1]] == 255, axis=1)
            if np.any(collision):
                first_collision = np.argmax(collision) * step_size + step_size
                distances[i] = min(first_collision, max_distance)
        # Normalize distances
        self.sensor_distances[:sensor_count, 0] = distances / max_distance
        # Normalize velocity
        self.sensor_distances[5] = self.velocity / self.max_speed
    
    def update_car_number_text(self):
        """Update the car number text with appropriate contrast color."""
        text_color = get_contrast_color(self.color)
        self.car_number_text = Car.font_small.render(f'{self.car_id}', True, text_color)
    
    def draw_car_body(self, display):
        """Draws the car body as a rounded rectangle."""
        car_color = self.color  # Car color
        car_rect = pygame.Rect(self.x - self.width // 2, self.y - self.height // 2, self.width, self.height)

        # Draw the car's main body as a rounded rectangle (or just a rectangle if needed)
        pygame.draw.rect(display, car_color, car_rect, border_radius=5)

    def draw(self, display):
        """Draw the car on the display."""
        # Draw the car using shapes
        self.draw_car(display)
        # Display car number above the car
        if self.show_car_number:
            display.blit(self.car_number_text, (self.center[0] - 10, self.center[1] - 40)) 
        
        if self.show_sensors:
            angles = [0, 45, -45, 90, -90]
            max_distance = 120
            for i, angle_offset in enumerate(angles):
                sensor_angle = self.angle + angle_offset
                sensor_end = move_point(self.center, sensor_angle, self.sensor_distances[i][0] * max_distance)
                pygame.draw.line(display, LINE_COLOR, self.center, sensor_end, 2)
        return self.car_rect # return car's rectangle
    
    def draw_car(self, display):
        """Draw the car as a rotated rectangle with wheels."""
        # Create a new surface with per-pixel alpha
        car_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        car_rect = car_surface.get_rect(center=(self.width / 2, self.height / 2))

        # Draw the car body as a rounded rectangle
        pygame.draw.rect(car_surface, self.color, car_rect, border_radius=5)

        # Draw the wheels
        wheel_color = BLACK
        wheel_width, wheel_height = 4, 10  # Dimensions of each wheel
        wheel_positions = [
            (-3, 2),  # Top right
            (self.width - wheel_width + 3, 2),  # Top left
            (-3, self.height - wheel_height - 2),  # Bottom right
            (self.width - wheel_width + 3, self.height - wheel_height - 2)  # Bottom left
        ]
        for pos in wheel_positions:
            wheel_rect = pygame.Rect(pos, (wheel_width, wheel_height))
            pygame.draw.rect(car_surface, wheel_color, wheel_rect, border_radius=2)

        # Rotate the car surface
        rotated_surface = pygame.transform.rotate(car_surface, -self.angle)
        rotated_rect = rotated_surface.get_rect(center=self.center)

        # Update the car's rectangle for collision detection
        self.car_rect = rotated_rect

        # Blit the rotated car onto the main display
        display.blit(rotated_surface, rotated_rect)

    def feedforward(self):
        """Feedforward the sensor inputs through the neural network."""
        self.output = self.neural_network.feedforward(self.sensor_distances)

    def take_action(self):
        """Take actions based on the neural network's output."""
        steering = self.output[0][0]
        accel_input = self.output[1][0]

        # Apply steering
        steering_angle = steering * self.get_max_steering_angle()
        self.rotate(steering_angle)

        # Apply acceleration
        max_acceleration = 0.15
        self.set_acceleration(accel_input * max_acceleration)

    def collision(self):
        """Check for collisions with the track boundaries."""
        for corner in self.corners:
            x, y = int(corner[0]), int(corner[1])
            if 0 <= x < width and 0 <= y < height:
                # Check if pixel is white (255, 255, 255)
                pixel = TRACK_BACK_ARRAY[x, y]
                if all(pixel == 255):  # Check if all RGB values are 255 (white)
                    return True
            else:
                return True  # Consider out-of-bounds as collision
        return False

    def reset_position(self):
        """Reset the car to the starting position."""
        self.x = 55
        self.y = START_LINE_Y + 100
        self.prev_position = (self.x, self.y)
        self.center = self.x, self.y
        self.velocity = self.min_speed
        self.acceleration = 0
        self.angle = 180
        self.collided = False
        self.fitness = 0
        self.lap_count = 0
        self.lap_times = []
        self.lap_time = 0
        self.lap_start_time = None
        self.completed_lap = False
        self.can_count_lap = True
        self.lap1_distance = 0

    def update_lap(self):
        """Update lap timing and count, adjusting fitness based on lap time."""
        car_point = self.center
        if START_FINISH_LINE_RECT.collidepoint(car_point) and self.can_count_lap:
            self.can_count_lap = False
            # Only increment lap count and record lap time if lap_start_time is set
            if self.lap_start_time is not None:
                self.lap_count += 1
                self.lap_time = time.time() - self.lap_start_time
                self.lap_times.append(self.lap_time)
                self.fastest_lap=min(self.lap_times)

                # Update fitness with lap time consideration
                lap_reward = 300  # Base reward for completing a lap
                fitness_adjustment = (30- (self.lap1_distance * 0.01)) + lap_reward / (self.fastest_lap + 1)  # +1 to avoid division by zero

                if self.lap_count == 1:
                    # First lap completed
                    self.fitness += fitness_adjustment
                else:
                    # Subsequent laps
                    self.fitness += fitness_adjustment
            # Set lap_start_time to current time
            self.lap_start_time = time.time()

        # Reset can_count_lap when car moves away from the line
        elif not START_FINISH_LINE_RECT.collidepoint(car_point) and not self.can_count_lap:
            self.can_count_lap = True


def redraw_game_window(display, cars, player_car, show_info, top_5_lap_times, population_size, generation, show_car_number, lap_count_next_gen, auto_next_generation=False):

    # First, draw the static background only once per frame
    display.blit(TRACK_FRONT, (0, 0))
   
    """Redraw the game window with more precise dirty rect optimization."""
    dirty_rects = []

    alive = 0
    for car in cars:
        if not car.collided:
            prev_rect = car.car_rect.copy()  # Store the previous rectangle
            car.update()

            if car.collision():
                car.collided = True
                car.fitness *= 0.9  # Penalize for collision
            else:
                car.feedforward()
                car.take_action()
            
            # Synchronize the car's show_car_number attribute
            car.show_car_number = show_car_number
            car.update_car_number_text()

            # Draw car and mark the new rectangle as dirty
            car_rect = car.draw(display)
            dirty_rects.append(prev_rect)  # Add previous rect as dirty to clear it
            dirty_rects.append(car_rect)  # Add new rect as dirty to update car

        if not car.collided:
            alive += 1

    # Draw the start/finish line only once
    if generation == 1:
        pygame.draw.rect(display, LINE_COLOR, START_FINISH_LINE_RECT)
        dirty_rects.append(START_FINISH_LINE_RECT)

    if player_car:
        player_car.update_car_number_text()
        player_car.show_car_number = show_car_number
        prev_rect = player_car.car_rect.copy()
        player_car.update()
        if player_car.collision():
            player_car.reset_position()

        # Clear the player's previous position and redraw
        player_car_rect = player_car.draw(display)
        dirty_rects.append(prev_rect)
        dirty_rects.append(player_car_rect)

    if show_info:
        info_rects = display_texts(display, alive, top_5_lap_times, population_size, show_car_number, generation, lap_count_next_gen, auto_next_generation)
        dirty_rects.extend(info_rects)
        

    # Update only the dirty rectangles (areas that have changed)
    pygame.display.flip()

def display_texts(display, alive, top_5_lap_times, population_size, show_car_number, generation, lap_count_next_gen, auto_next_generation=False):
    """Display game information and return a list of rectangles to update."""
    info_x, info_y = 20, 20
    dirty_rects = []  # Initialize an empty list to store dirty rects

    auto_gen_status = "Enabled" if auto_next_generation else "Disabled"
    texts = [
        f'Generation: {generation}',
        f'Alive: {alive}/{population_size}',
        'L - Lines',
        'A - Player',
        'D - Info',
        'N - Start Next Gen',
        'R - Reset',
        f'Z/X Adjust Laps: {lap_count_next_gen}',
        f'S - Car Number: {"On" if show_car_number else "Off"}',
        f'T - Auto Next Gen: {auto_gen_status}',
        '1-9: Adjust Pop (10-5000)'
    ]
    
    # Display the game information and store the areas that need updating
    for i, text in enumerate(texts):
        rendered_text = FONT_SMALL.render(text, True, WHITE)
        rect = display.blit(rendered_text, (info_x + 5, info_y + i * 20))
        dirty_rects.append(rect)  # Append the rectangle to the list

    # Display top 5 fastest lap times
    lap_time_x, lap_time_y = width - 220, 20
    title = FONT_MEDIUM.render('Top 5 Fastest Laps:', True, WHITE)
    title_rect = display.blit(title, (lap_time_x, lap_time_y))
    dirty_rects.append(title_rect)  # Append the title rect

    # Display each lap time and store the rects
    for i, (car_id, lap_time) in enumerate(top_5_lap_times):
        lap_text = f'#{car_id}: {lap_time:.2f}s'
        rendered_lap = FONT_SMALL.render(lap_text, True, WHITE)
        lap_rect = display.blit(rendered_lap, (lap_time_x + 5, lap_time_y + 25 + i * 20))
        dirty_rects.append(lap_rect)  # Append the lap time rect

    return dirty_rects  # Return the list of dirty rectangles

def get_contrast_color(rgb_color):
    """Return black or white depending on contrast with the given color."""
    r, g, b = rgb_color
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    if luminance > 0.5:
        return BLACK  # dark text on light background
    else:
        return WHITE  # light text on dark background

def display_menu(display, cars, best_lap_times, population_size, generation, auto_next_generation=False, auto_proceed=False):
    """Display the menu screen before starting a new generation without resizing."""
    running_menu = True

    # Total table width
    column_widths = [60, 60, 80, 120, 120, 150]  # Column widths
    total_table_width = sum(column_widths)
    stats_x = (width - total_table_width) // 2  # Center the table horizontally
    stats_y = 120  # Adjusted Y position

    # Alive and Dead counts under the title
    alive_count = sum(1 for car in cars if not car.collided)
    dead_count = population_size - alive_count

    # Generate graph surface once
    graph_surface = None
    if best_lap_times:
        graph_surface = plot_best_lap_times(best_lap_times)

    # Track whether to show stats or graph
    show_stats = True  # Initially show the stats table

    # Timing for auto_proceed
    start_time = pygame.time.get_ticks()
    display_duration = 2500  # 5000 milliseconds = 2.5 seconds

    while running_menu:
        # Fill background
        display.fill(BLACK)

        # Title
        title_text = FONT_LARGE.render(f'Generation {generation} Summary', True, WHITE)
        display.blit(title_text, (width // 2 - title_text.get_width() // 2, 20))

        # Alive and Dead counts under the title
        alive_dead_y = 60  # Adjusted Y position under the title
        alive_text = FONT_MEDIUM.render(f'Alive: {alive_count}', True, GREEN)
        dead_text = FONT_MEDIUM.render(f'Dead: {dead_count}', True, (255, 0, 0))
        display.blit(alive_text, (width // 2 - alive_text.get_width() - 10, alive_dead_y))
        display.blit(dead_text, (width // 2 + 10, alive_dead_y))

        if show_stats:
            # Draw background rectangle for stats table
            table_rect_x = stats_x - 20
            table_rect_y = stats_y - 20
            table_rect_width = total_table_width + 40
            row_height = 30  # Adjusted row height
            table_rect_height = row_height * (len(cars[:10]) + 2) + 20
            table_background = pygame.Surface((table_rect_width, table_rect_height))
            table_background.set_alpha(200)  # Transparency
            table_background.fill((30, 30, 30))  # Dark gray background
            display.blit(table_background, (table_rect_x, table_rect_y))

            # Display stats section
            top_10_cars = sorted(cars, key=lambda c: c.fitness, reverse=True)[:10]
            headers = ['Rank', 'Car #', 'Laps', 'Fastest Lap', 'Fitness', 'Parents']

            # Render headers
            header_y_offset = stats_y
            col_x = stats_x
            for i, header in enumerate(headers):
                header_text = FONT_MEDIUM.render(header, True, WHITE)
                col_width = column_widths[i]
                # Center header in the column
                display.blit(header_text, (col_x + col_width // 2 - header_text.get_width() // 2, header_y_offset))
                col_x += col_width

            # Draw horizontal line under headers
            pygame.draw.line(display, WHITE, (stats_x, header_y_offset + row_height - 5),
                             (stats_x + total_table_width, header_y_offset + row_height - 5), 2)

            # Render top 10 cars stats
            for idx, car in enumerate(top_10_cars):
                row_y = header_y_offset + row_height + idx * row_height

                # Draw background rectangle for elite cars
                if car.color != WHITE:
                    row_rect = pygame.Rect(stats_x, row_y, total_table_width, row_height)
                    pygame.draw.rect(display, car.color, row_rect)
                    # Get contrast color for text
                    text_color = get_contrast_color(car.color)
                else:
                    text_color = WHITE  # Default text color for non-elite cars

                stats = [
                    f'{idx + 1}',
                    f'{car.car_id}',
                    f'{car.lap_count}',
                    f'{min(car.lap_times) if car.lap_times else 0:.2f}s',
                    f'{car.fitness:.1f}',
                    f'{car.parent1 or "N/A"} + {car.parent2 or "N/A"}'
                ]
                col_x = stats_x
                for i, stat in enumerate(stats):
                    stat_text = FONT_SMALL.render(stat, True, text_color)
                    col_width = column_widths[i]
                    text_y = row_y + (row_height - stat_text.get_height()) // 2
                    display.blit(stat_text, (col_x + col_width // 2 - stat_text.get_width() // 2,
                                                text_y))
                    col_x += col_width

                # Draw horizontal line after each row
                pygame.draw.line(display, (70, 70, 70),
                                    (stats_x, header_y_offset + row_height * (idx + 2) - 5),
                                    (stats_x + total_table_width, header_y_offset + row_height * (idx + 2) - 5), 1)

                # Draw vertical lines between columns
                col_x = stats_x
                for i in range(len(headers) + 1):
                    pygame.draw.line(display, (70, 70, 70),
                                    (col_x, header_y_offset),
                                    (col_x, header_y_offset + row_height * (len(top_10_cars) + 1)), 1)
                    if i < len(column_widths):
                        col_x += column_widths[i]

        else:
            # Display the lap time graph instead of stats
            if graph_surface:
                graph_width = graph_surface.get_width()
                graph_height = graph_surface.get_height()
                graph_x = (width - graph_width) // 2
                graph_y = stats_y  # Positioned where the table would be
                display.blit(graph_surface, (graph_x, graph_y))

        # Draw the toggle button
        button_text = FONT_MEDIUM.render('Show Graph' if show_stats else 'Show Stats', True, WHITE)
        button_x = (width - button_text.get_width()) // 2
        button_y = stats_y + 370  # Positioned beneath the table/graph
        pygame.draw.rect(display, WHITE, (button_x - 10, button_y - 5,
                                          button_text.get_width() + 20, button_text.get_height() + 10), 2)
        display.blit(button_text, (button_x, button_y))

        # Display options at the bottom
        option_text = FONT_MEDIUM.render('Press N to Start Next Generation or Q to Quit', True, WHITE)
        option_y = height - 50
        display.blit(option_text, (width // 2 - option_text.get_width() // 2, option_y))

        pygame.display.update()

        # Handle auto_proceed timing
        if auto_proceed:
            current_time = pygame.time.get_ticks()
            elapsed_time = current_time - start_time
            if elapsed_time >= display_duration:
                running_menu = False  # Exit the menu after 5 seconds
                break

        # Event handling for menu
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    running_menu = False
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Check if the button was clicked
                if button_x - 10 <= mouse_x <= button_x + button_text.get_width() + 10 and \
                   button_y - 5 <= mouse_y <= button_y + button_text.get_height() + 5:
                    show_stats = not show_stats  # Toggle between stats and graph


def plot_best_lap_times(best_lap_times):
    """Plot the best lap times over generations and return as a Pygame surface without resizing the window."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend

    # Set the figure size to the desired pixel size
    graph_width = int(width * 0.7)
    graph_height = int(height * 0.6)

    # Calculate figure size in inches
    dpi = 100  # Standard DPI
    fig_width = graph_width / dpi
    fig_height = graph_height / dpi

    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor='black', constrained_layout=True)

    ax = fig.add_subplot(111, facecolor='black')
    ax.plot(range(1, len(best_lap_times) + 1), best_lap_times, marker='.', color='red')
    ax.set_title('Best Lap Time per Generation', fontsize=12, color='white')
    ax.set_xlabel('Generation', fontsize=10, color='white')
    ax.set_ylabel('Lap Time (s)', fontsize=10, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Save the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=dpi)
    buf.seek(0)
    plt.close(fig)

    # Load the image from the buffer into Pygame
    graph_surface = pygame.image.load(buf).convert_alpha()

    return graph_surface


def main():
    # Initialize game display, cached background, clock
    game_display = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    cached_background = TRACK_FRONT.copy()

    # Initialize variables
    population_size = 40
    input_size = 6
    hidden_sizes = [10, 6]
    output_size = 2  # Steering and Acceleration
    sizes = [input_size] + hidden_sizes + [output_size]
    Car.car_counter = 1 # Reset car counter to 1 at the start of each generation
    generation = 1
    cars = [Car(sizes, generation) for _ in range(population_size)]
    player_car = None
    show_info = True
    show_sensors = False  # Sensors are hidden by default
    show_car_number = True
    auto_next_generation = False

    # Genetic Algorithm parameters
    initial_mutation_rate = 0.35
    initial_mutation_strength = 0.15
    decay_factor = 0.995  # Decay rate per generation
    decay_factor_strength = 0.998

    # For tracking best lap times per generation
    best_lap_times = []

    # Initialize lap_count_next_gen before the main loop
    lap_count_next_gen = 2  # Starting with 2 laps

    running = True
    while running:
        # Reset top 5 lap times list
        top_5_lap_times = []
        simulation_running = True  # Flag to control the simulation loop

        # Calculate adaptive mutation parameters
        mutation_rate = initial_mutation_rate * (decay_factor ** generation)
        mutation_strength = initial_mutation_strength * (decay_factor_strength ** generation)

        # Main simulation loop
        while simulation_running:
            clock.tick(FPS)
            # Check if all cars have collided
            all_collided = all(car.collided for car in cars)
            
            # Check if any car has completed x laps
            any_completed_two_laps = any(car.lap_count >= lap_count_next_gen for car in cars)
            
            # Determine whether to terminate the simulation
            if all_collided:
                simulation_running = False
            elif auto_next_generation and any_completed_two_laps:
                simulation_running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    quit()
                # Keypress events
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_n:
                        simulation_running = False  # Exit simulation loop
                    elif event.key == pygame.K_l:
                        show_sensors = not show_sensors
                        for car in cars:
                            car.show_sensors = show_sensors
                            # Adjust population size based on key presses 1-9
                    elif event.key == pygame.K_1:
                        population_size = 10
                    elif event.key == pygame.K_2:
                        population_size = 25
                    elif event.key == pygame.K_3:
                        population_size = 50
                    elif event.key == pygame.K_4:
                        population_size = 100
                    elif event.key == pygame.K_5:
                        population_size = 200
                    elif event.key == pygame.K_6:
                        population_size = 500
                    elif event.key == pygame.K_7:
                        population_size = 750
                    elif event.key == pygame.K_8:
                        population_size = 1000
                    elif event.key == pygame.K_9:
                        population_size = 2500
                    elif event.key == pygame.K_a:
                        if player_car:
                            player_car = None
                        else:
                            player_car = Car(sizes, generation, color=GREEN)
                    elif event.key == pygame.K_x:
                        lap_count_next_gen = lap_count_next_gen + 1
                    elif event.key == pygame.K_z:
                        lap_count_next_gen = max(1, lap_count_next_gen -1)
                        
                    elif event.key == pygame.K_d:
                        show_info = not show_info
                    elif event.key == pygame.K_r:
                        generation = 1
                        Car.car_counter = 1  # Reset car counter
                        cars = [Car(sizes, generation) for _ in range(population_size)]
                        for car in cars:
                            car.reset_position()
                        if player_car:
                            player_car.reset_position()
                        best_lap_times = []
                        simulation_running = False
                        break  # Exit to reinitialize simulation
                    elif event.key == pygame.K_s:
                        show_car_number = not show_car_number
                    elif event.key == pygame.K_t:  # New key to toggle auto progression
                        auto_next_generation = not auto_next_generation
                        if not auto_next_generation:
                            auto_proceed = False   
                        status = "Enabled" if auto_next_generation else "Disabled"
                        print(f"Automatic Next Generation: {status}")  # Optional: Console 
           
            keys = pygame.key.get_pressed()
            if player_car:
                if keys[pygame.K_LEFT]:
                    player_car.rotate(-5)
                if keys[pygame.K_RIGHT]:
                    player_car.rotate(5)
                if keys[pygame.K_UP]:
                    player_car.set_acceleration(0.2)
                else:
                    player_car.set_acceleration(0)
                if keys[pygame.K_DOWN]:
                    player_car.set_acceleration(-0.2)

            # Collect lap times for top 5 fastest laps
            lap_times = []
            for car in cars:
                if car.lap_times:
                    fastest_lap = min(car.lap_times)
                    lap_times.append((car.car_id, fastest_lap))
            lap_times.sort(key=lambda x: x[1])
            top_5_lap_times = lap_times[:5]

            redraw_game_window(game_display, cars, player_car, show_info, top_5_lap_times, population_size, generation, show_car_number, lap_count_next_gen, auto_next_generation=auto_next_generation)

        # After simulation ends, display menu
        # Collect best lap time for the generation
        generation_best_lap = min((min(car.lap_times) for car in cars if car.lap_times), default=None)
        if generation_best_lap is not None:
            best_lap_times.append(generation_best_lap)

        # Determine if auto_proceed should be activated
        if auto_next_generation and any_completed_two_laps:
            auto_proceed = True
        else:
            auto_proceed = False

        display_menu(game_display, cars, best_lap_times, population_size, generation, auto_next_generation=auto_next_generation, auto_proceed=auto_proceed)

        # Breed next generation
        generation += 1

        # Define elite colors
        elite_colors = [GOLD, SILVER, BRONZE, RED, BLUE, YELLOW, GREEN]
        elite_count = min(len(elite_colors), max(4, int(0.1 * population_size)))

        # Ensure all fitness values are non-negative
        for car in cars:
            if car.fitness < 0:
                car.fitness = 0

        # Sort cars by fitness
        cars.sort(key=lambda c: c.fitness, reverse=True)

        # Breeding on top 15% of cars
        top_performers = cars[:max(4, int(0.15 * population_size))]
        # Reset car counter to 1 at the start of each generation
        Car.car_counter = 1

        next_gen = []

        # Calculate adaptive mutation parameters
        mutation_rate = initial_mutation_rate * (decay_factor ** generation)
        mutation_strength = initial_mutation_strength * (decay_factor ** generation)
        
        # Generate new cars using rank selection
        while len(next_gen) < population_size - elite_count:
            parent1 = rank_selection(top_performers)
            parent2 = rank_selection(top_performers)
            # Crossover neural networks
            child_nn = parent1.neural_network.crossover(parent2.neural_network)
            # Mutate the child neural network
            child_nn.mutate(mutation_rate, mutation_strength)
            # Create a new car with the child neural network
            child_car = Car(sizes, generation, parent1=parent1.car_id, parent2=parent2.car_id)
            child_car.neural_network = child_nn
            next_gen.append(child_car)

        # Carry over elite cars unchanged except new color
        for i in range(elite_count):
            elite_car = top_performers[i]
            elite_car.reset_position()
            elite_car.fitness = 0
            elite_car.collided = False
            elite_car.color = elite_colors[i] # Assign based on rank
            elite_car.update_car_number_text()  # Update the car number text to match the new color
            next_gen.append(elite_car)

        cars = next_gen

        # Reset player car if present
        if player_car:
            player_car.reset_position()

    pygame.quit()


if __name__ == "__main__":
    main()