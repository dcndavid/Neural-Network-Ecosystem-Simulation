import pygame
import random
import noise
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from PIL import Image
import cycler
import kiwisolver
from dateutil import parser

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480
MAP_WIDTH, MAP_HEIGHT = 1600, 1200
TILE_SIZE = 20
FPS = 60
REPRODUCTION_THRESHOLD = 100
HUNGER_THRESHOLD = 600
THIRST_THRESHOLD = 600
PLANT_PROBABILITY = 0.0001  # Probability that a land tile will have a plant initially
REGROWTH_TIME = 500  # Time steps for plant regrowth
MUTATION_PROBABILITY = 0.3  # Probability of gene mutation during reproduction
ADULT_AGE = 100  # Age at which entities can start seeking mates

# Colors
WHITE = (255, 255, 255)
BLUE = (135, 243, 255)
GREEN = (148, 222, 140)
BROWN = (77, 64, 64)
DARK_GREEN = (11, 181, 139)
SAND = (255, 227, 179)
GRAY = (235, 234, 232)
RED = (237, 84, 78)
BLACK = (0, 0, 0)
YELLOW = (255, 201, 115)
ORANGE = (255, 111, 0)
PURPLE = (128, 0, 128)  # Sensory radius color
CARNIVORE_COLOR = (255, 0, 0)  # Red
OMNIVORE_COLOR = (255, 165, 0)  # Yellowy Orange

# Set up display
win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ecosystem Simulation")

# Font
font = pygame.font.SysFont(None, 24)

# Clock
clock = pygame.time.Clock()

def get_reward(entity):
    reward = 0
    if entity.hunger >= entity.hunger_threshold:
        reward += 1
    if entity.thirst >= entity.thirst_threshold:
        reward += 1
    if entity.reproductive_urge >= entity.reproduction_threshold:
        reward += 1
    if entity.hunger < entity.hunger_threshold * 0.2:  # Penalize low hunger
        reward -= 5
    if entity.thirst < entity.thirst_threshold * 0.2:  # Penalize low thirst
        reward -= 5
    if not entity.alive:
        reward -= 10
    return reward

# Environment setup
class Tile:
    def __init__(self, x, y, tile_type):
        self.x = x
        self.y = y
        self.type = tile_type
        self.regrowth_timer = 0

    def update(self):
        if self.type == "land" and self.regrowth_timer > 0:
            self.regrowth_timer -= 1

    def draw(self, win, offset_x, offset_y, scale):
        if self.type == "water":
            color = BLUE
        elif self.type == "land":
            color = GREEN
        elif self.type == "tree":
            color = BROWN
        elif self.type == "sand":
            color = SAND
        pygame.draw.rect(win, color,
                         ((self.x * TILE_SIZE - offset_x) * scale,
                          (self.y * TILE_SIZE - offset_y) * scale,
                          TILE_SIZE * scale,
                          TILE_SIZE * scale))

class Plant:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, win, offset_x, offset_y, scale):
        pygame.draw.circle(win, DARK_GREEN,
                           (int((self.x * TILE_SIZE + TILE_SIZE / 2 - offset_x) * scale),
                            int((self.y * TILE_SIZE + TILE_SIZE / 2 - offset_y) * scale)),
                           int(TILE_SIZE / 4 * scale))

class Entity:
    def __init__(self, x, y, tiles, genes=None, generation=1, parent_genes=None):
        self.x, self.y = self.find_valid_position(int(x), int(y), tiles)
        self.target_x, self.target_y = self.x, self.y
        self.hunger = genes['hunger'] if genes else HUNGER_THRESHOLD
        self.thirst = genes['thirst'] if genes else THIRST_THRESHOLD
        self.reproductive_urge = genes['reproductive_urge'] if genes else 0
        self.speed = genes['speed'] if genes else random.uniform(1.0, 4.0)
        self.sensory_radius = genes['sensory_radius'] if genes else random.randint(1, 10)
        self.reproduction_threshold = genes['reproduction_threshold'] if genes else REPRODUCTION_THRESHOLD
        self.hunger_threshold = genes['hunger_threshold'] if genes else HUNGER_THRESHOLD
        self.thirst_threshold = genes['thirst_threshold'] if genes else THIRST_THRESHOLD
        self.diet_preference = genes['diet_preference'] if genes else 'herbivore'
        self.offspring_count = genes['offspring_count'] if genes else random.randint(1, 3)
        self.size_gene = genes['size'] if genes else random.uniform(0.5, 1.5)  # Size gene
        self.alive = True
        self.gender = random.choice(['male', 'female'])
        self.gestation_duration = genes['gestation_duration'] if genes and self.gender == 'female' else random.randint(100, 300)
        self.gestation_timer = genes['gestation_timer'] if genes and self.gender == 'female' else 0  # Timer for gestation period
        self.child_genes = genes.copy() if genes else None
        self.generation = generation
        self.parent_genes = parent_genes if parent_genes else {}
        self.unimpressed_females = set()
        self.state = 'idle'  # Current state of the entity
        self.direction = (random.uniform(-1, 1), random.uniform(-1, 1))  # Initial random direction
        self.steps_since_direction_change = 0  # Steps taken in the current direction
        self.pregnant = False if self.gender == 'female' else None  # Pregnancy state for females
        self.age = 0  # Age of the entity
        self.old_age_threshold = genes['old_age_threshold'] if genes else random.randint(400, 1000)
        self.size = 0.2  # Starting size, will grow as the entity ages

    def find_valid_position(self, x, y, tiles):
        while tiles[int(y)][int(x)].type == "water" or tiles[int(y)][int(x)].type == "tree":
            x, y = random.randint(0, MAP_WIDTH // TILE_SIZE - 1), random.randint(0, MAP_HEIGHT // TILE_SIZE - 1)
        return x, y

    def is_valid_move(self, x, y, tiles):
        if 0 <= int(x) < MAP_WIDTH // TILE_SIZE and 0 <= int(y) < MAP_HEIGHT // TILE_SIZE:
            return tiles[int(y)][int(x)].type in ["land", "sand"]
        return False

    def check_adjacent_water(self, tiles):
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if 0 <= int(self.x + dx) < MAP_WIDTH // TILE_SIZE and 0 <= int(self.y + dy) < MAP_HEIGHT // TILE_SIZE:
                    if tiles[int(self.y + dy)][int(self.x + dx)].type == "water":
                        return True
        return False

    def move(self, tiles, food_sources, mates):
        if not self.alive:
            self.state = 'dead'
            return

        self.hunger -= 1
        self.thirst -= 1
        self.age += 1

        if self.age >= self.old_age_threshold:
            self.alive = False
            self.state = 'dead'
            return

        # Grow as the entity ages
        if self.age < ADULT_AGE:
            self.size = 0.2 + 0.8 * (self.age / ADULT_AGE)
        else:
            self.size = 1.0

        if not self.pregnant:
            if self.reproductive_urge < self.reproduction_threshold:
                self.reproductive_urge += 1

        if self.hunger <= 0 or self.thirst <= 0:
            self.alive = False
            self.state = 'dead'
            return

        self.take_action(tiles, food_sources, mates)

        # Smooth movement interpolation
        self.x += (self.target_x - self.x) * 0.1
        self.y += (self.target_y - self.y) * 0.1

    def take_action(self, tiles, food_sources, mates):
        # Define hardcoded rules for action selection
        if self.hunger < self.hunger_threshold * 0.2:
            self.seek_food(tiles, food_sources)
        elif self.thirst < self.thirst_threshold * 0.2:
            self.seek_water(tiles)
        elif self.reproductive_urge >= self.reproduction_threshold:
            self.seek_mate(mates, tiles)
        else:
            self.random_walk(tiles)

    def seek_food(self, tiles, food_sources):
        target_food = None
        min_distance = float('inf')
        for food in food_sources:
            distance = np.sqrt((self.x - food.x) ** 2 + (self.y - food.y) ** 2)
            if distance < min_distance and distance <= self.sensory_radius:
                min_distance = distance
                target_food = food

        if target_food:
            if min_distance < 1:
                food_sources.remove(target_food)
                tiles[int(target_food.y)][int(target_food.x)].regrowth_timer = REGROWTH_TIME
                self.hunger = self.hunger_threshold
                self.state = 'idle'  # Reset state after eating
            else:
                direction_x = (target_food.x - self.x) / min_distance
                direction_y = (target_food.y - self.y) / min_distance
                new_x = self.x + direction_x * self.speed
                new_y = self.y + direction_y * self.speed
                if self.is_valid_move(new_x, new_y, tiles):
                    self.target_x, self.target_y = new_x, new_y
                    self.state = 'seeking food'

    def seek_water(self, tiles):
        if self.check_adjacent_water(tiles):
            self.thirst = self.thirst_threshold
            self.state = 'idle'  # Reset state after drinking
        else:
            for y in range(max(0, int(self.y - self.sensory_radius)), min(MAP_HEIGHT // TILE_SIZE, int(self.y + self.sensory_radius))):
                for x in range(max(0, int(self.x - self.sensory_radius)), min(MAP_WIDTH // TILE_SIZE, int(self.x + self.sensory_radius))):
                    if tiles[int(y)][int(x)].type == "water":
                        distance = np.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
                        if distance <= self.sensory_radius:
                            direction_x = (x - self.x) / distance
                            direction_y = (y - self.y) / distance
                            new_x = self.x + direction_x * self.speed
                            new_y = self.y + direction_y * self.speed
                            if self.is_valid_move(new_x, new_y, tiles):
                                self.target_x, self.target_y = new_x, new_y
                                self.state = 'seeking water'
                                return

    def seek_mate(self, mates, tiles):
        if self.reproductive_urge >= self.reproduction_threshold:
            if self.gender == 'male':
                for mate in mates:
                    if mate != self and mate.alive and mate.reproductive_urge >= self.reproduction_threshold and mate.gender == 'female':
                        distance = np.sqrt((self.x - mate.x) ** 2 + (self.y - mate.y) ** 2)
                        if distance <= self.sensory_radius:
                            self.reproductive_urge = 0
                            mate.reproductive_urge = 0
                            mate.gestation_timer = mate.gestation_duration  # Start gestation period
                            mate.child_genes = self.combine_genes(self, mate)
                            mate.parent_genes = {'mother': mate.get_genes(), 'father': self.get_genes()}  # Store parent genes
                            mate.pregnant = True  # Mark the female as pregnant
                            self.state = 'idle'  # Reset state after mating
                            return
            elif self.gender == 'female':
                for mate in mates:
                    if mate != self and mate.alive and mate.reproductive_urge >= self.reproduction_threshold and mate.gender == 'male':
                        distance = np.sqrt((self.x - mate.x) ** 2 + (self.y - mate.y) ** 2)
                        if distance <= self.sensory_radius:
                            self.reproductive_urge = 0
                            mate.reproductive_urge = 0
                            self.gestation_timer = self.gestation_duration  # Start gestation period
                            self.child_genes = self.combine_genes(self, mate)
                            self.parent_genes = {'mother': self.get_genes(), 'father': mate.get_genes()}  # Store parent genes
                            self.pregnant = True  # Mark the female as pregnant
                            self.state = 'idle'  # Reset state after mating
                            return
        if self.state == 'idle':
            self.random_walk(tiles)

    def random_walk(self, tiles):
        self.steps_since_direction_change += 1
        if self.steps_since_direction_change > 50:  # Change direction every 50 steps
            self.direction = (random.uniform(-1, 1), random.uniform(-1, 1))
            self.steps_since_direction_change = 0

        new_x = self.x + self.direction[0] * self.speed
        new_y = self.y + self.direction[1] * self.speed

        if self.is_valid_move(new_x, new_y, tiles):
            self.target_x, self.target_y = new_x, new_y
            self.state = 'wandering'

    def update_gestation(self, entities, tiles):
        if self.gestation_timer > 0:
            self.gestation_timer -= 1
            if self.gestation_timer == 0:
                num_offspring = max(1, int(self.offspring_count + random.uniform(-0.5, 0.5)))  # Random variation in offspring count
                for _ in range(num_offspring):
                    new_entity = self.__class__(*self.find_valid_position_adjacent(self.x, self.y, tiles), tiles, self.child_genes, generation=self.generation + 1, parent_genes=self.parent_genes)
                    entities.append(new_entity)
                self.child_genes = None
                self.state = 'giving birth'
                self.pregnant = False  # Mark the female as no longer pregnant
                self.reproductive_urge = 0  # Reset the reproductive urge

    def find_valid_position_adjacent(self, x, y, tiles):
        possible_positions = [(int(x + dx), int(y + dy)) for dx in range(-1, 2) for dy in range(-1, 2)]
        random.shuffle(possible_positions)
        for pos in possible_positions:
            if 0 <= pos[0] < MAP_WIDTH // TILE_SIZE and 0 <= pos[1] < MAP_HEIGHT // TILE_SIZE:
                if tiles[pos[1]][pos[0]].type in ["land", "sand"]:
                    return pos
        return int(x), int(y)

    def get_genes(self):
        return {
            'hunger': self.hunger,
            'thirst': self.thirst,
            'reproductive_urge': self.reproductive_urge,
            'speed': self.speed,
            'sensory_radius': self.sensory_radius,
            'gestation_duration': self.gestation_duration,
            'gestation_timer': self.gestation_timer,
            'reproduction_threshold': self.reproduction_threshold,
            'hunger_threshold': self.hunger_threshold,
            'thirst_threshold': self.thirst_threshold,
            'diet_preference': self.diet_preference,
            'offspring_count': self.offspring_count,
            'old_age_threshold': self.old_age_threshold,
            'size': self.size_gene
        }

    def combine_genes(self, parent1, parent2):
        genes = {}
        numeric_genes = ['hunger', 'thirst', 'reproductive_urge', 'speed', 'sensory_radius', 'gestation_duration', 'gestation_timer', 'reproduction_threshold', 'hunger_threshold', 'thirst_threshold', 'offspring_count', 'old_age_threshold', 'size']

        for key in numeric_genes:
            gene_value = (parent1.get_genes()[key] + parent2.get_genes()[key]) / 2
            if random.random() < MUTATION_PROBABILITY:
                gene_value *= random.uniform(0.7, 1.3)
            genes[key] = gene_value

        # Handle diet preference separately
        if random.random() < MUTATION_PROBABILITY:
            genes['diet_preference'] = random.choice(['herbivore', 'carnivore', 'omnivore'])
        else:
            genes['diet_preference'] = parent1.diet_preference if random.random() < 0.5 else parent2.diet_preference

        return genes

    def draw(self, selected_entity, win, offset_x, offset_y, scale):
        if self.alive:
            if self.diet_preference == 'herbivore':
                color = WHITE
            elif self.diet_preference == 'carnivore':
                color = CARNIVORE_COLOR
            elif self.diet_preference == 'omnivore':
                color = OMNIVORE_COLOR

            pygame.draw.circle(win, color,
                               (int((self.x * TILE_SIZE - offset_x) * scale),
                                int((self.y * TILE_SIZE - offset_y) * scale)),
                               int(TILE_SIZE / 2 * scale * self.size))
            self.draw_bars(win, offset_x, offset_y, scale)
            if selected_entity == self:
                self.draw_sensory_radius(win, offset_x, offset_y, scale)

    def draw_bars(self, win, offset_x, offset_y, scale):
        bar_width = TILE_SIZE * scale
        bar_height = 3
        x_pos = (self.x * TILE_SIZE - offset_x) * scale
        y_pos = (self.y * TILE_SIZE - offset_y) * scale - TILE_SIZE * scale / 2 - 10

        hunger_bar = pygame.Rect(x_pos, y_pos, bar_width * (self.hunger / self.hunger_threshold), bar_height)
        thirst_bar = pygame.Rect(x_pos, y_pos + bar_height + 2, bar_width * (self.thirst / self.thirst_threshold), bar_height)
        reproduction_bar = pygame.Rect(x_pos, y_pos + 2 * (bar_height + 2), bar_width * (self.reproductive_urge / self.reproduction_threshold), bar_height)

        pygame.draw.rect(win, RED, hunger_bar)
        pygame.draw.rect(win, BLUE, thirst_bar)
        pygame.draw.rect(win, YELLOW, reproduction_bar)

        # Black outlines for the bars
        pygame.draw.rect(win, BLACK, hunger_bar, 1)
        pygame.draw.rect(win, BLACK, thirst_bar, 1)
        pygame.draw.rect(win, BLACK, reproduction_bar, 1)

    def draw_sensory_radius(self, win, offset_x, offset_y, scale):
        center_x = int((self.x * TILE_SIZE + TILE_SIZE / 2 - offset_x) * scale)
        center_y = int((self.y * TILE_SIZE + TILE_SIZE / 2 - offset_y) * scale)
        radius = int(self.sensory_radius * TILE_SIZE * scale)
        pygame.draw.circle(win, PURPLE, (center_x, center_y), radius, 1)

def generate_map(width, height):
    scale = 30.0
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    seed = np.random.randint(0, 100)

    world = np.zeros((width // TILE_SIZE, height // TILE_SIZE))

    for i in range(width // TILE_SIZE):
        for j in range(height // TILE_SIZE):
            world[i][j] = noise.pnoise2(i/scale,
                                        j/scale,
                                        octaves=octaves,
                                        persistence=persistence,
                                        lacunarity=lacunarity,
                                        repeatx=width // TILE_SIZE,
                                        repeaty=height // TILE_SIZE,
                                        base=seed)

    tiles = []
    plants = []
    for y in range(height // TILE_SIZE):
        row = []
        for x in range(width // TILE_SIZE):
            if world[x][y] < -0.05:
                tile_type = "water"
            elif world[x][y] < 0:
                tile_type = "sand"
            elif world[x][y] < 0.3:
                tile_type = "land"
                if random.random() < PLANT_PROBABILITY:
                    plants.append(Plant(x, y))
            else:
                tile_type = "tree"
            row.append(Tile(x, y, tile_type))
        tiles.append(row)

    return tiles, plants

tiles, plants = generate_map(MAP_WIDTH, MAP_HEIGHT)

def update_environment(tiles, plants):
    for row in tiles:
        for tile in row:
            tile.update()
            if tile.type == "land" and tile.regrowth_timer == 0:
                if random.random() < PLANT_PROBABILITY:
                    plants.append(Plant(tile.x, tile.y))
                    tile.regrowth_timer = REGROWTH_TIME

def draw_environment(win, tiles, offset_x, offset_y, scale):
    for row in tiles:
        for tile in row:
            tile.draw(win, offset_x, offset_y, scale)

def calculate_avg_genes(entities):
    total_genes = {
        'speed': 0,
        'sensory_radius': 0,
        'reproduction_threshold': 0,
        'hunger_threshold': 0,
        'thirst_threshold': 0,
        'diet_preference': {'herbivore': 0, 'carnivore': 0, 'omnivore': 0},
        'offspring_count': 0,
        'old_age_threshold': 0,
        'size': 0
    }
    count = len(entities)

    for entity in entities:
        total_genes['speed'] += entity.speed
        total_genes['sensory_radius'] += entity.sensory_radius
        total_genes['reproduction_threshold'] += entity.reproduction_threshold
        total_genes['hunger_threshold'] += entity.hunger_threshold
        total_genes['thirst_threshold'] += entity.thirst_threshold
        total_genes['diet_preference'][entity.diet_preference] += 1
        total_genes['offspring_count'] += entity.offspring_count
        total_genes['old_age_threshold'] += entity.old_age_threshold
        total_genes['size'] += entity.size_gene

    if count == 0:
        return {key: 0 for key in total_genes.keys()}

    avg_genes = {key: value / count for key, value in total_genes.items() if key != 'diet_preference'}
    avg_genes['diet_preference'] = {key: value / count for key, value in total_genes['diet_preference'].items()}
    return avg_genes

def draw_stats(win, entities, plants):
    entity_count = sum(entity.alive for entity in entities)
    plant_count = len(plants)

    if entity_count > 0:
        avg_genes = calculate_avg_genes([entity for entity in entities if entity.alive])
    else:
        avg_genes = {key: 0 for key in ['speed', 'sensory_radius', 'reproduction_threshold', 'hunger_threshold', 'thirst_threshold', 'offspring_count', 'old_age_threshold', 'size']}
        avg_genes['diet_preference'] = {'herbivore': 0, 'carnivore': 0, 'omnivore': 0}

    stats_text = [
        f'Entities: {entity_count}',
        f'Plants: {plant_count}',
        f'Avg Speed: {avg_genes["speed"]:.2f}',
        f'Avg Sensory Radius: {avg_genes["sensory_radius"]:.2f}',
        f'Avg Reproduction Threshold: {avg_genes["reproduction_threshold"]:.2f}',
        f'Avg Hunger Threshold: {avg_genes["hunger_threshold"]:.2f}',
        f'Avg Thirst Threshold: {avg_genes["thirst_threshold"]:.2f}',
        f'Avg Offspring Count: {avg_genes["offspring_count"]:.2f}',
        f'Avg Old Age Threshold: {avg_genes["old_age_threshold"]:.2f}',
        f'Avg Size: {avg_genes["size"]:.2f}',
        f'Diet Preference:',
        f'  Herbivore: {avg_genes["diet_preference"]["herbivore"] * 100:.2f}%',
        f'  Carnivore: {avg_genes["diet_preference"]["carnivore"] * 100:.2f}%',
        f'  Omnivore: {avg_genes["diet_preference"]["omnivore"] * 100:.2f}%',
        "Select an entity to see their stats",
    ]

    for i, text in enumerate(stats_text):
        stats_surface = font.render(text, True, BLACK)
        win.blit(stats_surface, (10, 10 + i * 20))

# Initialize entities
entities = [Entity(random.randint(0, MAP_WIDTH // TILE_SIZE - 1), random.randint(0, MAP_HEIGHT // TILE_SIZE - 1), tiles) for _ in range(50)]

# Variables for tracking data over time
data_time = []
data_entity_population = []
data_avg_speed = []
data_avg_sensory = []
data_avg_reproduction_threshold = []
data_avg_hunger_threshold = []
data_avg_thirst_threshold = []
data_avg_offspring_count = []
data_avg_old_age_threshold = []
data_avg_size = []
data_avg_diet_preference = {'herbivore': [], 'carnivore': [], 'omnivore': []}

# Function to plot graphs
def plot_graphs():
    fig, axes = plt.subplots(6, 2, figsize=(15, 24))  # 9 rows, 2 columns to fit all graphs

    axes[0, 0].plot(data_time, data_entity_population, label="Entity Population", color='blue')
    axes[0, 0].set_title("Entity Population Over Time")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Population")
    axes[0, 0].legend()

    axes[1, 0].plot(data_time, data_avg_speed, label="Avg Speed", color='green')
    axes[1, 0].set_title("Avg Speed Over Time")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Speed")
    axes[1, 0].legend()

    axes[1, 1].plot(data_time, data_avg_sensory, label="Avg Sensory Radius", color='cyan')
    axes[1, 1].set_title("Avg Sensory Radius Over Time")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Sensory Radius")
    axes[1, 1].legend()

    axes[2, 0].plot(data_time, data_avg_reproduction_threshold, label="Avg Reproduction Threshold", color='magenta')
    axes[2, 0].set_title("Avg Reproduction Threshold Over Time")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("Reproduction Threshold")
    axes[2, 0].legend()

    axes[2, 1].plot(data_time, data_avg_hunger_threshold, label="Avg Hunger Threshold", color='gray')
    axes[2, 1].set_title("Avg Hunger Threshold Over Time")
    axes[2, 1].set_xlabel("Time")
    axes[2, 1].set_ylabel("Hunger Threshold")
    axes[2, 1].legend()

    axes[3, 0].plot(data_time, data_avg_thirst_threshold, label="Avg Thirst Threshold", color='blue')
    axes[3, 0].set_title("Avg Thirst Threshold Over Time")
    axes[3, 0].set_xlabel("Time")
    axes[3, 0].set_ylabel("Thirst Threshold")
    axes[3, 0].legend()

    axes[3, 1].plot(data_time, data_avg_offspring_count, label="Avg Offspring Count", color='orange')
    axes[3, 1].set_title("Avg Offspring Count Over Time")
    axes[3, 1].set_xlabel("Time")
    axes[3, 1].set_ylabel("Offspring Count")
    axes[3, 1].legend()

    axes[4, 0].plot(data_time, data_avg_old_age_threshold, label="Avg Old Age Threshold", color='purple')
    axes[4, 0].set_title("Avg Old Age Threshold Over Time")
    axes[4, 0].set_xlabel("Time")
    axes[4, 0].set_ylabel("Old Age Threshold")
    axes[4, 0].legend()

    axes[4, 1].plot(data_time, data_avg_size, label="Avg Size", color='brown')
    axes[4, 1].set_title("Avg Size Over Time")
    axes[4, 1].set_xlabel("Time")
    axes[4, 1].set_ylabel("Size")
    axes[4, 1].legend()

    axes[5, 0].plot(data_time, data_avg_diet_preference['herbivore'], label="Herbivore %", color='green')
    axes[5, 0].plot(data_time, data_avg_diet_preference['carnivore'], label="Carnivore %", color='red')
    axes[5, 0].plot(data_time, data_avg_diet_preference['omnivore'], label="Omnivore %", color='yellow')
    axes[5, 0].set_title("Diet Preference Over Time")
    axes[5, 0].set_xlabel("Time")
    axes[5, 0].set_ylabel("Percentage")
    axes[5, 0].legend()

    plt.tight_layout()
    plt.show()

async def main():
    # Main loop variables
    offset_x, offset_y = 0, 0
    scale = 1.0
    dragging = False
    last_mouse_pos = None
    selected_entity = None
    time_step = 0
    show_graphs_button = pygame.Rect(10, SCREEN_HEIGHT - 40, 150, 30)
    show_stats_button = pygame.Rect(10, 10, 150, 30)
    stats_visible = False

    # Main loop
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                    mouse_x, mouse_y = last_mouse_pos
                    if show_graphs_button.collidepoint(mouse_x, mouse_y):
                        plot_graphs()
                    elif show_stats_button.collidepoint(mouse_x, mouse_y):
                        stats_visible = not stats_visible
                    for entity in entities:
                        if entity.alive:
                            entity_x = (entity.x * TILE_SIZE - offset_x) * scale
                            entity_y = (entity.y * TILE_SIZE - offset_y) * scale
                            if entity_x - TILE_SIZE / 2 <= mouse_x <= entity_x + TILE_SIZE / 2 and entity_y - TILE_SIZE / 2 <= mouse_y <= entity_y + TILE_SIZE / 2:
                                selected_entity = entity
                                break
                elif event.button == 4:
                    scale += 0.1
                elif event.button == 5:
                    scale = max(0.1, scale - 0.1)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_pos = pygame.mouse.get_pos()
                    dx = mouse_pos[0] - last_mouse_pos[0]
                    dy = mouse_pos[1] - last_mouse_pos[1]
                    offset_x -= dx / scale
                    offset_y -= dy / scale
                    last_mouse_pos = mouse_pos

        # Update environment
        update_environment(tiles, plants)

        # Update entities
        for entity in entities:
            if entity.alive:
                entity.move(tiles, plants, entities)
            entity.update_gestation(entities, tiles)

        win.fill(WHITE)
        draw_environment(win, tiles, offset_x, offset_y, scale)
        for plant in plants:
            plant.draw(win, offset_x, offset_y, scale)
        for entity in entities:
            entity.draw(selected_entity, win, offset_x, offset_y, scale)

        if stats_visible:
            draw_stats(win, entities, plants)
            close_button = pygame.Rect(170, 10, 20, 20)
            pygame.draw.rect(win, RED, close_button)
            pygame.draw.line(win, BLACK, (172, 12), (188, 28), 2)
            pygame.draw.line(win, BLACK, (172, 28), (188, 12), 2)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if close_button.collidepoint(mouse_x, mouse_y):
                    stats_visible = False
        else:
            pygame.draw.rect(win, GRAY, show_stats_button)
            pygame.draw.rect(win, BLACK, show_stats_button, 2)
            stats_button_text = font.render("Show Stats", True, BLACK)
            win.blit(stats_button_text, (show_stats_button.x + 10, show_stats_button.y + 5))

        pygame.draw.rect(win, GRAY, show_graphs_button)
        pygame.draw.rect(win, BLACK, show_graphs_button, 2)
        button_text = font.render("Show Graphs", True, BLACK)
        win.blit(button_text, (show_graphs_button.x + 10, show_graphs_button.y + 5))

        if selected_entity:
            offset_x = selected_entity.x * TILE_SIZE - SCREEN_WIDTH // 2 / scale
            offset_y = selected_entity.y * TILE_SIZE - SCREEN_HEIGHT // 2 / scale
            popup_width, popup_height = 350, 400
            popup_x, popup_y = SCREEN_WIDTH - popup_width - 10, 10
            pygame.draw.rect(win, GRAY, (popup_x, popup_y, popup_width, popup_height))
            pygame.draw.rect(win, BLACK, (popup_x, popup_y, popup_width, popup_height), 2)
            info_text = [
                f'Type: {selected_entity.diet_preference.capitalize()}',
                f'Speed: {selected_entity.speed:.2f}',
                f'Sensory Radius: {selected_entity.sensory_radius}',
                f'Hunger: {selected_entity.hunger}',
                f'Thirst: {selected_entity.thirst}',
                f'Reproductive Urge: {selected_entity.reproductive_urge}',
                f'Gender: {selected_entity.gender}',
                f'Gestation Duration: {getattr(selected_entity, "gestation_duration", "N/A")}',
                f'Gestation Timer: {getattr(selected_entity, "gestation_timer", "N/A")}',
                f'Reproduction Threshold: {selected_entity.reproduction_threshold}',
                f'Hunger Threshold: {selected_entity.hunger_threshold}',
                f'Thirst Threshold: {selected_entity.thirst_threshold}',
                f'Offspring Count: {selected_entity.offspring_count}',
                f'Generation: {selected_entity.generation}',
                f'State: {selected_entity.state}',
                f'Pregnant: {selected_entity.pregnant if selected_entity.gender == "female" else "N/A"}',
                f'Age: {selected_entity.age}',
                f'Old Age Threshold: {selected_entity.old_age_threshold}',
                f'Size: {selected_entity.size_gene:.2f}'
            ]
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, BLACK)
                win.blit(text_surface, (popup_x + 10, popup_y + 10 + i * 20))

            close_button = pygame.Rect(popup_x + popup_width - 30, popup_y + 10, 20, 20)
            pygame.draw.rect(win, RED, close_button)
            pygame.draw.line(win, BLACK, (popup_x + popup_width - 28, popup_y + 12), (popup_x + popup_width - 12, popup_y + 28), 2)
            pygame.draw.line(win, BLACK, (popup_x + popup_width - 28, popup_y + 28), (popup_x + popup_width - 12, popup_y + 12), 2)
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if close_button.collidepoint(mouse_x, mouse_y):
                    selected_entity = None

        data_time.append(time_step)
        data_entity_population.append(sum(entity.alive for entity in entities))

        if sum(entity.alive for entity in entities) > 0:
            data_avg_speed.append(sum(entity.speed for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_sensory.append(sum(entity.sensory_radius for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_reproduction_threshold.append(sum(entity.reproduction_threshold for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_hunger_threshold.append(sum(entity.hunger_threshold for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_thirst_threshold.append(sum(entity.thirst_threshold for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_offspring_count.append(sum(entity.offspring_count for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_old_age_threshold.append(sum(entity.old_age_threshold for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            data_avg_size.append(sum(entity.size_gene for entity in entities if entity.alive) / sum(entity.alive for entity in entities))
            avg_diet_pref = calculate_avg_genes(entities)['diet_preference']
            data_avg_diet_preference['herbivore'].append(avg_diet_pref['herbivore'] * 100)
            data_avg_diet_preference['carnivore'].append(avg_diet_pref['carnivore'] * 100)
            data_avg_diet_preference['omnivore'].append(avg_diet_pref['omnivore'] * 100)
        else:
            data_avg_speed.append(0)
            data_avg_sensory.append(0)
            data_avg_reproduction_threshold.append(0)
            data_avg_hunger_threshold.append(0)
            data_avg_thirst_threshold.append(0)
            data_avg_offspring_count.append(0)
            data_avg_old_age_threshold.append(0)
            data_avg_size.append(0)
            data_avg_diet_preference['herbivore'].append(0)
            data_avg_diet_preference['carnivore'].append(0)
            data_avg_diet_preference['omnivore'].append(0)

        time_step += 1

        pygame.display.update()

    pygame.quit()

    plot_graphs()

asyncio.run(main())
