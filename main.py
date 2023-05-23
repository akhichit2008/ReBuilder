import pygame
import json
import sys

# Define some colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load the map data
with open("map.json") as f:
    map_data = json.load(f)

# Convert map_size values to integers
map_width = int(map_data["map_size"]["width"])
map_height = int(map_data["map_size"]["height"])


screen = pygame.display.set_mode((map_width, map_height))

# Create a sprite group for the objects
objects = pygame.sprite.Group()

# Create a function to draw the map tiles
def draw_map(map_data):
    for tile in map_data:
        tile_x = tile["x"]
        tile_y = tile["y"]
        tile_type = tile["type"]

        if tile_type == "grass":
            pygame.draw.rect(screen, GREEN, (tile_x, tile_y, 32, 32))
        elif tile_type == "water":
            pygame.draw.rect(screen, BLUE, (tile_x, tile_y, 32, 32))

# Create objects based on the map data
for obj_data in map_data["objects"]:
    obj_type = obj_data["type"]
    obj_x = obj_data["x"]
    obj_y = obj_data["y"]

    if obj_type == "house":
        obj = pygame.sprite.Sprite()
        obj.image = pygame.Surface((100, 100))
        obj.image.fill(RED)
        obj.rect = obj.image.get_rect()
        obj.rect.center = (obj_x, obj_y)
        objects.add(obj)

    elif obj_type == "factory":
        obj = pygame.sprite.Sprite()
        obj.image = pygame.Surface((50, 50))
        obj.image.fill(GREEN)
        obj.rect = obj.image.get_rect()
        obj.rect.center = (obj_x, obj_y)
        objects.add(obj)
    elif obj_type == "garden":
        obj = pygame.sprite.Sprite()
        obj.image = pygame.Surface()
        obj.image.fill((50,60,80))
        obj.rect = obj.image.get_rect()
        obj.rect.center = (obj_x,obj_y)
        objects.add(obj)

# Run the game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Clear the screen
    screen.fill((255, 250, 250))

    # Draw the map tiles
    draw_map(map_data["tiles"])

    # Draw the objects
    objects.draw(screen)

    # Update the screen
    pygame.display.update()
