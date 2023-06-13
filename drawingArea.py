from neural import *
from main import HyperParameters

import pygame
from PIL import Image
import numpy as np



hp = HyperParameters()
neural = NeuralFromJson("./saves/newDataSet.json", hp)

def RunImage():
    # Convert the drawn image to 28x28 pixels
    scaled_surface = pygame.transform.smoothscale(screen, (28, 28))
    drawn_image = pygame.surfarray.array2d(scaled_surface)

    # Create a new PIL Image with proper orientation
    pil_image = Image.fromarray(drawn_image)
    pil_image = pil_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    pil_image = pil_image.rotate(-90)

    # Adjust the aspect ratio to fill a 28x28 image
    new_image = Image.new("L", (28, 28), color=255)
    image_width, image_height = pil_image.size
    if image_width > image_height:
        new_height = int(28 / image_width * image_height)
        resized_image = pil_image.resize((28, new_height))
        top_margin = (28 - new_height) // 2
        new_image.paste(resized_image, (0, top_margin))
    else:
        new_width = int(28 / image_height * image_width)
        resized_image = pil_image.resize((new_width, 28))
        left_margin = (28 - new_width) // 2
        new_image.paste(resized_image, (left_margin, 0))

    pixels = np.array(pil_image.getdata())
    pixels = pixels * (1.0 / pixels.max())


    print(neural.Classify(pixels))


# Initialize Pygame
pygame.init()

# Set up the drawing area
size = 400
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Digit Recognition")

# Set up colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Set up the drawing variables
drawing = False

print("\nLeft click: draw / Right click: erase / c: Clear / Return: Submit")
# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                RunImage()
            if event.key == pygame.K_c:
                screen.fill(BLACK)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                pygame.draw.circle(screen, WHITE, pygame.mouse.get_pos(), 25)

    # Update the screen
    pygame.display.flip()


# Quit the program
pygame.quit()


