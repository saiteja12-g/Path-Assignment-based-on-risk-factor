import pygame
width = 800
GREY = (128, 128, 128)

WIDTH = 800
rows = 50
gap = width // rows
WIN = pygame.display.set_mode((WIDTH, WIDTH))
for i in range(rows):
  pygame.draw.line(WIN, GREY, (0, i*gap) , (width, i*gap))

  for j in range(rows):
	  pygame.draw.line(WIN, GREY, (j*gap,0), (j*gap,width))

pygame.display.set_caption("OWN")
pygame.display.update()

