import pygame
import math
import numpy as np
from queue import PriorityQueue
import statistics

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

ORANGE = (255, 165, 0)
TURQUOISE = (64, 224, 208)
WHITE = (255, 255, 255)
PURPLE = (128, 0, 128)
GREY = (128, 128, 128)
BLACK = (0, 0, 0)
DELTA = 4
ALPHA = 1


class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def is_barrier(self):
        return self.color == BLACK

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False

    def weights(self, grid):

        node = grid[self.row][self.col]

        if node.color == (255, 255, 255):
            return 2


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)


def algorithm(draw, grid, start, end):
    count = 0

    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            # end.make_end()
            nodes = []
            count = 0
            while current in came_from:
                current = came_from[current]
                count = count + current.weights(grid)
                # current.make_path()
                draw()
                nodes.append(current)
            efficiency = 10 / count
            nodes.insert(0, efficiency)

            return True, nodes

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + neighbor.weights(grid)

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
        draw()

    return False


def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col


def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)

    start = []
    end = []

    path = {}
    path1 = {}

    run = True
    while run:
        draw(win, grid, ROWS, width)
        pressed = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]

                if pressed[pygame.K_s]:
                    if spot not in start:
                        start.append(spot)
                        spot.make_start()


                elif pressed[pygame.K_e]:
                    if spot not in end:
                        end.append(spot)
                        spot.make_end()


                elif pressed[pygame.K_i]:
                    spot.make_barrier()

                elif pressed[pygame.K_r]:
                    spot.make_road()

                elif pressed[pygame.K_b]:
                    spot.make_bush()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    for i in range(len(start)):
                        path[start[i]] = path1
                        for j in range(len(end)):
                            nodes = algorithm(lambda: draw(win, grid, ROWS, width), grid, start[i], end[j])
                            print('path for start %d and end %d is formed' % (i + 1, j + 1))

                            path1[end[j]] = nodes[1]
                    return path, start, end

                if event.key == pygame.K_c:
                    grid = make_grid(ROWS, width)

    pygame.quit()


main(WIN, WIDTH)
