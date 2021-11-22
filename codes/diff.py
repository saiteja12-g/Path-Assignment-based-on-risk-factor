import pygame
import math
import numpy as np
from queue import PriorityQueue
import statistics

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

BUSH = (0, 100, 100)  # bush
ROAD = (192, 192, 192)  # road

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

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == GREEN

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == ORANGE

    def is_end(self):
        return self.color == TURQUOISE

    def is_bush(self):
        return self.color == BUSH

    def is_road(self):
        return self.color == ROAD

    def reset(self):
        self.color = WHITE

    def make_start(self):
        self.color = ORANGE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = GREEN

    def make_barrier(self):
        self.color = BLACK

    def make_bush(self):
        self.color = BUSH

    def make_end(self):
        self.color = TURQUOISE

    def make_road(self):
        self.color = ROAD

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

        if node.color == (0, 100, 100):
            return 8

        elif node.color == (192, 192, 192):
            return 1

        elif node.color == (255, 255, 255):
            return 2

        else:
            return 3


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


def demand_bound(efficiency):
    range1 = max(efficiency)
    sample = []
    for i in range(len(efficiency)):
        temp1 = 2 * efficiency[i] ** 2.5 / range1 + efficiency[i] - efficiency[i] ** 2.5 / range1
        sample.append(temp1)

    return max(sample), sample


def robot_demand_poisson(efficiency):
    range2 = max(efficiency)
    robot_demand_sample = np.zeros([2, 2])

    for i in range(0, 2, 1):
        for j in range(0, 2, 1):
            robot_demand_sample[i][j] = 2 * efficiency[i] ** 2.5 / range2 + efficiency[i] - efficiency[
                i] ** 2.5 / range2
    print('b:' ,robot_demand_sample)
    return robot_demand_sample


def efficiency_distribution(set1, robot_demand_sample, efficiency):
    max_each_demand = np.zeros(len(set1))
    sum_max_demand = np.zeros([1, 1000])
    robot_demand_sample = robot_demand_poisson(efficiency)
    for i in range(len(set1)):
        if set1[i][0] == 0:
            for j in range(len(set1[i])):
                max_each_demand[i:0] = max(max_each_demand[i:], robot_demand_sample[i, set1[i][j]])

            sum_max_demand[i][j] = sum_max_demand[i][j] + max_each_demand[i, :]
            print('a :', sum_max_demand)
    return sum_max_demand


def H_approx(set1, pair, tau, robot_demand_sample, alpha, efficiency):
    set1[pair[0]] = [set1[pair[0]], pair[1]]
    sum_max_demand = efficiency_distribution(set1, robot_demand_sample, efficiency)
    tail_h = max((tau) - sum_max_demand[pair[0]][pair[1]])
    h_approx = tau - (1 / alpha) * statistics.mean(tail_h)
    print(h_approx)
    return h_approx


def selection(win, width, delta, alpha):
    temp = []
    path2 = main(WIN, WIDTH)
    print(path2)
    for i in path2[1]:
        for j in path2[2]:
            temp.append(path2[0][i][j][0])
    print(temp)

    one_demand_bound = round(demand_bound(temp)[0])
    tau_bound = 2 * one_demand_bound
    n_tau = tau_bound / delta + 1
    H_value = []
    tau_hvalue = []
    H_star_value = []
    H_set = {}

    cnt = 1
    for tau in range(0, tau_bound, delta):
        gre_set = np.empty([2, 2], dtype='object')
        gre_selected = []
        gre_hvalue_last = tau * (1 - 1 / alpha)

        for r in range(1, 3, 1):
            margin_inx = np.zeros([2, 2 - len(gre_selected)])
            margin_gain = np.zeros([2, 2 - len(gre_selected)])
            hvalue_current = np.zeros([2, 2 - len(gre_selected)])

            for m in range(1, 3, 1):
                cnt_j = 1
                for n in range(1, 3, 1):
                    if path2[1][n] not in gre_selected:

                        gre_current_mn = H_approx(gre_set, [m, n], tau, demand_bound(temp)[1], alpha, temp)
                        margin_hvalue_mn = gre_current_mn - gre_hvalue_last;
                        margin_inx[m, cnt_j] = n
                        margin_gain[m, cnt_j] = margin_hvalue_mn;
                        hvalue_current[m, cnt_j] = gre_current_mn;
                        cnt_j = cnt_j + 1
                    else:
                        continue

                [row_inx, col_inx] = margin_gain == max(max(margin_gain))
                gre_set[row_inx] = [gre_set[row_inx], margin_inx[row_inx, col_inx]]
                gre_selected = [gre_selected, margin_inx[row_inx, col_inx]]
                gre_hvalue_last = hvalue_current[row_inx, col_inx]

    H_set[cnt] = gre_set
    H_value[cnt] = gre_hvalue_last
    tau_hvalue[cnt:] = [tau, gre_hvalue_last]
    H_star_value[cnt] = H_value[cnt] + (1 / 2) * tau * (1 / alpha - 1)
    cnt = cnt + 1

    max_hstar_bound = max(H_star_value)
    max_Hstar_inx = find(H_star_value == max_hstar_bound, 1)
    cvar_gre_set = H_set[max_Hstar_inx]
    print(cvar_gre_set)


selection(WIN, WIDTH, DELTA, ALPHA)
