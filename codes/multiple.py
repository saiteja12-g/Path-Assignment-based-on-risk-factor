import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Multiple")

BLACK = (0, 0, 0)                  #barrier
BUSH = (0, 100, 100)				 #bush
D_ROAD = (105, 105, 105)			 #damaged road
SAND = (255, 255, 51)				 #sand
GRASS = (124, 252, 0)				 #grass

RED = (255, 0, 0)
BLUE = (0, 255, 0)
WHITE = (255, 255, 255)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
GREEN = (0, 255, 0)

class Node:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.width = width
		self.total_rows =  total_rows
		self.neighours = []
		self.color = WHITE
	
	def pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == RED

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def is_barrier(self):
		return self.color == BLACK

	def is_bush(self):
		return self.color == BUSH

	def is_droad(self):
		return self.color == D_ROAD
	
	def is_sand(self):
		return self.color == SAND

	def is_grass(self):
		return self.color == GRASS

	def is_road(self):
		return self.color == ROAD
	
	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_end(self):
		self.color = TURQUOISE

	def make_barrier(self):
		self.color = BLACK

	def make_bush(self):
		self.color = BUSH

	def make_droad(self):
		self.color = D_ROAD
	
	def make_sand(self):
		self.color = SAND

	def make_grass(self):
		self.color = GRASS

	def make_road(self):
		self.color = ROAD

	def make_path(self):
		self.color = PURPLE

	def make_open(self):
		self.color = GREEN

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append([grid[self.row + 1][self.col],self.color])
			print(self.neighbors)

		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append([grid[self.row - 1][self.col], self.color])

		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append([grid[self.row][self.col + 1], self.color])

		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append([grid[self.row][self.col - 1], self.color])

	def __lt__(self, other):
		return False

def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()
		
def make_grid(rows, width):
		grid = []
		gap = width // rows
		for i in range(rows):
			grid.append([])
			for j in range(rows):
				node = Node(i, j, gap, rows)
				grid[i].append(node)

		return grid

def draw_grid(win ,rows, width):
		gap = width // rows
		for i in range(rows):
			pygame.draw.line(win, GREY, (0, i*gap) , (width, i*gap))
			for j in range(rows):
				pygame.draw.line(win, GREY, (j*gap,0), (j*gap,width))

def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()

def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y,x = pos

	row = y // gap
	col = x // gap

	return row, col 

def weights(grid,width):
	ROWS = 50
	wt = []
	for row in grid:
		for spot in row:
			if spot.color == (0, 100, 100):
				print(1)
				wt.append(8)
				return 8
			elif spot.color == (105, 105, 105):
				print(2)
				wt.append(6)
				return 6
			elif spot.color == (255, 255, 51):
				print(3)
				wt.append(5)
				return 5
			elif spot.color == (124, 252, 0):
				print(4)
				wt.append(3)
				return 3
			elif spot.color == (192, 192, 192):
				print(5)
				wt.append(1)
				return 1		 
				
			elif spot.color == (255, 255, 255):
				wt.append(2)
				return 2

def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)


def algorithm(draw, grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start[0]))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start[0]] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start[0]] = h(start[0].pos(), end[0].pos())

	open_set_hash = {start[0]}

	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end[0]:
			reconstruct_path(came_from, end[0], draw)
			end[0].make_end()
			return True

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + weights(grid, 100)
			print('g_score = ',g_score[current])
			print(weights(grid, 800))
			print('temp_g_score = ', temp_g_score)

			

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.pos(), end[0].pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		draw()

		if current != start:
			current.make_closed()

	return False

def main(win,width):
	ROWS = 50
	grid = make_grid(ROWS, width)
	
	start = []
	end = []
	
	run = True
	
	while run:
		events = pygame.event.get()
		pressed = pygame.key.get_pressed()
		draw(win, grid, ROWS, width)
		for event in events:
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]:
				pos = pygame.mouse.get_pos()
				row,col = get_clicked_pos(pos, ROWS, width)
				node = grid[row][col]

				if pressed[pygame.K_s]:
					start.append(node)
					node.make_start()

				elif pressed[pygame.K_e]:
					end.append(node)
					node.make_end()

				elif pressed[pygame.K_b]:
					node.make_bush()
					
				elif pressed[pygame.K_d]:
					node.make_droad()

				elif pressed[pygame.K_m]:
					node.make_sand()

				elif pressed[pygame.K_g]:
					node.make_grass()

				elif pressed[pygame.K_r]:
					node.make_road()

				elif pressed[pygame.K_i]:
					node.make_barrier()



			elif pygame.mouse.get_pressed()[2]:
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				node = grid[row][col]
				node.reset()


			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					for row in grid:
						for spot in row:
							spot.update_neighbors(grid)

					algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)

				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)

	pygame.quit()

main(WIN, WIDTH)
