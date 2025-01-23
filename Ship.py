import decimal
import random
from Cell import Cell
from Bot import Bot
from Mouse import Mouse

decimal.getcontext().prec = 5

class Ship:
    def __init__(self, d):

        self.d = d #Size
        self.grid = [[Cell(f, g) for g in range(d)] for f in range(d)] #Grid Layout
        self.no_enter = []  # Cells that cannot be entered
        self.blocked_neighbor = []  # Cells with 1 neighbor

        self.bot = None
        self.mouse = None

    def __str__(self):
        s = "Ship:\n"
        for f in range(self.d):
            for g in range(self.d):
                if self.bot != None and self.grid[f][g] == self.bot.cell:
                    s += "[R]"
                elif self.mouse != None and self.grid[f][g] == self.mouse.cell:
                    s += "[M]"
                elif self.grid[f][g].is_open is True:
                    s += "[  ]"
                else:
                    s += "[  ]"
            s += "\n"
        return s

    def addBot(self):
        tCell = self.grid[random.randint(0, self.d - 1)][random.randint(0, self.d - 1)] #Random cell to spawn bot

        while tCell.is_open is False:
            tCell = self.grid[random.randint(0, self.d - 1)][random.randint(0, self.d - 1)]
        self.bot = Bot(tCell)

    def addMouse(self, stochastic=False):
        tCell = self.grid[random.randint(0, self.d - 1)][random.randint(0, self.d - 1)]  # Random cell to spawn mouse

        while not tCell.is_open or tCell == self.bot.cell:
            tCell = self.grid[random.randint(0, self.d - 1)][random.randint(0, self.d - 1)]
        self.mouse = Mouse(tCell, stochastic)

    def manhattan_distance(self, cell1, cell2):
        return abs(cell1.f - cell2.f) + abs(cell1.g - cell2.g)
    
    def get_neighbors(self, cell): #Gets the current cells neighbors 
        neighbors = []
        if cell.f - 1 >= 0:
            neighbors.append(self.grid[cell.f - 1][cell.g])
        if cell.f + 1 < self.d:
            neighbors.append(self.grid[cell.f + 1][cell.g])
        if cell.g - 1 >= 0:
            neighbors.append(self.grid[cell.f][cell.g - 1])
        if cell.g + 1 < self.d:
            neighbors.append(self.grid[cell.f][cell.g + 1])
        return neighbors

    def single_neighbor(self, cell):  #Checks if the cell given has one open neighbor

        num_neighbors = 0
        
        while (num_neighbors < 2):
            if cell.f - 1 >= 0:
                if self.grid[cell.f - 1][cell.g].is_open is True: #West
                    num_neighbors += 1
            if cell.f + 1 < self.d:
                if self.grid[cell.f + 1][cell.g].is_open is True: #East
                    num_neighbors += 1
            if cell.g + 1 < self.d:
                if self.grid[cell.f][cell.g + 1].is_open is True: #North
                    num_neighbors += 1
            if cell.g - 1 >= 0:
                if self.grid[cell.f][cell.g - 1].is_open is True: #South
                    num_neighbors += 1
            
            if num_neighbors == 1:
                return True
            return False
    
    def open_cell(self, f, g): #Method to open a cell

        if self.grid[f][g].is_open is True:
            return

        self.grid[f][g].is_open = True

        if self.single_neighbor(self.grid[f][g]) is True and self.grid[f][g] not in self.no_enter:
            self.no_enter.append(self.grid[f][g])

        elif self.single_neighbor(self.grid[f][g]) is False and self.grid[f][g] in self.no_enter:
            self.no_enter.remove(self.grid[f][g])

        if f - 1 >= 0:
            if self.grid[f - 1][g].is_open is False and self.single_neighbor(self.grid[f - 1][g]) is True: #West
                self.blocked_neighbor.append(self.grid[f - 1][g])
            
            elif self.grid[f - 1][g] in self.blocked_neighbor:
                self.blocked_neighbor.remove(self.grid[f - 1][g])

        if f + 1 < self.d:
            if self.grid[f + 1][g].is_open is False and self.single_neighbor(self.grid[f + 1][g]) is True: #East
                self.blocked_neighbor.append(self.grid[f + 1][g])
            
            elif self.grid[f + 1][g] in self.blocked_neighbor:
                self.blocked_neighbor.remove(self.grid[f + 1][g])

        if g + 1 < self.d:
            if self.grid[f][g + 1].is_open is False and self.single_neighbor(self.grid[f][g + 1]) is True: #North
                self.blocked_neighbor.append(self.grid[f][g + 1])
            
            elif self.grid[f][g + 1] in self.blocked_neighbor:
                self.blocked_neighbor.remove(self.grid[f][g + 1])   

        if g - 1 >= 0:
            if self.grid[f][g - 1].is_open is False and self.single_neighbor(self.grid[f][g - 1]) is True: #South
                self.blocked_neighbor.append(self.grid[f][g - 1])
            
            elif self.grid[f][g - 1] in self.blocked_neighbor:
                self.blocked_neighbor.remove(self.grid[f][g - 1])