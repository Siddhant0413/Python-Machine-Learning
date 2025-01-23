import random
import math
from Cell import Cell

class Mouse:
    def __init__(self, cell, stochastic=False):
        self.cell = cell
        self.stochastic = stochastic
        self.movements = [cell]

    def move(self, ship):
        if not self.stochastic:
            return  # Stationary mouse does not move

        neighbors = ship.get_neighbors(self.cell)
        if neighbors:
            new_cell = random.choice(neighbors + [self.cell])  # Move to a random neighboring cell or stay in place
            self.cell = new_cell
            self.movements.append(new_cell)