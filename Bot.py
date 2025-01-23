import random
import math
from Cell import Cell

class Bot:
    def __init__(self, cell):
        self.cell = cell

    def move(self, cell):
        self.cell = cell

    def sense(self, ship, mouse, alpha):
        distance = ship.manhattan_distance(self.cell, mouse.cell)
        return math.exp(-alpha * (distance - 1))