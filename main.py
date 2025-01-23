import copy
import decimal
import random
import math
import queue
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import pickle
from queue import PriorityQueue, Queue
from collections import deque
from Ship import Ship
from Bot import Bot
from Mouse import Mouse
from Cell import Cell
from Trainer import BotActionNetworkCNN

#Constants for actions the bot will take 
MOVE_ACTION = 0
SENSE_ACTION = 1

decimal.getcontext().prec = 5

#Moves towards the cell with the highest probability of having the mouse
def bot1(ship, alpha):
    data = []  # To collect states and actions
    steps = 0
    currentShipLayout = None
    while True:
        currentShipLayout = get_ship_state(ship) #Get current ship layout and map 
        hProb_cell = None
        hProb = 0

        # Calculate probability for each cell and find the cell with the highest probability
        for row in ship.grid:
            for cell in row:
                if cell.is_open:
                    distance = ship.manhattan_distance(cell, ship.mouse.cell)
                    prob = math.exp(-alpha * (distance - 1))
                    if prob > hProb:
                        hProb = prob
                        hProb_cell = cell

        # Move bot towards the cell with the highest probability
        if hProb_cell:
            next_cell = nextStep(ship.bot.cell, hProb_cell, ship)
            ship.bot.move(next_cell)
            steps += 1
            ship.mouse.move(ship)
            data.append((currentShipLayout, MOVE_ACTION)) #Collect data as a movement
            print(f"Bot Location: ({ship.bot.cell.f}, {ship.bot.cell.g})")
            print(f"Mouse Location: ({ship.mouse.cell.f}, {ship.mouse.cell.g})")
            if ship.bot.cell == ship.mouse.cell:
                return True, steps , data
        else:
            return False, steps , data
        
#Alternates between moving and sensing the mouse
def bot2(ship, alpha):
    data = []  # To collect states and actions
    steps = 1
    move = True
    currentShipLayout = None

    while True:
        currentShipLayout = get_ship_state(ship) #Get current ship layout and map 
        if move:
            hProb_cell = None
            hProb = 0

            # Calculate probability for each cell and find the cell with the highest probability
            for row in ship.grid:
                for cell in row:
                    if cell.is_open:
                        distance = ship.manhattan_distance(cell, ship.mouse.cell)
                        prob = math.exp(-alpha * (distance - 1))
                        if prob > hProb:
                            hProb = prob
                            hProb_cell = cell

            # Move bot towards the cell with the highest probability
            if hProb_cell:
                next_cell = nextStep(ship.bot.cell, hProb_cell, ship)
                ship.bot.move(next_cell)
                steps += 1
                ship.mouse.move(ship)
                data.append((currentShipLayout, MOVE_ACTION)) #Collect data as a movement
                print(f"Bot Location: ({ship.bot.cell.f}, {ship.bot.cell.g})")
                print(f"Mouse Location: ({ship.mouse.cell.f}, {ship.mouse.cell.g})")
                if ship.bot.cell == ship.mouse.cell:
                    return True, steps , data
        else:
            # Sense the mouse and determine the probability of the mouse being at the current cell
            prob = ship.bot.sense(ship, ship.mouse, alpha)
            print(f"Bot sensed at: ({ship.bot.cell.f}, {ship.bot.cell.g}) with probability {prob}")
            data.append((currentShipLayout, SENSE_ACTION)) # Append state and action pair
            # Move toward the mouse if the probability is high
            if prob >= 0.5:
                next_cell = nextStep(ship.bot.cell, ship.mouse.cell, ship)
                ship.bot.move(next_cell)
                steps += 1
                if ship.bot.cell == ship.mouse.cell:
                    return True, steps , data
            else:
                # If probability is low, continue with the alternating strategy
                move = not move
                continue

        move = not move

def bot3(ship, alpha):
    data = []
    steps = 0
    move = True  # Alternate between moving and sensing
    map = [[1 / (ship.d ** 2) for _ in range(ship.d)] for _ in range(ship.d)]  # Initial map
    recent_moves = []
    
    while True:
        steps += 1
        currentShipLayout = get_ship_state(ship) #Get current ship layout and map 

        # Check if the bot has found the mouse
        if ship.bot.cell == ship.mouse.cell:
            print(f"Mouse found at ({ship.mouse.cell.f}, {ship.mouse.cell.g}) by bot at ({ship.bot.cell.f}, {ship.bot.cell.g}) after {steps} steps.")
            return True, steps , data

        if move:
            # Move towards the cell with the highest probability
            hProb_cell = get_highest_cell(ship, map)

            if hProb_cell is None:
                hProb_cell = ship.bot.cell  # Stay in place if no highest probability cell found

            # Check so that the bot does not move back and forth between cells and is always going to new ones
            if hProb_cell in recent_moves:
                # If recently moved to a cell, find the next best option
                for neighbor in ship.get_neighbors(ship.bot.cell):
                    if neighbor not in recent_moves:
                        next_cell = neighbor
                        break
                else:
                    next_cell = ship.bot.cell  # Keep the same cell if no better move can be made
            else:
                next_cell = nextStep(ship.bot.cell, hProb_cell, ship)

            # Update recent moves
            recent_moves.append(next_cell)
            if len(recent_moves) > 2:
                recent_moves.pop(0)

            ship.bot.move(next_cell)
            ship.mouse.move(ship)
            data.append((currentShipLayout, MOVE_ACTION))  #Collect data as a movement
            print(f"Bot Location: ({ship.bot.cell.f}, {ship.bot.cell.g})")
            print(f"Mouse Location: ({ship.mouse.cell.f}, {ship.mouse.cell.g})")
        else:
            # Sense the mouse and update the probability map
            prob = ship.bot.sense(ship, ship.mouse, alpha)
            print(f"Bot sensed at: ({ship.bot.cell.f}, {ship.bot.cell.g}) with probability {prob}")
            data.append((currentShipLayout, SENSE_ACTION))  # Append state and action pair
            for f in range(ship.d):
                for g in range(ship.d):
                    if ship.grid[f][g].is_open:
                        distance = ship.manhattan_distance(ship.grid[f][g], ship.mouse.cell)
                        if prob:
                            map[f][g] *= math.exp(-alpha * (distance - 1))
                        else:
                            map[f][g] *= (1 - math.exp(-alpha * (distance - 1)))

            # Values in the map will be normalized
            total_prob = sum(sum(row) for row in map)
            for f in range(ship.d):
                for g in range(ship.d):
                    map[f][g] /= total_prob

            # After sensing, move towards the cell with the highest updated probability
            hProb_cell = get_highest_cell(ship, map)

             # If highest probable cell is None, we need to handle it
            if hProb_cell is None:
                hProb_cell = ship.bot.cell  # Stay in place if no highest probability cell found

            next_cell = nextStep(ship.bot.cell, hProb_cell, ship)

            # Update recent moves
            recent_moves.append(next_cell)
            if len(recent_moves) > 2:
                recent_moves.pop(0)

            ship.bot.move(next_cell)
            ship.mouse.move(ship)
            data.append((currentShipLayout, MOVE_ACTION))
            print(f"Bot Location: ({ship.bot.cell.f}, {ship.bot.cell.g})")
            print(f"Mouse Location: ({ship.mouse.cell.f}, {ship.mouse.cell.g})")

        move = not move  # Alternate between moving and sensing

# Get the current state of the ship as a multi-dimensional array
def get_ship_state(ship):
    d = ship.d
    layout = np.zeros((d, d))
    probabilities = np.ones((d, d)) / (d * d)
    bot_position = np.zeros((d, d))

    for f in range(d):
        for g in range(d):
            if ship.grid[f][g].is_open:
                layout[f][g] = 1
            probabilities[f][g] = ship.grid[f][g].probability

    bot_position[ship.bot.cell.f][ship.bot.cell.g] = 1

    state = np.stack([layout, probabilities, bot_position], axis=0)
    
    return state

# Loop through the map to find the cell with the highest probability
def get_highest_cell(ship, map):
    hProb_cell = None
    hProb = 0

    for f in range(ship.d):
        for g in range(ship.d):
            if ship.grid[f][g].is_open:
                prob = map[f][g]
                if prob > hProb:
                    hProb = prob
                    hProb_cell = ship.grid[f][g]

    return hProb_cell

# Function is use to find the next cell that will be uesd to moved towards the target cell 
def nextStep(current: Cell, target: Cell, ship: Ship) -> Cell:
    neighbors = ship.get_neighbors(current)
    min_distance = float('inf')
    next_cell = current

    for neighbor in neighbors:
        distance = ship.manhattan_distance(neighbor, target)
        if distance < min_distance:
            min_distance = distance
            next_cell = neighbor

    return next_cell

# Predict the next step for the bot using the model and current ship layout
def predictStep(model, currentShipLayout):
    # Convert the layout to a torch tensor and make a prediction
    currentShipLayout = torch.tensor(currentShipLayout, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(currentShipLayout)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def trainedBot(ship, model):
    data = []  # To collect states and actions
    steps = 0
    while True:
        currentShipLayout = get_ship_state(ship)
        next_action = predictStep(model, currentShipLayout)

        # Apply the action value of 0 will move the bot 1 will sense
        if next_action == 0:
            next_cell = nextStep(ship.bot.cell, ship.mouse.cell, ship)
            ship.bot.move(next_cell)
            print(f"Bot moved to: ({ship.bot.cell.f}, {ship.bot.cell.g})")
        elif next_action == 1:
            prob = ship.bot.sense(ship, ship.mouse, 0.5)
            print(f"Bot sensed at: ({ship.bot.cell.f}, {ship.bot.cell.g}) with probability {prob}")
        ship.mouse.move(ship)

        steps += 1
        currentShipLayout = get_ship_state(ship)
        currentBotLocation = (ship.bot.cell.f, ship.bot.cell.g)
        data.append((currentShipLayout, currentBotLocation))

        if ship.bot.cell == ship.mouse.cell:
            return True, steps, data

        print(f"Bot Location: ({ship.bot.cell.f}, {ship.bot.cell.g})")
        print(f"Mouse Location: ({ship.mouse.cell.f}, {ship.mouse.cell.g})")

def main():
    data_collection = []
    d = 40
    print('Bot # to run:')
    bot_num = int(input())
    print('Number of simulations to run:')
    times_to_run = int(input())
    print('Alpha value for sensing:')
    alpha = float(input())
    print('Stochastic mouse? (yes/no):')
    stochastic = input().strip().lower() == 'yes'

    total_steps = 0
    output_size = 2  
    model = BotActionNetworkCNN(d, output_size)
    model.load_state_dict(torch.load('bot_action_network.pth'))

    for indef in range(times_to_run):

        ship = Ship(d)
        bot = Bot(ship)

        f = random.randint(0, d - 1)
        g = random.randint(0, d - 1)

        ship.open_cell(f, g)
        
        #Open cell in ship 
        while (ship.blocked_neighbor):
            cell_to_open_indef = random.randint(0, len(ship.blocked_neighbor) - 1)
            cell_to_open = ship.blocked_neighbor.pop(cell_to_open_indef)

            ship.open_cell(cell_to_open.f, cell_to_open.g)
        #Remove cells that cannot be spawned
        for dead_end in ship.no_enter[:]:
            if ship.single_neighbor(dead_end) is False:
                ship.no_enter.remove(dead_end)

        ship.addBot()
        ship.addMouse(stochastic)

        print("---BEGIN SIMULATION---")
        print("Bot Location: (", ship.bot.cell.f, ", ", ship.bot.cell.g, ")")
        print("Mouse Location: (" , ship.mouse.cell.f, ", ", ship.mouse.cell.g, ")")
        print(ship)

        if bot_num == 1:
            success, steps , data= bot1(ship, alpha)
        elif bot_num == 2:
            success, steps ,data  = bot2(ship, alpha)
        elif bot_num == 3:
            success, steps , data = bot3(ship, alpha)
        elif bot_num == 4:
            result, steps, data = trainedBot(ship, model)

        data_collection.extend(data)
        
        total_steps += steps
        
        print(f"Steps taken: {steps}")
        print("Final grid:")
        print(ship)
    
    #Save collected data for training
    with open('training_data.pkl', 'wb') as f:
        pickle.dump(data_collection, f)
         
    print(f"Average steps per simulation: {total_steps / times_to_run:.2f}")

if __name__ == "__main__":
    main()