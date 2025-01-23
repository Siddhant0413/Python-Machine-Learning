import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import pickle
from Ship import Ship
from Bot import Bot
from Mouse import Mouse
from Cell import Cell

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

class BotActionNetworkCNN(nn.Module):
    def __init__(self, d, output_size):
        super(BotActionNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #First convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #Second layer
        self.fc1 = nn.Linear(64 * d * d, 128) #Fully connected 
        self.fc2 = nn.Linear(128, output_size) #Output layer
        self.relu = nn.ReLU() #Activation
        self.flatten = nn.Flatten() #Flatten layer
        self.dropout = nn.Dropout(p=0.5) #Dropout Layer

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# Loads training data from the file 
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Training the CNN Model 
def train_model(model, optimizer, loss_function, train_data, epochs=10, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay= 0.0001)
    
    inputs = []
    targets = []

    # Prepare inputs and targets from training data
    for state, action in train_data:
        inputs.append(state)
        targets.append(action)

    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):  # Training loop for the specified number of epochs
        for batch in dataloader:
            input_batch, target_batch = batch
            optimizer.zero_grad()
            outputs = model(input_batch)
            loss = criterion(outputs, target_batch) # Loss
            loss.backward()
            optimizer.step() # Update Weights 

    return model

# Evaluate the model on test data 
def evaluate_model(model, test_data):
    model.eval()
    correct = 0
    inputs, targets = zip(*test_data)
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.long)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1) # Predictions
        correct = (predicted == targets).sum().item() # Calculate number of correct predictions
    
    accuracy = correct / len(test_data)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
     # Load data
    data = load_data('training_data.pkl')

    # Initialize model, loss function, and optimizer
    d = 40  
    output_size = 2  
    model = BotActionNetworkCNN(d, output_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Split data into training and testing sets
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]

    # Train the model
    trained_model = train_model(model, optimizer, loss_function, train_data, epochs=10, batch_size=32)
    # Evaluate the model
    evaluate_model(trained_model, test_data)

    # Save the trained model
    torch.save(trained_model.state_dict(), 'bot_action_network.pth')

