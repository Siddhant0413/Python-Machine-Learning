Input: The input data is some representation of the ship and the current state of knowledge/observations.
Output: The output is some choice of movement (up/down/left/right as appropriate for its current position),
or sensing.
Model: Parameterized and learnable, that maps from input to output.

The input can be split into three different things, the ship's layout, probability grid, and the bot's current position. All of these different inputs are stacked together to create a single input tensor with the shape ‘(3, 40, 40)’. 
The model uses grids so it uses the CNN model. 
The data set that is collected through the bots in the main method consists of the state-action pairs through the ship's environment. When we run the 3 bots in the main driver code, we capture each move that the bot is taking during 
both the scenarios of a stationary and moving mouse. At every timestep, we are capturing the current state of the ship, the bot’s knowledge/observations, and the action that the bot has made.
Each bot would be then run for a number of simulations each having a different alpha value incrementing by 0.1. The data from these simulations consisted of the ship at every given time step with the
probability and action that the bot has taken. All of this data was then loaded into tensors that were batched using a DataLoader.
