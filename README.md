# Robot Navigation with Neural Network

This project involves collecting training data from a wandering robot in a pyGame simulated environment and using the data to train a neural network to avoid obstacles, specifically walls. The simulation records data on the actions taken by the robot and its distance to walls at 5 different angles from the front.

## Features

- Data collection from a wandering robot in a pyGame simulated environment.
- Neural network training to avoid obstacles (walls).
- Data includes robot actions and distances to walls at 5 different angles.
- Simple Linear Neural Network with two dropout layers and ReLU activation function.

## Neural Network Architecture

The neural network architecture is a simple Linear NN with two dropout layers and a ReLU activation function. The network predicts actions based on the collected data. The training process involves using a Mean Squared Error (MSE) loss function to backpropagate the error through the network.

## Training Results

With a learning rate of 0.001, the network achieved a loss of 0.05 after approximately 45 epochs.

## Installation

To run the robot navigation project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/robot-navigation.git
