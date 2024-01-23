from collect_data import collect_training_data
from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os

def main():
    pass


    # Load Data
    # batch_size = 16
    # data_loaders = Data_Loaders(batch_size)
    #
    # for idx, sample in enumerate(data_loaders.train_loader):
    #     training_input, training_label = sample['input'], sample['label']
    # for idx, sample in enumerate(data_loaders.test_loader):
    #     _, _ = sample['input'], sample['label']
    #
    #
    # model = Action_Conditioned_FF()

    #
    # print(model.evaluate(model, data_loaders.train_loader, nn.MSELoss()))

def train_model(no_epochs):
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    losses = []
    min_loss = float('inf')

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0.0

        for idx, sample in enumerate(data_loaders.train_loader):
            inputs, labels = sample['input'], sample['label']
            #Feed data through the network and backpropagate the loss
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(data_loaders.train_loader)
        losses.append(avg_loss)


        model.eval()
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)

        # Save the model with the minimum test loss
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), 'saved/saved_model.pkl', _use_new_zipfile_serialization=False)

        print(f'Epoch {epoch_i + 1}/{no_epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}')

    # Plot the loss
    plt.plot(losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

if __name__ == '__main__':
    # Run Pygame Simulation and Log Data
    # total_actions = 10000
    # collect_training_data(total_actions)
    no_epochs = 150
    train_model(no_epochs)

