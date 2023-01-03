import argparse
import sys
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import click
import matplotlib.pyplot as plt

from data import LoadCorruption, ToTenzor
from model import ConvNet


@click.group()
def cli():
    pass


# Paths for data
basepath = '../../data/corruptmnist'
trainpaths = [f'{basepath}/{path}' for path in os.listdir(basepath) if 'train' in path]
testpaths = [f'{basepath}/{path}' for path in os.listdir(basepath) if 'test' in path] 

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=5, help='learning rate to use for training')
def train(lr, epochs):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    train_set = DataLoader(LoadCorruption(*trainpaths, transform=ToTenzor()), batch_size=8, shuffle=True)
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        running_loss = 0
        for i, (images, labels) in enumerate(train_set):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, torch.flatten(labels))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss/len(train_set)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        losses.append(epoch_loss)

    torch.save(model.state_dict(), 'cnn_checkpoint.pth')

    plt.figure()
    plt.plot([i+1 for i in range(epochs)],losses); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.show()



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = ConvNet()
    state_dict = torch.load('cnn_checkpoint.pth')
    model.load_state_dict(state_dict)
    test_set = DataLoader(LoadCorruption(*testpaths, transform=ToTenzor()), batch_size=8, shuffle=True)

    with torch.no_grad():
        accuracy = torch.tensor([0], dtype=torch.float)
        for images, labels in test_set:
            log_ps = model(images)
            top_p, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        
        print(f'Accuracy: {accuracy.item()*100/len(test_set)}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    