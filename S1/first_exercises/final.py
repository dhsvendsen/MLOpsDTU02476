import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

# Hyper-parameters 
num_epochs = 1
batch_size = 4
learning_rate = 0.001


class LoadCorruption(Dataset):
    def __init__(self, *filepath, transform=None):
        content = [np.load(path) for path in filepath]
        images = [data['images'] for data in content]
        labels = [data['labels'] for data in content]
        self.images, self.labels = np.concatenate(images), np.concatenate(labels).reshape(-1,1)
    
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        sample = np.expand_dims(self.images[index], axis=0), self.labels[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

# dataset transforms
class ToTenzor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)).type(torch.LongTensor)

# Paths for data
basepath = '../../data/corruptmnist'
trainpaths = [f'{basepath}/{path}' for path in os.listdir(basepath) if 'train' in path]
testpaths = [f'{basepath}/{path}' for path in os.listdir(basepath) if 'test' in path] 

# Load data
train = LoadCorruption(*trainpaths, transform=ToTenzor())
test = LoadCorruption(*testpaths, transform=ToTenzor())

trainloader = DataLoader(dataset=train,
    batch_size=batch_size,
    shuffle=True)

testloader = DataLoader(dataset=test,
    batch_size=batch_size,
    shuffle=False)



# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
print(images.shape)
print(labels.shape)
plt.imshow(images[0][0].numpy())
plt.show()


# This is from a tutorial i followed https://github.com/patrickloeber/pytorchTutorial/blob/master/14_cnn.py
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 16, 12, 12
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 5, 5
        x = x.view(-1, 32 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x


model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(trainloader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainloader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, torch.flatten(labels))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in testloader:
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of digit {i}: {acc} %')

print(n_samples)