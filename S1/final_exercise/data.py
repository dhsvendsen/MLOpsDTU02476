import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

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

class ToTenzor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)).type(torch.LongTensor)