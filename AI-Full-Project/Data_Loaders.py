import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        self.data = self.balance_data(self.data)

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms

        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return len(self.data)

    def balance_data(self, data):
        #add items to dict for easy count of labels
        dict = {}
        for i in range(len(data)):
            if(all(data[i][0:5] == [150, 150, 150, 150, 150])):
                np.delete(data, i, 0)
            elif data[i][6] not in dict.keys():
                dict[data[i][6]] = []
                dict[data[i][6]].append(data[i])
            else:
                dict[data[i][6]].append(data[i])
        #find min count of labels
        minNum = len(dict[0])
        for key in dict.keys():
            if(len(dict[key]) < minNum):
                minNum = len(dict[key])

        #randomly undersample data according to min count
        for key in dict.keys():
            while(len(dict[key]) > minNum):
                dict[key].pop(random.randrange(len(dict[key])))

        #reformat dictionary into a list again
        data = []
        for key in dict.keys():
            for row in dict[key]:
                data.append(row)
        return data

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
            # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
            # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
            # please do not deviate from these specifications.

        sample = self.normalized_data[idx]

        x = [float(i) for i in sample[:6]]
        y = [float(sample[6])]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return {'input': x, 'label': y}

    def get_data_loaders(self, batch_size):
        # STUDENTS: return two data.DataLoaders, one for train and one for test. Make sure to shuffle!
        # hint: look at the documentation for data.DataLoader
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        train_size = int(0.8 * len(self))
        test_size = len(self) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, test_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        self.train_loader, self.test_loader = self.nav_dataset.get_data_loaders(batch_size)


