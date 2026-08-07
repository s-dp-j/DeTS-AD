import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod



class MSLSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL/MSL_train.npy")
        self.scaler.fit(data)
        print("Before normalization - Min: {}, Max: {}, Mean: {}, Std: {}".format(
            np.min(data), np.max(data), np.mean(data), np.std(data)))
        data = self.scaler.transform(data)
        print("After normalization - Min: {}, Max: {}, Mean: {}, Std: {}".format(
            np.min(data), np.max(data), np.mean(data), np.std(data)))
        test_data = np.load(data_path + "/MSL/MSL_test.npy")
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.test_labels = np.load(data_path + "/MSL/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.train.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(np.zeros(self.win_size))
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])

class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])class BaseSegLoader(Dataset, ABC):
    
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
    
    @abstractmethod
    def __len__(self):
    
        pass
    
    @abstractmethod
    def __getitem__(self, index):
       
        pass

            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        
class SWaTSegLoader(BaseSegLoader):

    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation is not included in the open-source version.")
    
    def __len__(self):
        raise NotImplementedError("Implementation.")
    
    def __getitem__(self, index):
        raise NotImplementedError("Implementation.")


class PSMSegLoader(BaseSegLoader):
  
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation.")


class SMAPSegLoader(BaseSegLoader):
 
    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation.")


class SMDSegLoader(BaseSegLoader):

    def __init__(self, data_path, win_size, step, mode="train"):
        super().__init__(data_path, win_size, step, mode)
        raise NotImplementedError("Implementation.")


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', val_ratio=0.2):
    if dataset == 'SMD':
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, step, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoader(data_path, win_size, step, mode)
    elif dataset == 'SWaT':
        dataset = SWaTSegLoader(data_path, win_size, step, mode)


    shuffle = False

    if mode == 'train':
        shuffle = True

        dataset_len = int(len(dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))

        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)

        indices = torch.arange(dataset_len)
        
        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(dataset, val_sub_indices)
        
        train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=shuffle)


        return train_loader, val_loader

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader
