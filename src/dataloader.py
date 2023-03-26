import cv2
import numpy as np
import os
import random
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
import torchvision
import torch
from ot.bregman import sinkhorn
from ot.lp import emd


def get_random_colored_images(images, seed = 0x000000):
    np.random.seed(seed)
    
    images = 0.5*(images + 1)
    size = images.shape[0]
    colored_images = []
    hues = 360*np.random.rand(size)
    
    for V, H in zip(images, hues):
        V_min = 0
        
        a = (V - V_min)*(H%60)/60
        V_inc = a
        V_dec = V - a
        
        colored_image = torch.zeros((3, V.shape[1], V.shape[2]))
        H_i = round(H/60) % 6
        
        if H_i == 0:
            colored_image[0] = V
            colored_image[1] = V_inc
            colored_image[2] = V_min
        elif H_i == 1:
            colored_image[0] = V_dec
            colored_image[1] = V
            colored_image[2] = V_min
        elif H_i == 2:
            colored_image[0] = V_min
            colored_image[1] = V
            colored_image[2] = V_inc
        elif H_i == 3:
            colored_image[0] = V_min
            colored_image[1] = V_dec
            colored_image[2] = V
        elif H_i == 4:
            colored_image[0] = V_inc
            colored_image[1] = V_min
            colored_image[2] = V
        elif H_i == 5:
            colored_image[0] = V
            colored_image[1] = V_min
            colored_image[2] = V_dec
        
        colored_images.append(colored_image)
        
    colored_images = torch.stack(colored_images, dim = 0)
    colored_images = 2*colored_images - 1
    
    return colored_images


def load_dataset(name, path, img_size=64, batch_size=64, shuffle=True, device='cuda'):
    if name.startswith("MNIST"):
        # In case of using certain classe from the MNIST dataset you need to specify them by writing in the next format "MNIST_{digit}_{digit}_..._{digit}"
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: 2 * x - 1)
        ])
        
        dataset_name = name.split("_")[0]
        is_colored = dataset_name[-7:] == "colored"
        
        classes = [int(number) for number in name.split("_")[1:]]
        if not classes:
            classes = [i for i in range(10)]
        
        train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(path, train=False, transform=transform, download=True)
        
        train_test = []
        
        for dataset in [train_set, test_set]:
            data = []
            labels = []
            for k in range(len(classes)):
                data.append(torch.stack(
                    [dataset[i][0] for i in range(len(dataset.targets)) if dataset.targets[i] == classes[k]],
                    dim=0
                ))
                labels += [k]*data[-1].shape[0]
            data = torch.cat(data, dim=0)
            data = data.reshape(-1, 1, 32, 32)
            labels = torch.tensor(labels)
            
            if is_colored:
                data = get_random_colored_images(data)
            
            
            train_test.append(data)
            
        train_set, test_set = train_test  
    else:
        raise Exception('Unknown dataset')
        
    return train_set, test_set


def load_shoes_bags(data_path="./shoes_tensor_32.torch",total_size=11000):
    data = torch.load(data_path)
    
    random_idx = np.random.choice(np.arange(len(data)), size=total_size, replace=False)
    data = data[random_idx]
    
    np.random.shuffle(random_idx)
    train_data, test_data = data[:10000, :], data[10000:, :]
    assert len(train_data) == total_size - 1000
    return train_data, test_data


class ShoeBagDataset(Dataset):
    def __init__(self, data_from, data_to):
        self.data_from = data_from
        self.data_to = data_to

    def __len__(self):
        return len(self.data_from)

    def __getitem__(self, idx: int):
        """
            returns single sample
        """
        return self.data_from[idx], self.data_to[idx]
    

def collate_fn_emd(batch) -> tuple:
    batch_size = len(batch)

    data_from = torch.Tensor([b[0].tolist() for b in batch])
    data_to = torch.Tensor([b[1].tolist() for b in batch])

    distance = (torch.cdist(data_from.reshape(batch_size, -1), data_to.reshape(batch_size, -1))**2)/2
    distance = distance.numpy()

    epsilon = 10

    distance = distance/(3*32*32)
    epsilon = epsilon/(3*32*32)

    a = np.ones(batch_size)/batch_size
    b = np.ones(batch_size)/batch_size
    map_probs = emd(a, b, distance)

    to = []
    for b in range(batch_size):

        tranposrt_idx = np.random.choice(np.arange(batch_size), size=1, replace=True, p=map_probs[b]/(map_probs[b].sum()))
        img = data_to[tranposrt_idx[0]]
        to.append(img.tolist())
    
    return data_from, torch.Tensor(to)


def collate_fn_sinkhorn(batch) -> tuple:
    batch_size = len(batch)

    data_from = torch.Tensor([b[0].tolist() for b in batch])
    data_to = torch.Tensor([b[1].tolist() for b in batch])

    distance = (torch.cdist(data_from.reshape(batch_size, -1), data_to.reshape(batch_size, -1))**2)/2
    distance = distance.numpy()

    epsilon = 10

    distance = distance/(3*32*32)
    epsilon = epsilon/(3*32*32)

    a = np.ones(batch_size)/batch_size
    b = np.ones(batch_size)/batch_size
    map_probs = sinkhorn(a, b, distance, reg=epsilon, warn=True, verbose=False, numItermax=100000)

    to = []
    for b in range(batch_size):

        tranposrt_idx = np.random.choice(np.arange(batch_size), size=1, replace=True, p=map_probs[b]/(map_probs[b].sum()))
        img = data_to[tranposrt_idx[0]]
        to.append(img.tolist())
    
    return data_from, torch.Tensor(to)


class ColoredDataset(Dataset):
    def __init__(self, data_from, data_to):
        self.data_from = data_from
        self.data_to = data_to

    def __len__(self):
        return len(self.data_from)

    def __getitem__(self, idx: int):
        """
            returns single sample
        """
        return self.data_from[idx], self.data_to[idx]