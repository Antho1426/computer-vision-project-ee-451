import numpy as np
import matplotlib.pyplot as plt
import random

from skimage.transform import resize

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import datasets, transforms
import torchvision.transforms.functional as TF



def plot_images(X, label) :
    X = X.reshape(9, 28, 28)
    for i in range(9) :    
        plt.imshow(X[i], cmap='gray')
        plt.title('classified as : %d' %label[i])
        plt.show()


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    


def get_train_valid_loader(data_dir='data',
                           batch_size=128,
                           augment=True,
                           random_seed = 1,
                           valid_size=0.02,
                           shuffle=True,
                           show_sample=True,
                           num_workers=0, #4 initially
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[33.318447/255],
        std=[78.567444/255],
    )
    
    rotation_transform = MyRotationTransform(angles=[0, 180])
    
    # define transforms
    valid_transform = transforms.Compose([
            rotation_transform,
            transforms.ToTensor(),
            normalize,
    ])
    
    
    if augment:
        train_transform = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            rotation_transform,
            #transforms.RandomRotation((0,359), resample=False, expand=False, center=None, fill=None),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    
    
    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy().transpose([0, 2, 3, 1])
        plot_images(X, labels)
    
    return (train_loader, valid_loader)


class RotNet(nn.Module):
    def __init__(self, n_input_channels=1, n_output=10):
        super().__init__()

        # Convolutionnal layer 1
        self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.drop1 = nn.Dropout(0.25)

        
        # Fully connected layer 1
        self.fc1 = nn.Linear(64*23*23, 128)
        self.drop2 = nn.Dropout(0.25)
        # Fully connected layer 2
        self.fc2 = nn.Linear(128, n_output)
    
    def forward(self, x):
        ################################################################################
        # Set up the forward pass that the input data will go through.                 #
        # A good activation function betweent the layers is a ReLu function.           #
        #                                                                              #
        # Note that the output of the last convolution layer should be flattened       #
        # before being inputted to the fully connected layer. We can flatten           #
        # Tensor `x` with `x.view`.                                                    #
        ################################################################################
        # Forward pass of the convolutionnal layer
        x = self.conv1(x)     # 1st convolution
        x = F.relu(x)     # ReLu activation
        x = self.conv2(x)     # 2nd convolution
        x = F.relu(x)
        x = self.pool1(x)     # first max pooling
        #x = F.relu(x)     # ReLu activation
        x = self.drop1(x)
        # Flatenning
        x = x.view(-1, 64 * 23 * 23)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        #x = F.softmax(x)
        
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)


def predict_CNN(im, model, net, mu, std) :
    
    
    normalize = transforms.Normalize(
        mean=[mu/255],
        std=[std/255]
    )


    process = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    
    # Preprocessing
    im = resize(im, (28,28))
    mask = im>0.5
    im[mask] = 1
    im[~mask] = 0
    im = im.astype(np.float32)
    process(im)
    print(type(im))
    im  = torch.from_numpy(im)
    #normalize(im)
    im = im.reshape(1, 1, 28, 28)
    
    print(im[0][0])
    
    plt.imshow(im[0][0], cmap='gray')
    plt.title('Prediction input')
    plt.show()
    
    # Model loading
    checkpoint = torch.load(model)
    net.load_state_dict(checkpoint)
    
    #Prediction
    net.eval()
    with torch.no_grad():
        outputs = net(im)
        _, predicted = torch.max(F.softmax(outputs), 1)
        predicted_string = str(predicted)
    
    return predicted_string, im

