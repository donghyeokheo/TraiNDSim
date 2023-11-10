import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from util import Binarization

def get_transform(args, data_type):
    transform = [transforms.ToTensor()]
    if args.data_Grayscale:
        transform += [transforms.Grayscale()]
    if args.binary_mnist:
        transform += [Binarization()]
    
    if args.data_normalization:
        if (data_type == 'mnist') or (data_type == 'MNIST'):
            mean = 0.131
            if args.binary_mnist:
                mean = 0.132
            std = 0.308
        elif (data_type == 'cifar10') or (data_type == 'CIFAR10'):
            mean = [0.491, 0.482, 0.447]
            std = [0.247, 0.243, 0.262]
            if args.data_Grayscale:
                mean = 0.481
                std = 0.239
        elif (data_type == 'cifar100') or (data_type == 'CIFAR100'):
            mean = [0.507, 0.487, 0.441]
            std = [0.268, 0.257, 0.276]
            if args.data_Grayscale:
                mean = 0.487
                std = 0.251
        
        transform += [transforms.Normalize(mean=mean, std=std)]
    
    if args.image_CenterCrop:
        transform += [transforms.Resize(args.image_CenterCrop)]
    if args.image_Resize:
        transform += [transforms.Resize(args.image_Resize)]
    
    test_trans = transforms.Compose(transform)
    
    train_transform = []
    if args.image_RandomCrop:
        train_transform += [transforms.RandomCrop(args.image_RandomCrop)]
    if args.image_RandomHorizontalFlip:
        train_transform += [transforms.RandomHorizontalFlip()]
    if args.image_RandomVerticalFlip:
        train_transform += [transforms.RandomVerticalFlip()]
    if args.image_RandomRotation:
        train_transform += [transforms.RandomRotation(args.image_RandomRotation)]
    
    train_transform += transform
    train_trans = transforms.Compose(train_transform)
    
    return train_trans, test_trans

def dataset(args, data_type, root='dataset'):
    assert ((data_type == 'mnist') or (data_type == 'MNIST') or
            (data_type == 'cifar10') or (data_type == 'CIFAR10') or
            (data_type == 'cifar100') or (data_type == 'CIFAR100')), \
            'There is no choice rather than MNIST, CIFAR10, CIFAR100'

    train_transform, test_transform = get_transform(args, data_type)
    if (data_type == 'mnist') or (data_type == 'MNIST'):
        root += '/MNIST'
        train_set = datasets.MNIST(root=root,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_set = datasets.MNIST(root=root,
                                train=False,
                                transform=test_transform,
                                download=True)

    elif (data_type == 'cifar10') or (data_type == 'CIFAR10'):
        
        root += '/CIFAR10'
        train_set = datasets.CIFAR10(root=root,
                                    train=True,
                                    transform=train_transform,
                                    download=True)
        test_set = datasets.CIFAR10(root=root,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    elif (data_type == 'cifar100') or (data_type == 'CIFAR100'):
        root += '/CIFAR100'
        train_set = datasets.CIFAR100(root=root,
                                      train=True,
                                      transform=train_transform,
                                      download=True)
        test_set = datasets.CIFAR100(root=root,
                                     train=False,
                                     transform=test_transform,
                                     download=True)
    
    return train_set, test_set

def dataloader(args, **kwargs):
    train, test = dataset(args, args.data_type, args.data_root)
    kwargs.setdefault('shuffle', True)
    train_loader = DataLoader(dataset=train,
                            batch_size=args.batch_size,
                            **kwargs)
    test_loader = DataLoader(dataset=test,
                            batch_size=args.batch_size,
                            **kwargs)

    return train_loader, test_loader


