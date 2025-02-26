import torch
from torchvision import datasets, transforms

def load_cifar10_dataset(data_dir, use_augmentation=True, normalize=True):
    """
    Load CIFAR-10 dataset with optional augmentation and normalization.
    
    Args:
        data_dir: Directory where the dataset is stored or will be downloaded
        use_augmentation: Whether to use data augmentation for training
        normalize: Whether to normalize the data
        
    Returns:
        train_dataset, val_dataset: PyTorch datasets for training and validation
    """
    # Define transforms
    transform_list = []
    
    # Add augmentation if requested
    if use_augmentation:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    
    # Always convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Add normalization if requested
    if normalize:
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        )
    
    # Create transform compositions
    transform_train = transforms.Compose(transform_list)
    
    # Test transform (no augmentation)
    test_transform_list = [transforms.ToTensor()]
    if normalize:
        test_transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
        )
    transform_test = transforms.Compose(test_transform_list)
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    return train_dataset, val_dataset 