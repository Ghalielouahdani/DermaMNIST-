import numpy as np

from medmnist import DermaMNIST


def load_data(download=True):
    """
    Loads the DermaMNIST dataset and returns images and labels as NumPy arrays. 
    If download is True, the dataset will be downloaded if it does not already exist 
    and stored in the location where the medmnist package is installed.
    
    Arguments:
        download (boolean): If True, downloads the dataset if not already available.
    Returns:
        train_images (np.ndarray): Training set images, shape (N, H, W, C).
        test_images (np.ndarray): Test set images, shape (N', H, W, C).
        train_labels (np.ndarray): Training set labels, shape (N,).
        test_labels (np.ndarray): Test set labels, shape (N',).
    """
    train_dataset = DermaMNIST(split="train", download=download, size=28)
    test_dataset = DermaMNIST(split="test", download=download, size=28)

    train_images = np.stack([np.array(image) for image, _ in train_dataset])
    test_images = np.stack([np.array(image) for image, _ in test_dataset])

    train_labels = np.array([label for _, label in train_dataset]).reshape(-1)
    test_labels = np.array([label for _, label in test_dataset]).reshape(-1)
    
    return train_images, test_images, train_labels, test_labels