import wandb
import hydra
from omegaconf import OmegaConf
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.datasets import fetch_covtype
from torch import optim
import matplotlib.pyplot as plt
from architectures import Simple_rank1_CNN, BatchEnsemble_CNN, BNN_rank1, BatchEnsemble_FFNN, BNN, FFNN_simple, FFNN_DeepEnsemble, CNN_DeepEnsemble, ConvolutionalBNN, CNN_simple
# from helper_functions import get_probabilities, get_probabilities_dataset, calculate_entropy, calculate_predictive_entropy, calculate_mutual_information, plot_calibration_curve

#import all helper functions
from helper_functions import *


def load_cifar10_pytorch():
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64, shuffle=False)

def load_cifar100_pytorch():
    train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64, shuffle=False)

def load_mnist_pytorch():
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
 

def load_cifar10_OOD_pytorch():
    #we could change this to take any transform



    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        GaussianNoiseTransform(0., 0.1)])  # Apply Gaussian noise with mean 0 and standard deviation 0.1
    
    # Load the CIFAR-10 dataset
    OOD_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    return DataLoader(OOD_dataset, batch_size=64, shuffle=False)
    



def load_forest_cover_pytorch(test_split=0.2, batch_size=64):
    # Fetch the dataset
    dataset = fetch_covtype()
    data, targets = dataset.data, dataset.target - 1

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    # Split dataset into training and test sets
    total_size = len(data_tensor)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    train_dataset, test_dataset = random_split(TensorDataset(data_tensor, targets_tensor), [train_size, test_size])

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def init_model(model_name, config):
    if model_name == "BatchEnsemble_CNN":
        return BatchEnsemble_CNN(config)
    elif model_name == "Simple_rank1_CNN":
        return Simple_rank1_CNN(config)
    elif model_name == "BNN_rank1":
        return BNN_rank1(config)
    elif model_name == "BatchEnsemble_FFNN":
        return BatchEnsemble_FFNN(config)
    elif model_name == "BNN":
        return BNN(config)
    elif model_name == "FFNN_simple":
        return FFNN_simple(config)
    elif model_name == "FFNN_DeepEnsemble":
        return FFNN_DeepEnsemble(config)
    elif model_name == "CNN_DeepEnsemble":
        return CNN_DeepEnsemble(config)
    elif model_name == "ConvolutionalBNN":
        return ConvolutionalBNN(config)
    elif model_name == "CNN_simple":
        return CNN_simple(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}")



@hydra.main(config_name="config.yaml", config_path="./", version_base="1.3")
def main(config):
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    
    # # Initiate wandb logger
    try:
        # project is the name of the project in wandb, entity is the username
        # You can also add tags, group etc.
        run = wandb.init(project=config.wandb.project, 
                   config=OmegaConf.to_container(config), 
                   entity=config.wandb.entity)
        print(f"wandb initiated with run id: {run.id} and run name: {run.name}")
    except Exception as e:
        print(f"\nCould not initiate wandb logger\nError: {e}")

    # Usual PyTorch training code using a Trainer class
    dataset_name = config.data.dataset_name
    if dataset_name == "cifar10":
        train_loader, test_loader = load_cifar10_pytorch()
        test_loader_OOD = load_cifar10_ODD_pytorch()

    elif dataset_name == "cifar100":
        train_loader, test_loader = load_cifar100_pytorch()
    elif dataset_name == "mnist":
        train_loader, test_loader = load_mnist_pytorch()
    elif dataset_name == "forest_cover":
        train_loader, test_loader = load_forest_cover_pytorch()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


    model = init_model(config.model.name, config)
    # torch.manual_seed(config.constants.seed)

    model.train(train_loader, test_loader)

    #calibration curve
    probabilities, labels = get_probabilities_dataset(test_loader, model)

    probabilities = probabilities.mean(dim=0).cpu().detach().numpy() #average over the forward passes (ensemble members)
    labels = labels.cpu().detach().numpy()
    n_bins = 10
    plot_calibration_curve(labels, probabilities, n_bins, name=config.bsub.name)

    #OOD detection
    plot_uncertainty_histograms(test_loader, test_loader_OOD, model)




    
    
if __name__ == "__main__":
    main()