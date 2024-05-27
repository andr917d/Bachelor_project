import wandb
import hydra
from omegaconf import OmegaConf
import numpy as np
import requests
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

class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
 

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
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64, shuffle=False)



# Function to download and save a file given its URL
def download_file(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")


def load_Unown_MNIST_pytorch():
    # URLs for the test images and labels
    # test_images_url = 'https://github.com/lopeLH/unown-mnist/raw/main/X_test.npy'
    # Download the test images
    # download_file(test_images_url, 'unown_mnist_test_images.npy')
    # Load the test images
    test_images = np.load('unown_mnist_test_images.npy')
    # Convert the images to PyTorch tensors
    test_images_tensor = torch.tensor(test_images, dtype=torch.float)
    test_images_tensor = test_images_tensor.unsqueeze(1)

    #add taget labels (just some random labels since we are not going to use them)
    test_labels = np.random.randint(0, 10, test_images_tensor.shape[0])


    # Create a DataLoader for the test images
    test_loader = DataLoader(TensorDataset(test_images_tensor, torch.tensor(test_labels)), batch_size=64, shuffle=False)
    return test_loader


def load_mnist_OOD_pytorch():
    #we could change this to take any transform
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        GaussianNoiseTransform(0., 0.2)])  # Apply Gaussian noise with mean 0 and standard deviation 0.1
    
    # Load the MNIST dataset
    OOD_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(OOD_dataset, batch_size=64, shuffle=False)

def load_cifar10_OOD_pytorch():
    #we could change this to take any transform
    # Define the transformations to apply to the images
    transform = transforms.Compose([
        transforms.ToTensor(),
        GaussianNoiseTransform(0., 0.2)])  # Apply Gaussian noise with mean 0 and standard deviation 0.1
    
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
    elif model_name == "CNN_MCD":
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
        test_loader_OOD = load_cifar10_OOD_pytorch()

    elif dataset_name == "cifar100":
        train_loader, test_loader = load_cifar100_pytorch()
    elif dataset_name == "mnist":
        train_loader, test_loader = load_mnist_pytorch()
        test_loader_OOD = load_Unown_MNIST_pytorch()

    elif dataset_name == "forest_cover":
        train_loader, test_loader = load_forest_cover_pytorch()
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


    model = init_model(config.model.name, config)
    # torch.manual_seed(config.constants.seed)

    model.train_custom(train_loader, test_loader)

    #calibration curve
    probabilities, labels = get_probabilities_dataset(test_loader, model)

    #calculate negative log likelihood
    NLL = calculate_cross_entropy(labels, probabilities)
    print(f"NLL: {NLL}")

    #take softmax over the probabilities to get the probabilities
    probabilities = torch.nn.functional.softmax(probabilities, dim=-1)
    probabilities = probabilities.numpy()
    labels = labels.numpy()


    # probabilities = probabilities.mean(dim=0).cpu().detach().numpy() #average over the forward passes (ensemble members)
    # labels = labels.cpu().detach().numpy()
    n_bins = 10
    plot_calibration_curve(labels, probabilities, n_bins, name=config.bsub.name, save_name=f'calibration_curve_test_{config.bsub.name}.png')

    #calculate accuracy
    accuracy = calculate_accuracy(labels, probabilities)
    print(f"Accuracy: {accuracy}")


    # calculate ECE
    ECE = calculate_ECE(labels, probabilities, n_bins)
    print(f"ECE: {ECE}")


    


    # #calibration curve for OOD
    # probabilities_OOD, labels_OOD = get_probabilities_dataset(test_loader_OOD, model)

    # NLL_OOD = calculate_cross_entropy(labels_OOD, probabilities_OOD)
    # print(f"NLL OOD: {NLL_OOD}")

    # #take softmax over the probabilities to get the probabilities
    # probabilities_OOD = torch.nn.functional.softmax(probabilities_OOD, dim=-1)
    # probabilities_OOD = probabilities_OOD.numpy()
    # labels_OOD = labels_OOD.numpy()

    # plot_calibration_curve(labels_OOD, probabilities_OOD, n_bins, name=config.bsub.name, save_name=f'calibration_curve_OOD_{config.bsub.name}.png')

    # #calculate accuracy
    # accuracy_OOD = calculate_accuracy(labels_OOD, probabilities_OOD)
    # print(f"Accuracy OOD: {accuracy_OOD}")

    # # calculate ECE
    # ECE_OOD = calculate_ECE(labels_OOD, probabilities_OOD, n_bins)
    # print(f"ECE OOD: {ECE_OOD}")



    #log to wandb
    wandb.log({"accuracy": accuracy, "NLL": NLL, "ECE": ECE})
    # wandb.log({"accuracy": accuracy, "NLL": NLL, "ECE": ECE, "accuracy_OOD": accuracy_OOD, "NLL_OOD": NLL_OOD, "ECE_OOD": ECE_OOD})


    #OOD detection
    plot_uncertainty_histograms(test_loader, test_loader_OOD, model)

    #plot rotated images
    plot_rotated_image(test_loader, model, label_number=6)




    
    
if __name__ == "__main__":
    main()