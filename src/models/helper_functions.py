import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np




def get_probabilities(input_images, model):

    #check name of model¨
    if model.config.model.name == "CNN_simple":
        outputs = model(input_images.to(model.device)).unsqueeze(0)

    elif model.config.model.name == "ConvolutionalBNN":
        outputs = []
        for i in range(5):
            model.sample()
            output = model(input_images)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=0)

    elif model.config.model.name == "BatchEnsemble_CNN":
        outputs = model(input_images.to(model.device))

    elif model.config.model.name == "Simple_rank1_CNN":
        outputs = []
        for i in range(5):
            model.sample()
            output = model(input_images)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=0)

    #get the probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    return probabilities



    #  #check if model can sample
    # if hasattr(model, 'sample'):
    #     #perform monte carlo sampling
    #     outputs = []
    #     for i in range(5):
    #         model.sample()
    #         output = model(input_images)
    #         outputs.append(output)
        
    #     #check dimensions of output to see if it is BNN or Rank1
    #     if len(output.shape) == 2:
    #         #append the outputs to the outputs tensor on the first dimension so it is a tensor of shape (4*ensemble_size, batch_size, num_classes)
    #         outputs = torch.stack(outputs, dim=0)

    #     elif len(output.shape) == 3:
    #         #append the outputs to the outputs tensor on the first dimension so it is a tensor of shape (4*ensemble_size, batch_size, num_classes)
    #         outputs = torch.cat(outputs, dim=0)

    #     # outputs = torch.stack(outputs, dim=0)
    # else:
    #     #pass the image through the model
    #     outputs = model(input_images.to(model.device))


    #get the probabilities
    probabilities = torch.nn.functional.softmax(outputs, dim=-1)
    return probabilities

def get_probabilities_dataset(data_loader, model):
    probabilities = None
    label_list = None
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(model.device), labels.to(model.device)
        batch_probabilities = get_probabilities(images, model)
        if probabilities is None:
            probabilities = batch_probabilities
            label_list = labels
        else:
            probabilities = torch.cat((probabilities, batch_probabilities), dim=1)
            label_list = torch.cat((label_list, labels), dim=0)

    return probabilities, label_list



def calculate_entropy(probabilities): 
    epsilon = 1e-10
    entropy = -torch.sum(probabilities * torch.log2(probabilities+epsilon), dim=-1)
    return entropy


def calculate_predictive_entropy(probabilities):
    # Calculate the entropy of the averaged predictions
    probabilities = torch.mean(probabilities, dim=0)

    entropy = calculate_entropy(probabilities)
    return entropy

def calculate_mutual_information(probabilities):
    # Calculate the entropy of the averaged predictions
    # entropy_average = calculate_entropy(torch.mean(probabilities, dim= (0,1))) # (0,1) because we want to average over the ensemble and the forward passes (change this when using BNN or batch ensemble)
    entropy_average = calculate_entropy(torch.mean(probabilities, dim=0))

    # Calculate the averaged entropy of the predictions
    entropy_values = calculate_entropy(probabilities)
    # entropy_average_predictions = torch.mean(entropy_values, dim=(0,1)) #same as above
    entropy_average_predictions = torch.mean(entropy_values, dim=0)

    # Mutual Information is the difference between these two values
    mutual_information = entropy_average - entropy_average_predictions


    return mutual_information


def calculate_alerotic_uncertainty(probabilities):


    # Calculate the averaged entropy of the predictions
    entropy_values = calculate_entropy(probabilities)
    entropy_average_predictions = torch.mean(entropy_values, dim=0) 

    return entropy_average_predictions


def plot_calibration_curve(y_true, y_prob, n_bins=10, name='calibration curve'):
    # Initialize lists to store accuracy and confidence for each bin
    accuracy_list = []

    # Calculate accuracy and confidence for each bin
    for m in range(1, n_bins + 1):
        # Define the bin range
        bin_lower = (m - 1) / n_bins
        bin_upper = m / n_bins
        # Get indices of samples whose predicted confidence falls into the current bin
        bin_indices = [i for i, p in enumerate(y_prob) if bin_lower < max(p) <= bin_upper]
        
        # Calculate accuracy and confidence for the current bin
        if bin_indices:
            bin_accuracy = sum([1 for i in bin_indices if y_true[i] == y_prob[i].argmax()]) / len(bin_indices)
        else:
            bin_accuracy = 0

        accuracy_list.append(bin_accuracy)

    # Calculate the width of each bin
    bin_width = 1.0 / n_bins

    # Create an array for the x-coordinates of the bars
    bar_positions = [bin_width * (m - 0.5) for m in range(1, n_bins + 1)]
    accuracy_list_well_calibrated = bar_positions
    
    # Calculate the differences
    differences = np.array(accuracy_list) - np.array(accuracy_list_well_calibrated)

    # Create masks for positive and negative differences
    positive_differences = differences > 0
    negative_differences = differences < 0

    # Convert the masks to integer indices
    positive_indices = np.where(positive_differences)[0]
    negative_indices = np.where(negative_differences)[0]


    # Plotting the reliability diagram as a bar plot

    # Plot the positive differences above the bars
    plt.bar(np.array(bar_positions)[positive_indices], differences[positive_indices], bottom=np.array(accuracy_list)[positive_indices]-differences[positive_indices], width=bin_width, align='center', edgecolor="red", facecolor="none", alpha=1.)
    # Plot the negative differences below the bars
    plt.bar(np.array(bar_positions)[negative_indices], -differences[negative_indices], bottom=np.array(accuracy_list)[negative_indices], width=bin_width, align='center', edgecolor="red", facecolor="none", alpha=1., label = 'Gap')

    # Plot the original bars
    plt.bar(bar_positions, accuracy_list, width=bin_width, align='center', alpha=0.6, label='Model Calibration', color='blue')

    # Plot the line for perfect calibration
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Average confidence')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.legend()
    #save the plot
    plt.savefig('calibration_curve.png')
    # plt.show()




def calculate_entropy_distribution(test_loader, model):
    entropies = None
    for i, (images, _) in enumerate(test_loader):
        probabilities = get_probabilities(images, model)
        entropy = calculate_predictive_entropy(probabilities)
        if entropies is None:
            entropies = entropy
        else:
            entropies = torch.cat((entropies, entropy), dim=0)

    return entropies.detach().cpu().numpy()

def calculate_alerotic_uncertainty_distribution(test_loader, model):
    uncertainties = None
    for i, (images, _) in enumerate(test_loader):
        probabilities = get_probabilities(images, model)
        uncertainty = calculate_alerotic_uncertainty(probabilities)
        if uncertainties is None:
            uncertainties = uncertainty
        else:
            uncertainties = torch.cat((uncertainties, uncertainty), dim=0)

    return uncertainties.detach().cpu().numpy()

def calculate_mutual_information_distribution(test_loader, model):
    mutual_informations = None
    for i, (images, _) in enumerate(test_loader):
        probabilities = get_probabilities(images, model)
        mutual_information = calculate_mutual_information(probabilities)
        if mutual_informations is None:
            mutual_informations = mutual_information
        else:
            mutual_informations = torch.cat((mutual_informations, mutual_information), dim=0)

    return mutual_informations.detach().cpu().numpy()



def plot_uncertainty_histograms(test_loader, test_loader_OOD, model):

    entropies = calculate_entropy_distribution(test_loader, model)
    entropies_OOD = calculate_entropy_distribution(test_loader_OOD, model)

    aleatoric_uncertainties = calculate_alerotic_uncertainty_distribution(test_loader, model)
    aleatoric_uncertainties_OOD = calculate_alerotic_uncertainty_distribution(test_loader_OOD, model)

    epistemic_uncertainties = calculate_mutual_information_distribution(test_loader, model)
    epistemic_uncertainties_OOD = calculate_mutual_information_distribution(test_loader_OOD, model)


    # find max value of all the uncertainties
    max_value = max(np.max(entropies), np.max(entropies_OOD), np.max(aleatoric_uncertainties), np.max(aleatoric_uncertainties_OOD), 
                    np.max(epistemic_uncertainties), np.max(epistemic_uncertainties_OOD))

    #ciel the max value to the nearest 0.1
    max_value = np.ceil(max_value*10)/10

    # Define the bin edges
    bin_edges = np.arange(start=0, stop=max_value+0.1, step=0.1)  # From 0 to max of uncertainties in steps of 0.1

    fig, axs = plt.subplots(1, 3, figsize=(13,5), sharey=True, tight_layout=True)

    alpha = 0.6

    axs[0].hist(entropies, bins=bin_edges, alpha=alpha, label='entropies')
    axs[0].hist(entropies_OOD, bins=bin_edges, alpha=alpha, label='entropies OOD')
    axs[0].set_title('Predictive Entropies')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].hist(aleatoric_uncertainties, bins=bin_edges, alpha=alpha, label='aleatoric uncertainties')
    axs[1].hist(aleatoric_uncertainties_OOD, bins=bin_edges, alpha=alpha, label='aleatoric uncertainties OOD')
    axs[1].set_title('Aleatoric Uncertainties')
    axs[1].set_xlabel('Value')
    axs[1].legend()

    axs[2].hist(epistemic_uncertainties, bins=bin_edges, alpha=alpha, label='epistemic uncertainties')
    axs[2].hist(epistemic_uncertainties_OOD, bins=bin_edges, alpha=alpha, label='epistemic uncertainties OOD')
    axs[2].set_title('Epistemic Uncertainties')
    axs[2].set_xlabel('Value')
    axs[2].legend()

    plt.savefig('uncertainty_histograms.png')
    
