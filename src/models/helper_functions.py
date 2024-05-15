import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np




def get_probabilities(input_images, model):

     #check if model can sample
    if hasattr(model, 'sample'):
        #perform monte carlo sampling
        outputs = []
        for i in range(5):
            model.sample()
            output = model(input_images)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=0)
    else:
        #pass the image through the model
        outputs = model(input_images.to(model.device))


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


def plot_calibration_curve(y_true, y_prob, n_bins=10):
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
    plt.title('Reliability Diagram')
    plt.legend()
    #save the plot
    plt.savefig('calibration_curve.png')
    # plt.show()
