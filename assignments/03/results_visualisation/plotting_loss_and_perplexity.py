import re
import matplotlib.pyplot as plt
import os

# Specify the folder containing your model files
folder_path = "atmt_2024\assignments\03\results_visualisation"  # The path to folder containing log files as .txt files

# Function to extract data from model files
def extract_data(folder_path):
    model_data = {}

    # Loop through each .txt file in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            model_name = os.path.splitext(file_name)[0]  # Get model name from file name
            epochs = []
            valid_losses = []
            valid_perplexities = []

            # Open and read each file
            with open(os.path.join(folder_path, file_name), 'r') as file:
                for line in file:
                    # Find Epoch and valid_loss using regex
                    epoch_match = re.search(r'Epoch (\d+):', line)
                    valid_loss_match = re.search(r'valid_loss ([\d.]+)', line)
                    valid_perplexity_match = re.search(r'valid_perplexity ([\d.]+)', line)

                    # If both Epoch and valid_loss are found, extract and store them
                    if epoch_match and valid_loss_match:
                        epoch = int(epoch_match.group(1))
                        valid_loss = float(valid_loss_match.group(1))
                        epochs.append(epoch)
                        valid_losses.append(valid_loss)

                    # If both Epoch and valid_perplexity are found, extract and store them
                    if epoch_match and valid_perplexity_match:
                        valid_perplexity = float(valid_perplexity_match.group(1))
                        valid_perplexities.append(valid_perplexity)

            # Store the data in the dictionary with model name as key
            model_data[model_name] = (epochs, valid_losses, valid_perplexities)

    return model_data

# Function to plot validation loss
def plot_validation_loss(model_data):
    plt.figure(figsize=(10, 6))

    for model_name, (epochs, valid_losses, _) in model_data.items():
        plt.plot(epochs, valid_losses, linestyle='-', label=f'{model_name}')

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss per Epoch for Different Models')
    plt.legend()
    plt.show()

# Function to plot perplexity
def plot_perplexity(model_data):
    plt.figure(figsize=(10, 6))

    for model_name, (epochs, _, valid_perplexities) in model_data.items():
        plt.plot(epochs, valid_perplexities, linestyle='-', label=f'{model_name}')

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity per Epoch for Different Models')
    plt.legend()
    plt.show()

# Extract data from the files
model_data = extract_data(folder_path)

# Plot the results
plot_validation_loss(model_data)
plot_perplexity(model_data)
