# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score

#  ---------------  Dataset  ---------------

class TimeSinceSonarExposureDataset(Dataset):
    """Acoustics Performance dataset"""

    def __init__(self, csv_file):
        """
        Initializes instance of class UnderwaterAcoustic
        Args:
            csv_file (str): Path to the csv file with the acoustic data
        """
        df = pd.read_csv(csv_file)  # Read .csv file

        self.categorical = ["day_of_week", "time_of_day"]
        self.target = "time_since_exposure"

        # StudentsPerformance
        # self.categorical = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
        # self.target = "math score"

        # One-hot encoding of categorical variables
        self.exposure_frame = pd.get_dummies(df, columns=self.categorical, prefix=self.categorical)

        # Save target and predictors
        self.X = self.exposure_frame.drop(self.target, axis=1)
        self.y = self.exposure_frame[self.target]

    def __len__(self):
        return len(self.exposure_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values.astype(np.float32), self.y[idx]]

#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H=15, D_out=1):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x.squeeze()

# Calculate F1 score
def calculate_f1_score(y_true, y_pred):
    """
    Measure model performance with precision/recall
    """
    tp = ((y_true == 1) & (y_pred == 1))
    fp = ((y_true == 0) & (y_pred == 1))
    fn = ((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    """
    Determine accuracy of predictions
    """
    correct = (y_true == y_pred)
    return correct / len(y_true)

# Calculate MSE
def calculate_mse(y_true, y_pred):
    """
    Determine mean square error
    """
    # Convert to arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    error = (y_true - y_pred) ** 2
    return np.mean(error)

#  ---------------  Training  ---------------

def train(csv_file, n_epochs=100):
    """Trains the model.

    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    # Load dataset
    dataset = TimeSinceSonarExposureDataset(csv_file)

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model with the correct input dimension
    D_in, H = len(dataset.X.columns), 15
    net = Net(D_in, H).to(device)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), weight_decay=0.0001)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []
    test_loss_per_epoch = []

    for epoch in range(n_epochs):
        net.train()

        # Training phase
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []

        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()

            # Collect predictions and labels for metrics
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(outputs.cpu().detach().numpy())

        loss_per_batch.append(running_loss / len(trainloader))

        # Comparing training to test
        y_true_train_binary = [1 if label >= 0.5 else 0 for label in y_true_train]
        y_pred_train_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_train]

        train_accuracy = calculate_accuracy(y_true_train_binary, y_pred_train_binary)
        mse_score = calculate_mse(y_true_train_binary, y_pred_train_binary)
        train_precision = precision_score(y_true_train_binary, y_pred_train_binary)
        train_f1 = calculate_f1_score(y_true_train_binary, y_pred_train_binary)

        epoch_loss = running_loss / len(trainloader)
        loss_per_iter.append(epoch_loss)

        # Validation phase
        net.eval()
        test_loss = 0.0
        y_true_test = []
        y_pred_test = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs.float())
                test_loss += criterion(outputs, labels.float()).item()

                # Store predictions and true values
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(outputs.cpu().numpy())

        # Calculate test metrics
        y_true_test_binary = [1 if label >= 0.5 else 0 for label in y_true_test]
        y_pred_test_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_test]

        test_accuracy = calculate_accuracy(y_true_test_binary, y_pred_test_binary)
        test_accuracy = test_accuracy
        test_precision = precision_score(y_true_test_binary, y_pred_test_binary)
        test_f1 = calculate_f1_score(y_true_test_binary, y_pred_test_binary)

        test_loss = test_loss / len(testloader)
        test_loss_per_epoch.append(test_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print("Epoch", epoch + 1, "/", n_epochs,
                  "\nTrain Loss:", epoch_loss,
                  "\nTest Loss:", test_loss,
                  "\nTrain Accuracy:", train_accuracy,
                  "\nTest MSE:", mse_score,
                  "\nTest Accuracy:", test_accuracy,
                  "\nTrain Precision:", train_precision,
                  "\nTest Precision:", test_precision,
                  "\nTrain F1:", train_f1,
                  "\nTest F1:", test_f1)

    # Final evaluation
    print("\nFinal Results:")
    print("Training Loss Over Epochs:", loss_per_iter)
    print("Validation Loss Over Epochs:", test_loss_per_epoch)

    # Plot training loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(loss_per_iter, label="Training Loss")
    plt.plot(test_loss_per_epoch, label="Validation Loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("NNProj.png")
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    # By default, read csv file in the same directory as this script
    csv_file = os.path.join(sys.path[0], "Before_During_After_Exposure_0601_0719.csv")

    # StudentsPerformance
    # csv_file = os.path.join(sys.path[0], "StudentsPerformance.csv")

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                        help="Dataset file used for training")
    parser.add_argument("--epochs", "-e", type=int, nargs="?", default=100, help="Number of epochs to train")
    args = parser.parse_args()

    # Call the main function of the script
    train(args.file, args.epochs)
