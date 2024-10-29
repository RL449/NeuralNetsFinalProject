import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


#  ---------------  Dataset  ---------------

class StudentsPerformanceDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Define categorical columns - only include columns that actually exist in the dataset
        self.categorical = []
        for col in ["index", "splpk", "splrms", "dissim", "impulsivity", "peakcount"]:
            if col in df.columns:
                self.categorical.append(col)

        self.target = "time_since_exposure"

        # Verify target exists
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in the dataset")

        # One-hot encoding of categorical variables - only encode columns that exist
        if self.categorical:
            # Create prefix dictionary mapping each categorical column to itself as prefix
            prefix_dict = {col: col for col in self.categorical}
            self.students_frame = pd.get_dummies(df, columns=self.categorical, prefix=prefix_dict)
        else:
            self.students_frame = df.copy()
            print("Warning: No categorical columns found in dataset")

        # Save target and predictors
        self.X = self.students_frame.drop(self.target, axis=1)
        self.y = self.students_frame[self.target]

        # Print dataset info
        print(f"Dataset loaded with {len(self.X.columns)} features and {len(self.students_frame)} samples")
        if self.categorical:
            print(f"Categorical columns encoded: {', '.join(self.categorical)}")

    def __len__(self):
        return len(self.students_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values.astype(np.float32), self.y[idx]]


#  ---------------  Model  ---------------

class Net(nn.Module):

    def __init__(self, D_in, H1=1024, H2=256, H3=64, D_out=1):
        """
        Args:
            D_in (int): Input dimension (number of features)
            H1 (int): First hidden layer size
            H2 (int): Second hidden layer size
            H3 (int): Third hidden layer size
            D_out (int): Output dimension
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(D_in, H1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H2, H3),
            nn.ReLU(),
            nn.Linear(H3, D_out)
        )

    def forward(self, x):
        return self.network(x).squeeze()


#  ---------------  Training  ---------------

def train(csv_file, n_epochs=100):
    """Trains the model.
    Args:
        csv_file (str): Absolute path of the dataset used for training.
        n_epochs (int): Number of epochs to train.
    """
    # Load dataset
    dataset = StudentsPerformanceDataset(csv_file)

    # Get input dimension from the dataset
    D_in = len(dataset.X.columns)
    print(f"Input dimension: {D_in}")

    # Split into training and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Dataloaders with smaller batch size due to larger model
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model with the correct input dimension
    net = Net(D_in).to(device)
    print(f"Model architecture:\n{net}")

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer with learning rate schedule
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train the net
    loss_per_epoch = []
    test_loss_per_epoch = []
    best_test_loss = float('inf')

    print("Starting training...")
    for epoch in range(n_epochs):
        # Training phase
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        loss_per_epoch.append(epoch_loss)

        # Validation phase
        net.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs.float())
                test_loss += criterion(outputs, labels.float()).item()

        test_loss = test_loss / len(testloader)
        test_loss_per_epoch.append(test_loss)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(net.state_dict(), 'best_model.pth')

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], '
                  f'Train Loss: {epoch_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}')

    print("Training finished!")

    # Load best model
    net.load_state_dict(torch.load('best_model.pth'))

    # Final evaluation
    net.eval()
    with torch.no_grad():
        train_loss = sum(loss.item() for inputs, labels in trainloader
                         for loss in [criterion(net(inputs.to(device).float()), labels.to(device).float())]) / len(
            trainloader)
        test_loss = sum(loss.item() for inputs, labels in testloader
                        for loss in [criterion(net(inputs.to(device).float()), labels.to(device).float())]) / len(
            testloader)

    print("\nFinal Results:")
    print(f"Training RMSE: {np.sqrt(train_loss):.4f}")
    print(f"Test RMSE: {np.sqrt(test_loss):.4f}")

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(loss_per_epoch, label='Training Loss')
    plt.plot(test_loss_per_epoch, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.show()


if __name__ == "__main__":
    import os
    import sys
    import argparse

    # By default, read csv file in the same directory as this script
    csv_file = os.path.join(sys.path[0], "Before_During_After_Exposure_Complete.csv")

    start_time = time.time()

    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", nargs="?", const=csv_file, default=csv_file,
                        help="Dataset file used for training")
    parser.add_argument("--epochs", "-e", type=int, nargs="?", default=100, help="Number of epochs to train")
    args = parser.parse_args()

    # Call the main function of the script
    train(args.file, args.epochs)

    end_time = time.time()

    print("Time:", end_time - start_time)