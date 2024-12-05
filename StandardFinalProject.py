# Imports
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
import matplotlib.pyplot as plt

# Standard model architecture
class CNNMetricsStatistics(nn.Module):
    def __init__(self, input_dimension, output_dimension, dropout_rate=0.2):
        super(CNNMetricsStatistics, self).__init__()

        self.input_dimension = input_dimension

        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate)
        )
        print("conv_layers: ", self.conv_layers)  # Display convolutional layer dimensions

        # Calculate flattened features size
        self.flatten_size = 32 * input_dimension

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dimension)
        )
        print("fc_layers: ", self.fc_layers)  # Display fully connected layer dimensions

    def forward(self, x):
        """
        Forward pass through network
        """
        # Reshape input for 1D convolution
        x = x.unsqueeze(1)  # Channel dimension
        x = self.conv_layers(x)  # Pass through convolutional layers
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc_layers(x)  # Pass through fully connected layers

        return x

# Helper functions
def time_to_hour(time_str):
    """Convert time from string to integer"""
    time_parts = str(time_str).split(":")  # Convert from HH:MM to HH
    return int(time_parts[0])

def one_hot_encoding(column):
    """Perform one-hot encoding on non-numerical columns"""
    unique_values = sorted(column.unique())
    oh_encoding = np.zeros((len(column), len(unique_values)))

    # Assign values
    val_idx = {value: idx for idx, value in enumerate(unique_values)}
    for i, value in enumerate(column):
        oh_encoding[i, val_idx[value]] = 1

    return oh_encoding, unique_values

def prepare_data(file_path):
    df = pd.read_csv(file_path)  # Read from .csv file

    day_one_hot, day_categories = one_hot_encoding(df["day_of_week"])  # Perform one hot encoding

    # StudentsPerformance
    # gender_one_hot, gender_categories = one_hot_encoding(df["gender"])
    # race_one_hot, race_categories = one_hot_encoding(df["race/ethnicity"])
    # parent_one_hot, parent_categories = one_hot_encoding(df["parental level of education"])
    # lunch_one_hot, lunch_categories = one_hot_encoding(df["lunch"])
    # test_one_hot, test_categories = one_hot_encoding(df["test preparation course"])

    time_hour = df["time_of_day"].apply(time_to_hour).values.reshape(-1, 1)  # Convert time_of_day to hours

    feature_columns = ["splpk", "splrms", "dissim", "impulsivity", "peakcount"]  # Columns used to predict time since exposure
    numerical_features = df[feature_columns].values

    # StudentsPerformance
    # feature_columns = [gender_one_hot, race_one_hot, parent_one_hot, lunch_one_hot, test_one_hot]

    # Combine all features
    x = np.hstack([numerical_features, day_one_hot, time_hour])
    y = df["time_since_exposure"].values.reshape(-1, 1)  # Other columns used to predict time since exposure

    # StudentsPerformance
    # x = np.hstack([gender_one_hot, race_one_hot, parent_one_hot, lunch_one_hot, test_one_hot])
    # y = df["math score"].values.reshape(-1, 1)

    # Scale features
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Scale features
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y)

    # Convert to tensors
    x_tensor = torch.FloatTensor(x_scaled)
    y_tensor = torch.FloatTensor(y_scaled)

    return x_tensor, y_tensor, scaler_y

# Split train/test
def train_test_split(x, y, test_size=0.2, random_seed=42):
    torch.manual_seed(random_seed)  # Set random seed for reproducibility
    n_samples = x.shape[0]  # Set number of samples
    indices = torch.randperm(n_samples)  # Set random indices
    split_idx = int(n_samples * (1 - test_size))  # Determine indices for train/test

    train_indices = indices[:split_idx]  # Training data
    test_indices = indices[split_idx:]  # Testing data

    # Training/testing subsets
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return x_train, x_test, y_train, y_test

# Training
def train_model(model, x_train, y_train, x_test, y_test, criterion, optimizer, num_epochs):
    train_losses = []  # Training loss list
    test_losses = []  # Testing loss list

    # Training/testing
    for epoch in range(num_epochs):
        model.train()  # Model will be trained
        optimizer.zero_grad()  # Will not calculate gradients
        outputs = model(x_train)  # Model training data
        loss = criterion(outputs, y_train)  # Calculate training loss
        loss.backward()  # Back propagate
        optimizer.step()  # Update parameters

        train_losses.append(loss.item())  # Append training loss

        model.eval()  # Model will be evaluated
        with torch.no_grad():  # Will not calculate gradients
            test_outputs = model(x_test)  # Model test data
            test_loss = criterion(test_outputs, y_test)  # Calculate loss
            test_losses.append(test_loss.item())  # Append current loss

        if (epoch + 1) % 100 == 0:  # Display sample results
            print("Epoch", epoch + 1, "/", num_epochs, "\tTraining Loss:", train_losses[-1], "\tTest Loss:", test_loss)

    return train_losses, test_losses

# Model parameters
output_dimension = 1  # One target value
dropout_rate = 0.6  # Dropout rate
learning_rate = 0.01  # Learning rate
num_epochs = 1500  # Number of epochs to iterate through

# Prepare data
x, y, scaler_y = prepare_data("Before_During_After_Exposure_0601_0719.csv")

# StudentsPerformance
# x, y, scaler_y = prepare_data("StudentsPerformance.csv")

input_dimension = x.shape[1]  # Number of input features
x_train, x_test, y_train, y_test = train_test_split(x, y)  # Perform train/test split

cnn_model = CNNMetricsStatistics(input_dimension, output_dimension, dropout_rate)  # Initialize CNN model

criterion = nn.MSELoss()  # Loss function
optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=learning_rate, weight_decay=0.01)  # Initialize optimizer

train_losses, test_losses = train_model(cnn_model, x_train, y_train, x_test, y_test, criterion, optimizer, num_epochs)  # Initialize optimizer

# Confusion matrix
def confusion_matrix(y_true, y_pred, num_classes):
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)  # Initialize empty matrix

    for true, pred in zip(y_true, y_pred):  # Iterate through labels
        conf_mat[true, pred] += 1  # Increment value in appropriate location

    return conf_mat

# Use one threshold for evaluation
threshold = 0.1
y_true = (y_test.numpy() > threshold).astype(int).flatten()  # Convert ground truth to binary
y_pred = (cnn_model(x_test).detach().numpy() > threshold).astype(int).flatten()  # Convert predicted to binary
num_classes = 2  # Binary classification
conf_mat = confusion_matrix(y_true, y_pred, num_classes)  # Create confusion matrix
print("Confusion Matrix:\n", conf_mat)  # Display confusion matrix

# ARIMA analysis
def arima_analysis(predictions, order=(1, 1, 1)):
    """
    Autoregressive integrated moving average
    Predicts future values
    """
    arima_model = ARIMA(predictions, order=order)  # Initialize model
    fitted_model = arima_model.fit()  # Fit model
    forecast = fitted_model.predict(len(predictions))  # Make prediction
    return arima_model, forecast

def plot_arima_forecast(model, x_test):
    """Plot ARIMA forecast"""
    # Get the predictions from the trained model
    predictions = model(x_test).detach().numpy().flatten()  # Get trained model predictions

    _, forecast = arima_analysis(predictions)  # Perform ARIMA forecast on predictions

    # Display data
    plt.figure(figsize=(15, 6))
    plt.plot(predictions, label="Original Predictions", color="blue")

    # Plot ARIMA forecast
    plt.title("ARIMA Forecast")
    plt.xlabel("Time Steps")
    plt.ylabel("Predicted Values")
    plt.show()

predictions = cnn_model(x_test).detach().numpy().reshape(-1, 1)  # Ensure data is 1D
_, forecast = arima_analysis(predictions)  # Make prediction using ARIMA

print("Forecast:\n", forecast)  # Display forecast

# ANOVA Analysis
before_exposure = predictions[:144]  # Before first exposure
during_exposure = predictions[144:375]  # During on/off exposure
after_exposure = predictions[375:]  # After final exposure

def anova_analysis(groups):
    """
    Performs analysis of variance/displays results
    """
    num_groups = len(groups)  # Number of groups
    total_samples = sum(len(group) for group in groups)  # Number of samples
    overall_mean = np.mean(np.concatenate(groups))  # Overall mean
    ss_between = sum(len(group) * (np.mean(group) - overall_mean) ** 2 for group in groups)  # Sum of squares between groups
    ss_within = sum(np.sum((group - np.mean(group)) ** 2) for group in groups)  # Sum of squares within groups
    ss_total = ss_between + ss_within  # Sum of squares total

    # Degrees of freedom
    df_between = num_groups - 1
    df_within = total_samples - num_groups
    df_total = total_samples - 1

    # Mean square
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_value = ms_between / ms_within  # F-value
    p_value = stats.f.sf(f_value, df_between, df_within)  # P-value
    f_crit = stats.f.ppf(0.95, df_between, df_within)  # F critical value

    # Create a formatted output table
    data = [
        ["Between Groups\t\t", ss_between, df_between, ms_between, f_value, p_value, f_crit],
        ["Within Groups", ss_within, df_within, ms_within, "", "", ""],  # "" removes NaN
        ["Total", ss_total, df_total, "", "", "", ""],  # "" removes NaN
    ]

    bg = data[0]  # Between groups
    wg = data[1]  # Within groups
    t = data[2]  # Total

    return bg, wg, t

between_groups, within_groups, total = anova_analysis([before_exposure, during_exposure, after_exposure])  # Perform analysis

# Display ANOVA results
print("ANOVA Analysis")
print("Source of Variation\t SS\t\t\t df\t\t\t MS\t\t\t F\t\t\t P-value\t\t F crit")
print(between_groups[0], round(between_groups[1], 4), "\t", round(between_groups[2], 4), "\t\t\t", round(between_groups[3], 4),
      "\t\t", round(between_groups[4], 4), "\t", round(between_groups[5], 4), "\t\t", round(between_groups[6], 4))
print(within_groups[0], "\t\t", round(within_groups[1], 4), "\t", round(within_groups[2], 4), "\t\t", round(within_groups[3], 4))
print(total[0], "\t\t\t\t", round(total[1], 4), "\t", round(total[2], 4))

# ANOVA trained predictions
trained_predictions = cnn_model(x_train).detach().numpy().reshape(-1, 1)
arima_model, forecast = arima_analysis(trained_predictions)
forecast_predictions = forecast

# Determine if large difference in mean is present
f_stat, p_value = stats.f_oneway(before_exposure, during_exposure, after_exposure)
print("ANOVA F:", f_stat)
print("ANOVA p-value:", p_value)

# Paired t-test for exposure periods
def t_test_exposure_periods(group1, group2):
    """
    Performs t-test between during exposure and after exposure
    """

    # Ensure equal lengths of groups
    min_length = min(len(group1), len(group2))
    group1 = group1[:min_length]
    group2 = group2[:min_length]

    t_stat, p_value = stats.ttest_rel(group1, group2)  # Perform t-test between groups
    return t_stat, p_value

# Significance test: T-test
t_stat, p_value = t_test_exposure_periods(during_exposure, after_exposure)
print("T-Test Exposure Periods (During vs After Exposure):\nT-Statistic:", t_stat, "\tP-Value:", p_value)

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    """
    Determine accuracy of predictions
    """
    correct = (y_true == y_pred).mean()
    return correct / len(y_true)

# Calculate MSE
def calculate_mse(y_true, y_pred):
    """
    Determine mean square error
    """
    error = (y_true - y_pred) ** 2
    return np.mean(error)

# Calculate F1 score
def calculate_f1_score(y_true, y_pred):
    """
    Measure model performance with precision/recall
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

y_pred_binary = (cnn_model(x_test).detach().numpy() > threshold).astype(int).flatten()  # Predicted values
y_true_binary = y_true  # Real values

# Calculate evaluation metrics
accuracy = calculate_accuracy(y_true_binary, y_pred_binary)
mse = calculate_mse(y_test.numpy().flatten(), cnn_model(x_test).detach().numpy().flatten())

# Binarize labels (if target should be binary)
y_true_test = [1 if label >= 0.5 else 0 for label in y_true_binary]
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_binary]

f1 = calculate_f1_score(np.array(y_true_test), np.array(y_pred_binary))  # Calculate F1 score
test_precision = precision_score(y_true_test, y_pred_binary)  # Calculate precision score

# Display performance results
print("Accuracy:", accuracy)
print("MSE:", mse)
print("F1:", f1)
print("Precision:", test_precision)

# Determine significance of p_value
if p_value < 0.05:
    print("There is a statistical significance")
else:
    print("There is no statistical significance")

# ANOVA bar chart
plt.figure(figsize=(15, 6))
exposure_periods = ["Pre-Exposure", "During-Exposure", "Post-Exposure"]
means = [np.mean(before_exposure), np.mean(during_exposure), np.mean(after_exposure)]
plt.bar(exposure_periods, means, color=["blue", "orange", "green"])
plt.title("Mean Values Across Exposure Periods (ANOVA)")
plt.ylabel("Mean Predicted Value")
plt.show()

plot_arima_forecast(cnn_model, x_test)  # Display ARIMA
