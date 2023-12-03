# I am importing necessary libraries
import pandas as pd  # I'm using pandas to handle data frames.
import torch  # PyTorch is a library I'll use for building the neural network.
import torch.nn as nn  # nn is a module in PyTorch for building neural networks.
import torch.optim as optim  # optim is for optimization algorithms.
from torch.utils.data import Dataset, DataLoader  # These are for handling the dataset.
from sklearn.model_selection import train_test_split  # I will use these to split the dataset into training and testing sets.
from sklearn.preprocessing import StandardScaler  # This is for standardizing the dataset.
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix  # This is to calculate various performance metrics.
import seaborn as sns  # This is for better visualization of data.
import matplotlib.pyplot as plt  # This is for plotting graphs and charts.
import numpy as np  # This is for numerical operations in Python.

# I am checking if GPU is available for training, otherwise using CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This line will load and preprocess my blackjack data from a CSV file.
data = pd.read_csv('8_4_blackjack_training_data.csv')

# This will select the columns that will be used as features.
feature_columns = [col for col in data.columns if col not in ['Action Taken', 'Outcome']]

# I am standardizing the features using StandardScaler.
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

# I am setting my target variable for the model.
target_column = 'Action Taken'
X = data.drop(columns=target_column)
y = data[target_column]

# This will split the dataset into training and testing sets.
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# I am defining a custom class for my dataset.
class BlackjackDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        # This will return the number of samples in the dataset.
        return len(self.features)

    def __getitem__(self, idx):
        # I am retrieving the features and label of a specific sample.
        feature = self.features.iloc[idx].values
        label = self.labels.iloc[idx]
        return torch.tensor(feature), torch.tensor(label)

# Defining my neural network architecture.
class BlackjackModel(nn.Module):
    def __init__(self, input_size):
        super(BlackjackModel, self).__init__()
        # I am creating the first fully connected layer with input size to 64 units.
        self.fc1 = nn.Linear(input_size, 64)
        # I am creating the second fully connected layer with 64 input units and 32 output units.
        self.fc2 = nn.Linear(64, 32)
        # I am creating the third fully connected layer with 32 input units and 1 output unit.
        self.fc3 = nn.Linear(32, 1)
        # I am defining a dropout layer with a dropout rate of 60% to reduce overfitting.
        self.dropout = nn.Dropout(0.6)
        
        # I am initializing weights for the layers.
        # I am initializing weights of the first layer using Kaiming normal initialization, suitable for ReLU.
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        # I am also applying Kaiming normal initialization to the weights of the second layer.
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        # I am using Xavier normal initialization for the weights of the third layer, which is commonly used for output layers.
        nn.init.xavier_normal_(self.fc3.weight)

    # This is a forward pass through the network.
    def forward(self, x):
        # I am applying the ReLU activation function to the output of the first fully connected layer.
        x = torch.relu(self.fc1(x))
        # I am applying dropout after the first layer, which helps prevent overfitting.
        x = self.dropout(x)
        # I am applying the ReLU activation function again, but this time to the output of the second fully connected layer.
        x = torch.relu(self.fc2(x))
        # I am applying dropout once more after the second layer for the same reason as before.
        x = self.dropout(x)
        # Finally I am passing over the output through the third fully connected layer and returning the result.
        return self.fc3(x)

# I am creating an instance of the model.
model = BlackjackModel(input_size=train_features.shape[1]).to(device)

# This will set up the loss function and optimizer.
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# I am using this to prepare data loaders for training and testing.
batch_size = 32
# I am setting the batch size to 32. This means that 32 samples will be used to train the model at each iteration.
train_dataset = BlackjackDataset(train_features, train_labels)
# Here I am creating a dataset from the training features and labels using my custom BlackjackDataset class.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# DataLoader takes my train_dataset and returns batches of data. 'shuffle=True' means the data will be shuffled at every epoch.
test_dataset = BlackjackDataset(test_features, test_labels)
# Similarly, I'm creating a dataset for testing.
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initializing lists to track various metrics.
epoch_losses, epoch_accuracies, epoch_aucs, epoch_f1s, epoch_recalls, epoch_precisions = [], [], [], [], [], []

# Starting the training loop.
num_epochs = 200
for epoch in range(num_epochs):
    total_loss, correct, total = 0.0, 0, 0
    model.train()  # I am setting the model to training mode.
    for inputs, targets in train_loader:
        # This moves inputs and targets to the device.
        inputs, targets = inputs.to(device), targets.to(device)
        # I reset the gradients before forward pass.
        optimizer.zero_grad()
        # Start the forward pass through the model.
        outputs = model(inputs.float())
        # This wil reshape targets to match output dimensions.
        targets = targets.float().view(-1, 1)
        # This will help me figure out loss.
        loss = criterion(outputs, targets)
        # Performing backpropagation to help me figure out weight and bias impacts on loss.
        loss.backward()
        # This will update model parameters.
        optimizer.step()
        # Adding up total loss and correct predictions.
        total_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    # Calculating and storing metrics after each epoch so that I can keep track.
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct / total
    epoch_losses.append(epoch_loss)
    epoch_accuracies.append(epoch_accuracy)

    # Evaluating the model for additional metrics.
    model.eval()  # Iam setting the model to evaluation mode.
    with torch.no_grad():  # This tells PyTorch not to compute or store gradients which is important for efficiency during evaluation.
        all_targets, all_outputs = [], []  # Initializing lists to store all the true labels and the model's predictions.
        for inputs, targets in test_loader:  # This will loop through the batches of the test dataset.
            inputs, targets = inputs.to(device), targets.to(device)  # I am moving inputs and targets to the current device.
            outputs = model(inputs.float())  # I am getting the model's predictions for the current batch.
            all_targets.extend(targets.cpu().numpy())  # Adding the true labels to the all_targets list.
            all_outputs.extend(outputs.cpu().numpy())  # Adding the model's predictions to the all_outputs list.

        # This will calculate various performance metrics based on the accumulated true labels and predictions.
        epoch_auc = roc_auc_score(all_targets, all_outputs)  # Calculating the Area Under the Receiver Operating Characteristic Curve (ROC AUC).
        epoch_f1 = f1_score(all_targets, (np.array(all_outputs) >= 0.5).astype(int))  # Calculating the F1 score, a balance between precision and recall.
        epoch_recall = recall_score(all_targets, (np.array(all_outputs) >= 0.5).astype(int))  # Calculating the recall, the ability of the classifier to find all the positive samples.
        epoch_precision = precision_score(all_targets, (np.array(all_outputs) >= 0.5).astype(int))  # Using this to calculate the precision score.

        epoch_aucs.append(epoch_auc) # I am adding the ROC AUC score of the current epoch to the epoch_aucs list. This keeps track of how this score changes over each epoch.
        epoch_f1s.append(epoch_f1) # I am appending the F1 score from this epoch to the epoch_f1s list. It's a way to see how the balance between precision and recall evolves as training progresses.
        epoch_recalls.append(epoch_recall) # Here I am adding the recall score for this epoch to the epoch_recalls list. This helps me monitor the model's ability to correctly identify all positive samples over time.
        epoch_precisions.append(epoch_precision) # I am appending the precision score for this epoch to the epoch_precisions list. This shows how the accuracy of the model's positive predictions changes throughout the epochs.

    # Printing metrics every 5 epochs for monitoring progress.
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, AUC: {epoch_auc:.4f}")

# Plotting Loss and Accuracy over the epochs.
epochs = range(1, num_epochs + 1)  # I'm creating a range of numbers from 1 to the total number of epochs for the x-axis in my plots.
plt.figure(figsize=(12, 5))  # Setting the size of the figure for my plots.

# Plotting the Loss over Epochs
plt.subplot(1, 2, 1)  # Creating a subplot in a 1 row, 2 columns format, and this is the first plot.
plt.plot(epochs, epoch_losses, label='Loss')  # Plotting the loss values over each epoch.
plt.title('Loss over Epochs')  # Giving a title to this plot.
plt.xlabel('Epoch')  # Labeling the x-axis as 'Epoch'.
plt.ylabel('Loss')  # Labeling the y-axis as 'Loss'.
plt.legend()  # Adding a legend to the plot.

# Plotting the Accuracy over Epochs
plt.subplot(1, 2, 2)  # This is the second plot in my subplot.
plt.plot(epochs, epoch_accuracies, label='Accuracy', color='orange')  # Plotting the accuracy values over each epoch, in orange color.
plt.title('Accuracy over Epochs')  # Giving a title to this plot.
plt.xlabel('Epoch')  # Labeling the x-axis.
plt.ylabel('Accuracy')  # Labeling the y-axis.
plt.legend()  # Adding a legend.
plt.show()  # Displaying the plots.

# Plotting ROC AUC over the epochs.
plt.figure(figsize=(6, 5))  # Setting the size of the ROC AUC plot.
plt.plot(epochs, epoch_aucs, label='ROC AUC', color='green')  # Plotting ROC AUC values over epochs, in green color.
plt.title('ROC AUC over Epochs')  # Giving a title to the plot.
plt.xlabel('Epoch')  # Labeling the x-axis.
plt.ylabel('ROC AUC')  # Labeling the y-axis.
plt.legend()  # Adding a legend.
plt.show()  # Displaying the plot.

# Plotting F1 Score, Recall, and Precision in a bar plot.
plt.figure(figsize=(8, 5))  # Setting the size of the F1, Recall, Precision plot.
width = 0.2  # Setting the width for each bar in the bar plot.
# Creating a bar plot for F1 Scores, offset by 'width' to the left.
plt.bar(np.array(epochs) - width, epoch_f1s, width=width, label='F1 Score')  
# Creating a bar plot for Recall, aligned with the epoch numbers.
plt.bar(epochs, epoch_recalls, width=width, label='Recall')  
# Creating a bar plot for Precision, offset by 'width' to the right.
plt.bar(np.array(epochs) + width, epoch_precisions, width=width, label='Precision')  
plt.title('F1, Recall, Precision over Epochs')  # Giving a title to the plot.
plt.xlabel('Epoch')  # Labeling the x-axis.
plt.ylabel('Score')  # Labeling the y-axis.
plt.legend()  # Adding a legend.
plt.show()  # Displaying the plot.

