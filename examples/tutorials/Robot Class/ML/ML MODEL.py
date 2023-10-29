import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Step 1: Load and preprocess the data
data = pd.read_csv('OmegaII_6_6.csv')
# Preprocess categorical variables (e.g., one-hot encoding for dealer_card)
data = pd.get_dummies(data, columns=['dealer_card'])

# Convert all data to float32 (to match the default data type of model weights)
data = data.astype('float32')

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Step 2: Create a custom PyTorch Dataset
class BlackjackDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, 1:-1].values
        target = self.data.iloc[idx, -1]
        return torch.tensor(features), torch.tensor(target)

# Create DataLoader instances
batch_size = 32
train_dataset = BlackjackDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = BlackjackDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Step 3: Design the neural network model
class BlackjackModel(nn.Module):
    def __init__(self, input_size):
        super(BlackjackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Remove sigmoid activation
        x = torch.round(torch.sigmoid(x))  # Round to the nearest integer (0 or 1)
        return x
    
# Initialize the model, loss function, and optimizer
model = BlackjackModel(input_size=len(train_dataset[0][0]))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (you need to complete this part)
num_epochs = 100
for inputs, targets in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs.float())  # Convert inputs to float32

    # Reshape the target tensor to match the output shape
    targets = targets.view(-1, 1)  # Reshape to have shape [batch_size, 1]

    loss = criterion(outputs, targets)
    print(loss.item())
    loss.backward()
    optimizer.step()

# Evaluation (you need to complete this part)
# Initialize variables to accumulate evaluation metrics
true_labels = []
predicted_labels = []

# Set the model to evaluation mode
model.eval()

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.float())  # Convert inputs to float32
        predicted = (outputs >= 0.5).view(-1)  # Convert model output to binary (0 or 1)

        # Accumulate true labels and predicted labels
        true_labels.extend(targets.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

# Convert accumulated lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate precision, recall, and F1-score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(predicted_labels)