import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = pd.read_csv('OmegaII_6_6.csv')
data = pd.get_dummies(data, columns=['dealer_card'])

# Normalize the data
scaler = StandardScaler()
data.iloc[:, 1:-1] = scaler.fit_transform(data.iloc[:, 1:-1])

data = data.astype('float32')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Custom PyTorch Dataset
class BlackjackDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx, 1:-1].values
        target = self.data.iloc[idx, -1]
        return torch.tensor(features), torch.tensor(target)

# DataLoader instances
batch_size = 32
train_dataset = BlackjackDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = BlackjackDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Design of the enhanced neural network model
class BlackjackModel(nn.Module):
    def __init__(self, input_size):
        super(BlackjackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

        # Weight Initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def inference(model, inputs):
    outputs = model(inputs)
    return torch.round(torch.sigmoid(outputs))

# Initialize the model, loss function, and optimizer
model = BlackjackModel(input_size=len(train_dataset[0][0])).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training loop
num_epochs = 10
print_every = 1
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        targets = targets.view(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {total_loss/len(train_loader):.4f}")

# Evaluation
true_labels = []
predicted_labels = []
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = inference(model, inputs.float())
# I am using the following line below to get the predicted labels
# but I am displaying it as tru and false in my print out at the end.
        predicted = (outputs >= 0.5).view(-1)
        true_labels.extend(targets.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
# I am using the following line below to get the predicted labels and print them out as results.
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
print(predicted_labels)
