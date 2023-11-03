import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Adjusted Learning Rate
lr = 0.0005

# Initialize the model, loss function, and optimizer
model = BlackjackModel(input_size=len(train_dataset[0][0])).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

# Implementing learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Additional Metrics Tracking
losses = []          # Store losses over epochs
roc_aucs = []        # Store ROC AUCs over epochs
accuracies = []      # Store accuracies over epochs

# Training loop
num_epochs = 10
print_every = 1
for epoch in range(num_epochs):
    total_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Print the first batch of outputs and targets for debugging
        if i == 0 and epoch == 0:
            print("First batch outputs:", torch.sigmoid(model(inputs.float())))
            print("First batch targets:", targets)

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

    # Tracking Metrics
    model.eval()
    epoch_labels = []
    epoch_outputs = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            epoch_outputs.extend(torch.sigmoid(outputs).cpu().numpy())
            epoch_labels.extend(targets.cpu().numpy())
    
    auc_score = roc_auc_score(epoch_labels, epoch_outputs)
    epoch_accuracy = accuracy_score(epoch_labels, (np.array(epoch_outputs) >= 0.5).astype(int))
    
    losses.append(total_loss/len(train_loader))
    accuracies.append(epoch_accuracy)
    roc_aucs.append(auc_score)

# ... [rest of the code for evaluation and plotting remains unchanged]


# Plotting Loss over epochs
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Plotting ROC AUC over epochs
plt.plot(roc_aucs)
plt.title("ROC AUC over Epochs")
plt.xlabel("Epochs")
plt.ylabel("ROC AUC Score")
plt.show()

# Plotting Accuracy over epochs
plt.plot(accuracies)
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Bar plot for evaluation metrics
metrics = [precision, recall, f1]
metric_names = ['Precision', 'Recall', 'F1']

plt.bar(metric_names, metrics)
plt.title('Evaluation Metrics')
plt.ylim(0, 1)  # since these metrics range from 0 to 1
plt.show()
